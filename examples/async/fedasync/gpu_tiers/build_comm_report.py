#!/usr/bin/env python3
"""Render the comm-latency sweep (BRIEF_comm_sweep.md) into a self-contained HTML page.

Reads figures/comm/{tta_comm,steering_comm}.csv (written by comm_sweep.py) and
renders inline SVG figures — same visual kit as report_hsweep_mix.html, so this
reads as the next chapter of that report rather than a different document.

Safe to run mid-sweep: cells with no .done marker are drawn as open circles and
labelled "running" rather than silently plotted as finished data.

Usage:
    python3 comm_sweep.py --plots      # writes figures/comm/*.csv first
    python3 build_comm_report.py       # then this
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).parent
FIG = HERE / "figures" / "comm"

TIERS = ["xavier", "orin", "thor"]
ARMS = ["gpu", "cpu", "baseline"]
ARM_LABEL = {"gpu": "gpu", "cpu": "cpu", "baseline": "baseline"}
# Plot order: inf (control) on the left, then falling bandwidth.
LABELS = ["inf", "100", "50", "20", "10"]
LABEL_TEXT = {"inf": "∞", "100": "100", "50": "50", "20": "20", "10": "10"}
C_OF = {"inf": 0.0, "100": 3.4152, "50": 6.8304, "20": 17.076, "10": 34.152}


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


TTA = load_csv(FIG / "tta_comm.csv")
STEER = load_csv(FIG / "steering_comm.csv")

DONE_CELLS = {(r["arm"], r["label"]) for r in TTA if r["done"] == "True"}
ALL_CELLS = {(a, l) for a in ARMS for l in LABELS} | {("gpu", "10nc")}
STARTED_CELLS = {(r["arm"], r["label"]) for r in STEER}
RUNNING_CELLS = STARTED_CELLS - DONE_CELLS
NOT_STARTED = ALL_CELLS - STARTED_CELLS
N_DONE = len({(a, l) for (a, l) in DONE_CELLS if l != "10nc"})
N_TOTAL = 11  # BRIEF § 3: 4x2 (gpu/cpu) + 2 baseline + 1 ablation


def steer_row(arm: str, label: str, tier: str) -> dict | None:
    for r in STEER:
        if r["arm"] == arm and r["label"] == label and r["tier"] == tier:
            return r
    return None


def pref_of(arm: str, label: str) -> float | None:
    r = steer_row(arm, label, "xavier")
    if r is None or r["pref_xavier_over_thor"] in ("", "nan", None):
        return None
    return float(r["pref_xavier_over_thor"])


def tta_of(arm: str, label: str, target: float) -> dict | None:
    for r in TTA:
        if r["arm"] == arm and r["label"] == label and float(r["target"]) == target:
            return r
    return None


# --------------------------------------------------------------------------- #
# tiny SVG chart kit — same conventions as build_report.py
# --------------------------------------------------------------------------- #

def axes(x0, y0, w, h, ymax, yticks, ylab, xs, xlabels, ymin=0.0):
    out = []
    for t in yticks:
        y = y0 + h - (t - ymin) / (ymax - ymin) * h
        out.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{x0-8}" y="{y+3.5:.1f}" text-anchor="end">{t:g}</text>')
    for x, lab in zip(xs, xlabels):
        out.append(f'<text class="tick" x="{x:.1f}" y="{y0+h+18}" text-anchor="middle">{lab}</text>')
    out.append(f'<line class="axis" x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}"/>')
    out.append(f'<text class="axlab" x="{x0}" y="{y0-14}" text-anchor="start">{ylab}</text>')
    return "\n".join(out)


def polyline(pts, cls, dash=None):
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dd = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline class="{cls}" points="{d}"{dd}/>'


def marker(x, y, cls, running=False):
    if running:
        return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="var(--surface)" '
                f'class="{cls}" stroke-width="2"/>')
    return f'<circle class="mk {cls}" cx="{x:.1f}" cy="{y:.1f}" r="4.5"/>'


W, H = 620, 340
X0, Y0, PW, PH = 58, 40, 520, 234
XS = [X0 + PW * i / (len(LABELS) - 1) for i in range(len(LABELS))]


def status_of(arm: str, label: str) -> str:
    if (arm, label) in DONE_CELLS:
        return "done"
    if (arm, label) in RUNNING_CELLS:
        return "running"
    return "pending"


# --------------------------------------------------------------------------- #
# Figure 1 — dispatch share vs bandwidth (gpu), the headline
# --------------------------------------------------------------------------- #

def fig_dispatch() -> str:
    ymax = 60.0
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Percent of dispatches per vehicle tier versus uplink bandwidth, gpu arm">']
    s.append(axes(X0, Y0, PW, PH, ymax, [0, 20, 40, 60], "dispatch share %",
                  XS, [LABEL_TEXT[l] for l in LABELS]))
    s.append(f'<text class="sub" x="{X0-8}" y="{Y0+PH+34}" text-anchor="end">B (Mbps)</text>')
    for tier in TIERS:
        pts, mks = [], []
        for x, label in zip(XS, LABELS):
            r = steer_row("gpu", label, tier)
            if r is None:
                continue
            y = Y0 + PH - float(r["dispatch_share"]) * 100 / ymax * PH
            pts.append((x, y))
            mks.append(marker(x, y, tier, running=(("gpu", label) in RUNNING_CELLS)))
            pool_y = Y0 + PH - float(r["pool_share"]) * 100 / ymax * PH
        if pts:
            s.append(polyline(pts, f"ln {tier}"))
            s.extend(mks)
            lx, ly = pts[-1]
            s.append(f'<text class="dl {tier}" x="{lx+10:.1f}" y="{ly+4:.1f}">{tier}</text>')
    # pool-share reference lines (dashed), one per tier, constant at 50/35/15
    for tier, share in (("xavier", 50), ("orin", 35), ("thor", 15)):
        y = Y0 + PH - share / ymax * PH
        s.append(f'<line class="pool" x1="{X0}" y1="{y:.1f}" x2="{X0+PW}" y2="{y:.1f}"/>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------- #
# Figure 2 — pref vs bandwidth (gpu + cpu), the mechanism
# --------------------------------------------------------------------------- #

def fig_pref() -> str:
    ymax, ymin = 3.6, 1.0
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Predicted speed-term preference for thor over xavier versus uplink bandwidth">']
    s.append(axes(X0, Y0, PW, PH, ymax, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                  "pref  (xavier-over-thor speed term)", XS, [LABEL_TEXT[l] for l in LABELS], ymin))
    s.append(f'<text class="sub" x="{X0-8}" y="{Y0+PH+34}" text-anchor="end">B (Mbps)</text>')
    y1 = Y0 + PH - (1.0 - ymin) / (ymax - ymin) * PH
    s.append(f'<line class="unity" x1="{X0}" y1="{y1:.1f}" x2="{X0+PW}" y2="{y1:.1f}"/>')
    for arm in ("gpu", "cpu"):
        pts, mks = [], []
        for x, label in zip(XS, LABELS):
            v = pref_of(arm, label)
            if v is None:
                continue
            y = Y0 + PH - (v - ymin) / (ymax - ymin) * PH
            pts.append((x, y))
            mks.append(marker(x, y, arm, running=((arm, label) in RUNNING_CELLS)))
        if pts:
            s.append(polyline(pts, f"ln {arm}"))
            s.extend(mks)
            lx, ly = pts[-1]
            s.append(f'<text class="dl {arm}" x="{lx+10:.1f}" y="{ly+4:.1f}">{arm}</text>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------- #
# Figure 3 — hours to 55% vs bandwidth, all arms
# --------------------------------------------------------------------------- #

def fig_tta() -> str:
    ymax = 3.2
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Simulated hours to 55 percent accuracy versus uplink bandwidth">']
    s.append(axes(X0, Y0, PW, PH, ymax, [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                  "simulated hours to 55%", XS, [LABEL_TEXT[l] for l in LABELS]))
    s.append(f'<text class="sub" x="{X0-8}" y="{Y0+PH+34}" text-anchor="end">B (Mbps)</text>')
    for arm in ARMS:
        pts, mks = [], []
        for x, label in zip(XS, LABELS):
            r = tta_of(arm, label, 55.0)
            if r is None or r["hours"] in ("", None):
                continue
            v = float(r["hours"])
            y = Y0 + PH - v / ymax * PH
            pts.append((x, y))
            mks.append(marker(x, y, arm, running=((arm, label) in RUNNING_CELLS)))
        if pts:
            s.append(polyline(pts, f"ln {arm}"))
            s.extend(mks)
            lx, ly = pts[-1]
            s.append(f'<text class="dl {arm}" x="{lx+10:.1f}" y="{ly+4:.1f}">{arm}</text>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #

def table_rows() -> str:
    rows = []
    for label in LABELS:
        cells = []
        for arm in ARMS:
            gp = pref_of(arm, label) if arm != "baseline" else None
            share = steer_row(arm, label, "xavier")
            xav = f'{float(share["dispatch_share"])*100:.1f}%' if share else "—"
            st = status_of(arm, label)
            badge = {"done": "", "running": ' <span class="run">running</span>',
                     "pending": ' <span class="pend">queued</span>'}[st]
            pref_s = f"{gp:.2f}" if gp is not None else "—"
            cells.append(f"<td>{xav} <span class=\"pm\">/</span> {pref_s}{badge}</td>")
        c_s = f"{C_OF[label]:.2f}" if C_OF[label] else "0"
        rows.append(
            f'<tr><th>{"∞" if label == "inf" else label}</th><td>{c_s}</td>{"".join(cells)}</tr>'
        )
    return "\n".join(rows)


def status_banner() -> str:
    running = sorted(f"{a}/{l}" for a, l in RUNNING_CELLS)
    pending = sorted(f"{a}/{l}" for a, l in NOT_STARTED)
    return f"""
<div class="flag">
  <h3>In progress — {N_DONE} / {N_TOTAL} runs finished</h3>
  <p>Running now: {esc(', '.join(running)) or 'none'}.
  Queued: {esc(', '.join(pending)) or 'none'}.
  Figures below plot every finished cell as a filled marker and every running
  cell as a hollow one; queued cells are simply absent. This page regenerates
  from <code>comm_sweep.py --plots</code> + <code>build_comm_report.py</code> —
  rerun both once the sweep finishes for the final version.</p>
</div>"""


STYLE = (HERE / "report_hsweep_mix.html").read_text()
STYLE = STYLE[STYLE.index("<style>"):STYLE.index("</style>") + len("</style>")]
STYLE = STYLE.replace(
    "</style>",
    "var{font-style:italic;color:var(--ink)}\n"
    ".run{font:600 .66rem/1 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--accent);"
    "background:color-mix(in srgb, var(--accent) 14%, transparent);"
    "padding:1px 6px;border-radius:10px;margin-left:4px}\n"
    ".pend{font:600 .66rem/1 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink-3);"
    "background:var(--surface-2);padding:1px 6px;border-radius:10px;margin-left:4px}\n"
    "</style>",
)

PAGE = f"""<title>Comm-Latency Sweep</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
{STYLE}
<div class="wrap stack-lg">

<header class="head">
  <span class="eyebrow">Async federated learning · vehicle GPU tiers · third sweep</span>
  <h1>Does bandwidth dilute the speed signal?</h1>
  <p class="lede measure">Fleet pinned at 50/35/15, tier-speed gap pinned at G=30.
  Only the uplink bandwidth <var>B</var> moves, adding a transfer cost
  <var>C&nbsp;=&nbsp;D/B</var> identically to belief and reality. As <var>C</var>
  grows it pushes every tier's training time toward the same additive floor, so
  the GPU-TOPS speed signal should be <em>diluted, not degraded</em> — this sweep
  measures where that happens.</p>
  <div class="meta">
    <span><b>Target</b> 55% top-1, CIFAR-10 / ResNet-18</span>
    <span><b>Arms</b> gpu · cpu · baseline</span>
    <span><b>t_round_model</b> queue</span>
    <span><b>B=∞ control</b> reused from the fleet-mix sweep</span>
  </div>
</header>

{status_banner()}

<section class="stack">
  <div class="sec-head">
    <span class="eyebrow">Figure 1 — the headline</span>
    <h2>Which vehicle tier gets picked, as bandwidth falls</h2>
    <p class="fig-sub measure">gpu arm, percent of dispatches per tier. Dashed
    lines are the pool's own 50/35/15. Predicted: xavier's share climbs back
    toward the pool share as <var>B</var> falls (BRIEF § 5.2) — read straight off
    the dispatch log, no evaluation noise.</p>
  </div>
  <div class="card">
    <div class="plot">{fig_dispatch()}</div>
    <div class="legend">
      <span><i class="sw xavier"></i>xavier (slowest, 20 s)</span>
      <span><i class="sw orin"></i>orin (3.65 s)</span>
      <span><i class="sw thor"></i>thor (fastest, 0.67 s)</span>
      <span><i class="sw dash"></i>pool share</span>
      <span>hollow marker = run still in progress</span>
    </div>
  </div>
</section>

<section class="pair">
  <div class="stack">
    <div class="sec-head">
      <span class="eyebrow">Figure 2 — the mechanism</span>
      <h2>Predicted speed-term preference</h2>
      <p class="fig-sub measure"><var>pref</var> = the speed term's ratio favouring
      thor over xavier, computed directly from <code>gpu_tier_sleep</code> against
      each cell's own config. Pre-registered: 3.42 → 2.12 → 1.75 → 1.39 → 1.22
      for gpu (BRIEF § 5).</p>
    </div>
    <div class="card">
      <div class="plot">{fig_pref()}</div>
      <div class="legend">
        <span><i class="sw gpu"></i>gpu</span>
        <span><i class="sw cpu"></i>cpu</span>
        <span><i class="sw" style="border-top:1.5px dashed var(--ink-3);height:0"></i>pref = 1 (no signal)</span>
      </div>
    </div>
  </div>
  <div class="stack">
    <div class="sec-head">
      <span class="eyebrow">Figure 3 — end-to-end (provisional)</span>
      <h2>Time to 55% accuracy</h2>
      <p class="fig-sub measure">Two cells (cpu at B=50, B=10) are still running
      and have not yet crossed 55% within their logged rounds so far — their
      points are omitted here, not zero.</p>
    </div>
    <div class="card">
      <div class="plot">{fig_tta()}</div>
      <div class="legend">
        <span><i class="sw gpu"></i>gpu</span>
        <span><i class="sw cpu"></i>cpu</span>
        <span><i class="sw baseline"></i>baseline</span>
      </div>
    </div>
  </div>
</section>

<section class="stack">
  <div class="sec-head">
    <span class="eyebrow">Data table</span>
    <h2>xavier dispatch share and pref, per bandwidth</h2>
  </div>
  <div class="tw">
    <table>
      <caption>Each cell: xavier dispatch share % / predicted pref. Pool share is 50% xavier.</caption>
      <thead><tr><th>B (Mbps)</th><th>C (s)</th><th>gpu</th><th>cpu</th><th>baseline</th></tr></thead>
      <tbody>
        {table_rows()}
      </tbody>
    </table>
  </div>
</section>

<section class="stack">
  <details>
    <summary>Reading notes</summary>
    <div class="stack" style="padding-top:12px">
      <p class="note measure">The gpu arm's full B-axis (∞/100/50/20/10) is
      already complete and matches the pre-registered <var>pref</var> table
      exactly, including the required "must reproduce 3.42 at C=0" sanity check.
      cpu is complete at ∞/100/20 and running at 50/10. baseline (sanity anchor
      at B=20/10) and the belief-vs-reality ablation (gpu, B=10,
      <code>comm_in_predictor=false</code>) have not started yet.</p>
      <p class="note measure">Time-to-target figures carry the usual ~10%
      evaluation-noise floor (HANDOFF § 4.13) — a gpu-vs-cpu ratio under ~1.15×
      is not a resolvable result on its own. The dispatch-share and
      <var>pref</var> figures do not carry this limit.</p>
    </div>
  </details>
</section>

</div>
"""

if __name__ == "__main__":
    out = HERE / "report_comm_sweep.html"
    out.write_text(PAGE)
    print(f"wrote {out}")
