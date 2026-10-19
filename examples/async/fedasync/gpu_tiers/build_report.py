#!/usr/bin/env python3
"""Render report_data.json into a self-contained HTML report.

All four figures are hand-built inline SVG so they inherit the page's theme
tokens (no canvas, no external chart library — the Artifact CSP blocks both).
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
D = json.loads((HERE / "report_data.json").read_text())

ARMS = ["gpu", "cpu", "baseline"]
ARM_LABEL = {"gpu": "gpu", "cpu": "cpu", "baseline": "baseline"}
TIERS = ["xavier", "orin", "thor"]
GAPS = ["01", "10", "20", "30", "40"]
GAP_LABEL = {"01": "1", "10": "10", "20": "20", "30": "30", "40": "40"}
EFF = {"01": "1.0", "10": "7.0", "20": "10.5", "30": "12.6", "40": "14.0"}
MIXES = ["m70_20_10", "m50_35_15", "m34_33_33", "m15_35_50"]
MIX_LABEL = {m: "/".join(m[1:].split("_")) for m in MIXES}

H = D["hsweep"]
M = D["mix"]


def esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- #
# tiny SVG chart kit
# --------------------------------------------------------------------------- #

def axes(x0, y0, w, h, ymax, yticks, ylab, xs, xlabels, ymin=0.0):
    """Recessive grid + axis labels.  Returns svg string."""
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


def marker(x, y, cls, label, value):
    """Marker + oversized transparent hit target carrying the tooltip payload."""
    return (f'<circle class="mk {cls}" cx="{x:.1f}" cy="{y:.1f}" r="4.5"/>'
            f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="14" '
            f'data-l="{esc(label)}" data-v="{esc(value)}"/>')


def errbar(x, ylo, yhi, cls):
    return (f'<line class="eb {cls}" x1="{x:.1f}" y1="{ylo:.1f}" x2="{x:.1f}" y2="{yhi:.1f}"/>'
            f'<line class="eb {cls}" x1="{x-4:.1f}" y1="{ylo:.1f}" x2="{x+4:.1f}" y2="{ylo:.1f}"/>'
            f'<line class="eb {cls}" x1="{x-4:.1f}" y1="{yhi:.1f}" x2="{x+4:.1f}" y2="{yhi:.1f}"/>')


# --------------------------------------------------------------------------- #
# Figure 1 — H-sweep, hours to 55 %
# --------------------------------------------------------------------------- #

def fig1() -> str:
    W, Hh = 760, 400
    x0, y0, w, h = 62, 44, 560, 286
    ymax, ymin = 1.7, 0.0
    xs = [x0 + w * i / (len(GAPS) - 1) for i in range(len(GAPS))]
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="Hours to 55 percent accuracy versus tier speed gap">']
    s.append(axes(x0, y0, w, h, ymax, [0, 0.4, 0.8, 1.2, 1.6], "simulated hours to 55 %",
                  xs, [f"G={GAP_LABEL[g]}" for g in GAPS], ymin))
    for g, x in zip(GAPS, xs):
        s.append(f'<text class="sub" x="{x:.1f}" y="{y0+h+34}" text-anchor="middle">{EFF[g]}×</text>')
    s.append(f'<text class="sub" x="{x0-8}" y="{y0+h+34}" text-anchor="end">eff.</text>')
    for arm in ARMS:
        pts, mks = [], []
        for g, x in zip(GAPS, xs):
            c = H["cells"][f"{arm}_{g}"]
            v, sp = c["mean_hours"], c["spread"]
            y = y0 + h - (v - ymin) / (ymax - ymin) * h
            pts.append((x, y))
            if sp:
                ylo = y0 + h - (v - sp - ymin) / (ymax - ymin) * h
                yhi = y0 + h - (v + sp - ymin) / (ymax - ymin) * h
                mks.append(errbar(x, ylo, yhi, arm))
            mks.append(marker(x, y, arm, f"{arm} · G={GAP_LABEL[g]}",
                              f"{v:.2f} h" + (f" ± {sp:.2f}" if sp else "")))
        s.append(polyline(pts, f"ln {arm}"))
        s.extend(mks)
        lx, ly = pts[-1]
        s.append(f'<text class="dl {arm}" x="{lx+12:.1f}" y="{ly+4:.1f}">{ARM_LABEL[arm]}</text>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------- #
# Figure 2 — H-sweep, dispatch share (3 panels)
# --------------------------------------------------------------------------- #

def fig2() -> str:
    W, Hh = 960, 330
    pw, ph = 230, 210
    y0 = 46
    ymax = 60.0
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="Percent of dispatches per vehicle tier versus gap, one panel per arm">']
    for pi, arm in enumerate(ARMS):
        x0 = 56 + pi * (pw + 50)
        xs = [x0 + pw * i / (len(GAPS) - 1) for i in range(len(GAPS))]
        s.append(f'<text class="panel" x="{x0}" y="{y0-22}">{ARM_LABEL[arm]}</text>')
        s.append(axes(x0, y0, pw, ph, ymax, [0, 20, 40, 60],
                      "% of dispatches" if pi == 0 else "",
                      xs, [f"{GAP_LABEL[g]}" for g in GAPS]))
        # pool reference — fixed 50/35/15 fleet
        for tier in TIERS:
            py = y0 + ph - H["pool"][tier] / ymax * ph
            s.append(f'<line class="pool" x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}"/>')
        for tier in TIERS:
            pts, mks = [], []
            for g, x in zip(GAPS, xs):
                v = H["cells"][f"{arm}_{g}"]["share"][tier]
                y = y0 + ph - v / ymax * ph
                pts.append((x, y))
                mks.append(marker(x, y, tier, f"{arm} · {tier} · G={GAP_LABEL[g]}", f"{v:.1f} %"))
            s.append(polyline(pts, f"ln {tier}"))
            s.extend(mks)
            if pi == 2:
                lx, ly = pts[-1]
                s.append(f'<text class="dl {tier}" x="{lx+10:.1f}" y="{ly+4:.1f}">{tier}</text>')
        s.append(f'<text class="sub" x="{x0+pw/2:.1f}" y="{y0+ph+36}" text-anchor="middle">gap G</text>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------- #
# Figure 3 — mix, hours to 55 % + ratio to that fleet's own baseline
# --------------------------------------------------------------------------- #

CAPACITY = {"m70_20_10": "1.00", "m50_35_15": "1.30", "m34_33_33": "1.80", "m15_35_50": "3.21"}
ET = {"m70_20_10": "14.8", "m50_35_15": "11.4", "m34_33_33": "8.2", "m15_35_50": "4.6"}


def fig3() -> str:
    """Absolute hours to 55 % per fleet — same full-width single panel as figure 1.

    No direct labels: gpu and baseline genuinely coincide on the straggler-heavy
    fleets (2.7 px and 7.1 px apart at 50/35/15 and 70/20/10 — that IS the result,
    ratio 1.02-1.03), so labels there would misrepresent the data as much as they
    would collide.  The legend and the table carry identity instead.
    """
    W, Hh = 760, 410
    x0, y0, w, h = 62, 44, 560, 286
    ymax = 1.1
    xs = [x0 + w * i / (len(MIXES) - 1) for i in range(len(MIXES))]
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="Simulated hours to 55 percent accuracy for each arm, across the four fleet mixes">']
    s.append(axes(x0, y0, w, h, ymax, [0, 0.25, 0.5, 0.75, 1.0], "simulated hours to 55 %",
                  xs, [MIX_LABEL[m] for m in MIXES]))
    for m, x in zip(MIXES, xs):
        s.append(f'<text class="sub" x="{x:.1f}" y="{y0+h+34}" text-anchor="middle">{CAPACITY[m]}×</text>')
        s.append(f'<text class="sub" x="{x:.1f}" y="{y0+h+48}" text-anchor="middle">{ET[m]} s</text>')
    s.append(f'<text class="sub" x="{x0-8}" y="{y0+h+34}" text-anchor="end">capacity</text>')
    s.append(f'<text class="sub" x="{x0-8}" y="{y0+h+48}" text-anchor="end">E[T]</text>')
    for arm in ARMS:
        pts, mks = [], []
        for m, x in zip(MIXES, xs):
            v = M["cells"][f"{arm}_{m}"]["hours"]
            y = y0 + h - v / ymax * h
            pts.append((x, y))
            mks.append(marker(x, y, arm, f"{arm} · {MIX_LABEL[m]}", f"{v:.2f} h"))
        s.append(polyline(pts, f"ln {arm}"))
        s.extend(mks)
    s.append("</svg>")
    return "\n".join(s)


def fig4() -> str:
    """The same four fleets divided by each one's own baseline — capacity removed."""
    W, Hh = 760, 250
    x0, y0, w, h = 62, 42, 560, 150
    rmin, rmax = 0.5, 1.6
    xs = [x0 + w * i / (len(MIXES) - 1) for i in range(len(MIXES))]
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="Each arm divided by that fleet own baseline, by fleet mix">']
    s.append(axes(x0, y0, w, h, rmax, [0.6, 0.8, 1.0, 1.2, 1.4], "÷ that fleet's own baseline",
                  xs, [MIX_LABEL[m] for m in MIXES], rmin))
    yone = y0 + h - (1.0 - rmin) / (rmax - rmin) * h
    s.append(f'<line class="unity" x1="{x0}" y1="{yone:.1f}" x2="{x0+w}" y2="{yone:.1f}"/>')
    s.append(f'<text class="sub" x="{x0+w}" y="{yone-7:.1f}" text-anchor="end">1.00 = no better than random selection</text>')
    for arm in ("gpu", "cpu"):
        pts, mks = [], []
        for m, x in zip(MIXES, xs):
            r = M["cells"][f"{arm}_{m}"]["hours"] / M["cells"][f"baseline_{m}"]["hours"]
            y = y0 + h - (r - rmin) / (rmax - rmin) * h
            pts.append((x, y))
            mks.append(marker(x, y, arm, f"{arm} ÷ baseline · {MIX_LABEL[m]}", f"{r:.2f}×"))
        s.append(polyline(pts, f"ln {arm}"))
        s.extend(mks)
        lx, ly = pts[0]
        s.append(f'<text class="dl {arm}" x="{lx-10:.1f}" y="{ly+4:.1f}" text-anchor="end">{ARM_LABEL[arm]}</text>')
    s.append(f'<text class="sub" x="{x0+w/2:.1f}" y="{y0+h+38}" text-anchor="middle">fleet  xavier/orin/thor</text>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------- #
# Figure 4 — mix, dispatch share vs each fleet's own pool
# --------------------------------------------------------------------------- #

def fig5() -> str:
    """Dispatch share per tier, one panel per arm — mirrors figure 2's layout."""
    W, Hh = 960, 344
    pw, ph = 230, 210
    y0 = 46
    ymax = 75.0
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="Percent of dispatches per vehicle tier across fleet mixes, one panel per arm, against each fleet pool share">']
    for pi, arm in enumerate(ARMS):
        x0 = 56 + pi * (pw + 50)
        xs = [x0 + pw * i / (len(MIXES) - 1) for i in range(len(MIXES))]
        s.append(f'<text class="panel" x="{x0}" y="{y0-22}">{ARM_LABEL[arm]}</text>')
        s.append(axes(x0, y0, pw, ph, ymax, [0, 25, 50, 75],
                      "% of dispatches" if pi == 0 else "",
                      xs, [MIX_LABEL[m] for m in MIXES]))
        # each fleet has its own pool share, so the reference is a line that moves
        for tier in TIERS:
            pts = [(x, y0 + ph - M["pool"][m][tier] / ymax * ph) for m, x in zip(MIXES, xs)]
            s.append(polyline(pts, "poolline"))
        for tier in TIERS:
            pts, mks = [], []
            for m, x in zip(MIXES, xs):
                v = M["cells"][f"{arm}_{m}"]["share"][tier]
                y = y0 + ph - v / ymax * ph
                pts.append((x, y))
                mks.append(marker(x, y, tier,
                                  f"{arm} · {tier} · {MIX_LABEL[m]}",
                                  f"{v:.1f} % (pool {M['pool'][m][tier]:.0f} %)"))
            s.append(polyline(pts, f"ln {tier}"))
            s.extend(mks)
            if pi == 2:
                lx, ly = pts[-1]
                s.append(f'<text class="dl {tier}" x="{lx+10:.1f}" y="{ly+4:.1f}">{tier}</text>')
        s.append(f'<text class="sub" x="{x0+pw/2:.1f}" y="{y0+ph+38}" text-anchor="middle">fleet  ·  thin line = pool share</text>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

def table_h_time() -> str:
    r = ['<table><caption>Hours to 55 %, mean of seeds 1234 / 5678 (± half-range)</caption>',
         "<thead><tr><th>gap G</th><th>eff.</th>"
         + "".join(f"<th>{a}</th>" for a in ARMS)
         + "<th>gpu÷base</th><th>gpu÷cpu</th></tr></thead><tbody>"]
    for g in GAPS:
        c = {a: H["cells"][f"{a}_{g}"] for a in ARMS}
        r.append(f"<tr><th>{GAP_LABEL[g]}</th><td>{EFF[g]}×</td>"
                 + "".join(f'<td>{c[a]["mean_hours"]:.3f} <span class="pm">± {c[a]["spread"]:.3f}</span></td>' for a in ARMS)
                 + f'<td>{c["gpu"]["mean_hours"]/c["baseline"]["mean_hours"]:.2f}</td>'
                 + f'<td>{c["gpu"]["mean_hours"]/c["cpu"]["mean_hours"]:.2f}</td></tr>')
    return "\n".join(r) + "</tbody></table>"


def table_h_share() -> str:
    r = ['<table><caption>Dispatch share %, mean of seeds 1234 / 5678 · pool is 50 / 35 / 15</caption>',
         "<thead><tr><th>gap G</th>"
         + "".join(f'<th colspan="3">{a}</th>' for a in ARMS)
         + "</tr><tr><th></th>"
         + "".join(f"<th>{t[:3]}</th>" for _ in ARMS for t in TIERS)
         + "</tr></thead><tbody>"]
    for g in GAPS:
        cells = ""
        for a in ARMS:
            sh = H["cells"][f"{a}_{g}"]["share"]
            cells += "".join(f"<td>{sh[t]:.1f}</td>" for t in TIERS)
        r.append(f"<tr><th>{GAP_LABEL[g]}</th>{cells}</tr>")
    return "\n".join(r) + "</tbody></table>"


def table_mix_time() -> str:
    """Hours to 55 % laid out like figure 3 — fleets across, arms down."""
    r = ['<table><caption>Hours to 55 % by fleet (seed 8888, n=1) · capacity is E[T] of the '
         'slowest fleet ÷ this one</caption>',
         "<thead><tr><th>fleet</th>" + "".join(f"<th>{MIX_LABEL[m]}</th>" for m in MIXES)
         + "</tr></thead><tbody>"]
    r.append('<tr><th>capacity</th>' + "".join(f"<td>{CAPACITY[m]}×</td>" for m in MIXES) + "</tr>")
    r.append('<tr><th>E[T]</th>' + "".join(f"<td>{ET[m]} s</td>" for m in MIXES) + "</tr>")
    for i, a in enumerate(ARMS):
        cls = ' class="grp"' if i == 0 else ""
        r.append(f'<tr{cls}><th>{a}</th>'
                 + "".join(f'<td>{M["cells"][f"{a}_{m}"]["hours"]:.3f} h</td>' for m in MIXES)
                 + "</tr>")
    for a in ("gpu", "cpu"):
        r.append(f'<tr><th>{a} ÷ base</th>'
                 + "".join(f'<td>{M["cells"][f"{a}_{m}"]["hours"]/M["cells"][f"baseline_{m}"]["hours"]:.2f}×</td>'
                           for m in MIXES) + "</tr>")
    return "\n".join(r) + "</tbody></table>"


def table_mix() -> str:
    r = ['<table><caption>Fleet-mix sweep · hours to 55 % and dispatch share % (seed 8888, n=1)</caption>',
         "<thead><tr><th>fleet</th><th>arm</th><th>hours</th><th>÷ base</th>"
         + "".join(f"<th>{t[:3]}</th>" for t in TIERS) + "</tr></thead><tbody>"]
    for m in MIXES:
        for i, a in enumerate(ARMS):
            c = M["cells"][f"{a}_{m}"]
            ratio = c["hours"] / M["cells"][f"baseline_{m}"]["hours"]
            first = (f'<th rowspan="3">{MIX_LABEL[m]}</th>' if i == 0 else "")
            r.append(f'<tr class="{"grp" if i == 0 else ""}">{first}<td>{a}</td>'
                     f'<td>{c["hours"]:.3f}</td><td>{ratio:.2f}</td>'
                     + "".join(f'<td>{c["share"][t]:.1f}</td>' for t in TIERS) + "</tr>")
    return "\n".join(r) + "</tbody></table>"


# --------------------------------------------------------------------------- #

CSS = """
:root{
  color-scheme: light;
  --ground:#eeeeec; --surface:#fcfcfb; --surface-2:#f6f6f3;
  --ink:#14140f; --ink-2:#4d4d44; --ink-3:#7d7d70;
  --rule:#dededa; --rule-2:#e8e8e4;
  --accent:#2a78d6;
  --arm-gpu:#2a78d6; --arm-cpu:#eb6834; --arm-baseline:#1baf7a;
  --t-xavier:#86b6ef; --t-orin:#2a78d6; --t-thor:#104281;
  --pool:#b4b4aa;
  --flag:#8a5a00; --flag-bg:#fdf5e3; --flag-rule:#e8d9b0;
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    color-scheme: dark;
    --ground:#121211; --surface:#1a1a19; --surface-2:#212120;
    --ink:#f3f3ee; --ink-2:#bcbcb2; --ink-3:#8a8a80;
    --rule:#33332f; --rule-2:#2a2a27;
    --accent:#3987e5;
    --arm-gpu:#3987e5; --arm-cpu:#d95926; --arm-baseline:#199e70;
    --t-xavier:#cde2fb; --t-orin:#5598e7; --t-thor:#184f95;
    --pool:#57574f;
    --flag:#e5c169; --flag-bg:#241f14; --flag-rule:#4a3f24;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --ground:#121211; --surface:#1a1a19; --surface-2:#212120;
  --ink:#f3f3ee; --ink-2:#bcbcb2; --ink-3:#8a8a80;
  --rule:#33332f; --rule-2:#2a2a27;
  --accent:#3987e5;
  --arm-gpu:#3987e5; --arm-cpu:#d95926; --arm-baseline:#199e70;
  --t-xavier:#cde2fb; --t-orin:#5598e7; --t-thor:#184f95;
  --pool:#57574f;
  --flag:#e5c169; --flag-bg:#241f14; --flag-rule:#4a3f24;
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font:16px/1.65 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1060px;margin:0 auto;padding:56px 24px 96px}
.measure{max-width:68ch}

h1,h2,h3{font-family:Charter,"Bitstream Charter","Iowan Old Style",Georgia,serif;
  font-weight:600;text-wrap:balance;margin:0}
h1{font-size:2.5rem;line-height:1.12;letter-spacing:-.015em}
h2{font-size:1.6rem;line-height:1.2;margin-top:0}
h3{font-size:1.08rem;line-height:1.3}
p{margin:0}
.eyebrow{font:600 .72rem/1 ui-monospace,SFMono-Regular,Menlo,monospace;
  letter-spacing:.14em;text-transform:uppercase;color:var(--ink-3)}
.lede{font-size:1.09rem;color:var(--ink-2)}
.stack{display:flex;flex-direction:column;gap:18px}
.stack-lg{display:flex;flex-direction:column;gap:56px}
header.head{display:flex;flex-direction:column;gap:16px;
  padding-bottom:32px;border-bottom:2px solid var(--ink);margin-bottom:56px}
.meta{display:flex;flex-wrap:wrap;gap:8px 26px;font:.8rem/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink-3)}
.meta b{color:var(--ink-2);font-weight:600}

section{scroll-margin-top:24px}
.sec-head{display:flex;flex-direction:column;gap:10px;margin-bottom:26px}

.card{background:var(--surface);border:1px solid var(--rule);border-radius:3px;
  padding:24px 26px 20px;display:flex;flex-direction:column;gap:16px}
.fig-head{display:flex;flex-direction:column;gap:5px}
.fig-n{font:600 .72rem/1 ui-monospace,SFMono-Regular,Menlo,monospace;
  letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}
.fig-sub{font-size:.88rem;color:var(--ink-2);max-width:62ch}
.plot{width:100%;overflow-x:auto}
svg{width:100%;height:auto;display:block;overflow:visible}

.grid{stroke:var(--rule-2);stroke-width:1}
.axis{stroke:var(--rule);stroke-width:1}
.pool,.poolline{stroke:var(--pool);stroke-width:1.25;stroke-dasharray:3 3;fill:none}
.unity{stroke:var(--ink-3);stroke-width:1.25;stroke-dasharray:4 3}
text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
.tick{font-size:11px;fill:var(--ink-3)}
.sub{font-size:10.5px;fill:var(--ink-3)}
.axlab{font-size:11px;fill:var(--ink-2);font-weight:600}
.panel{font-size:12.5px;fill:var(--ink);font-weight:700}
.dl{font-size:11.5px;font-weight:700}
.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.mk{stroke:var(--surface);stroke-width:2}
.eb{stroke-width:1.5;opacity:.55}
.hit{fill:transparent;cursor:crosshair}

.gpu{stroke:var(--arm-gpu)} circle.gpu{fill:var(--arm-gpu)} text.gpu{fill:var(--arm-gpu);stroke:none}
.cpu{stroke:var(--arm-cpu)} circle.cpu{fill:var(--arm-cpu)} text.cpu{fill:var(--arm-cpu);stroke:none}
.baseline{stroke:var(--arm-baseline)} circle.baseline{fill:var(--arm-baseline)} text.baseline{fill:var(--arm-baseline);stroke:none}
.xavier{stroke:var(--t-xavier)} circle.xavier{fill:var(--t-xavier)} text.xavier{fill:var(--t-xavier);stroke:none}
.orin{stroke:var(--t-orin)} circle.orin{fill:var(--t-orin)} text.orin{fill:var(--t-orin);stroke:none}
.thor{stroke:var(--t-thor)} circle.thor{fill:var(--t-thor)} text.thor{fill:var(--t-thor);stroke:none}

.legend{display:flex;flex-wrap:wrap;gap:6px 20px;font:.8rem/1 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:7px}
.sw{width:15px;height:3px;border-radius:2px;flex:none}
.sw.dash{height:0;border-top:1.5px dashed var(--pool);width:15px}

details{border-top:1px solid var(--rule-2);padding-top:12px}
summary{cursor:pointer;font:.8rem/1 ui-monospace,SFMono-Regular,Menlo,monospace;
  color:var(--ink-2);letter-spacing:.03em;list-style:none;display:flex;align-items:center;gap:7px}
summary::-webkit-details-marker{display:none}
summary::before{content:"▸";color:var(--ink-3);transition:transform .15s}
details[open] summary::before{transform:rotate(90deg)}
summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px;border-radius:2px}
.tw{overflow-x:auto;margin-top:14px}
table{border-collapse:collapse;font:.79rem/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums;width:100%}
caption{text-align:left;color:var(--ink-3);font-size:.76rem;padding-bottom:9px}
th,td{padding:5px 9px;text-align:right;border-bottom:1px solid var(--rule-2);white-space:nowrap}
thead th{color:var(--ink-3);font-weight:600;border-bottom:1px solid var(--rule)}
tbody th{text-align:left;color:var(--ink);font-weight:600}
td{color:var(--ink-2)}
tr.grp td,tr.grp th{border-top:1px solid var(--rule)}
.pm{color:var(--ink-3)}

.pair{display:grid;grid-template-columns:1fr 1fr;gap:22px;align-items:start}
@media(max-width:820px){.pair{grid-template-columns:1fr}}

.flag{background:var(--flag-bg);border:1px solid var(--flag-rule);border-left:3px solid var(--flag);
  border-radius:3px;padding:16px 20px;display:flex;flex-direction:column;gap:8px}
.flag h3{font-size:.95rem;color:var(--flag);font-family:inherit;font-weight:700}
.flag p{font-size:.88rem;color:var(--ink-2)}
.note{font-size:.86rem;color:var(--ink-3)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.87em;
  background:var(--surface-2);padding:1px 5px;border-radius:3px;color:var(--ink-2)}
.keys{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);border-radius:3px;overflow:hidden}
.key{background:var(--surface);padding:13px 16px;display:flex;flex-direction:column;gap:3px}
.key b{font:600 .7rem/1.3 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.08em;
  text-transform:uppercase;color:var(--ink-3)}
.key span{font-size:.92rem;color:var(--ink)}

#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;z-index:50;
  background:var(--ink);color:var(--ground);padding:6px 10px;border-radius:3px;
  font:.76rem/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap;
  box-shadow:0 3px 14px rgba(0,0,0,.22)}
#tip b{display:block;font-weight:600;opacity:.72;font-size:.72rem}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

JS = """
(function(){
  var tip=document.getElementById('tip');
  document.querySelectorAll('.hit').forEach(function(h){
    function show(e){
      var r=(e.touches?e.touches[0]:e);
      tip.innerHTML='<b>'+h.dataset.l+'</b>'+h.dataset.v;
      tip.style.opacity=1;
      var x=r.clientX+14, y=r.clientY-12;
      if(x+tip.offsetWidth>window.innerWidth-8) x=r.clientX-tip.offsetWidth-14;
      tip.style.left=x+'px'; tip.style.top=y+'px';
    }
    h.addEventListener('mouseenter',show);
    h.addEventListener('mousemove',show);
    h.addEventListener('mouseleave',function(){tip.style.opacity=0;});
  });
})();
"""


def legend(items) -> str:
    out = ['<div class="legend">']
    for cls, lab in items:
        if cls == "pool":
            out.append(f'<span><i class="sw dash"></i>{esc(lab)}</span>')
        else:
            out.append(f'<span><i class="sw" style="background:var(--{cls})"></i>{esc(lab)}</span>')
    return "\n".join(out) + "</div>"


def main() -> None:
    arm_leg = legend([("arm-gpu", "gpu — GPU TFLOPS predictor"),
                      ("arm-cpu", "cpu — CPU GHz predictor (wrong on purpose)"),
                      ("arm-baseline", "baseline — random, no predictor")])
    tier_leg = legend([("t-xavier", "xavier (slowest, 20 s)"),
                       ("t-orin", "orin (3.65 s)"),
                       ("t-thor", "thor (fastest, 0.67 s)"),
                       ("pool", "pool share")])

    html = f"""<title>Tier Steering Sweeps</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style>
<div id="tip"></div>
<div class="wrap">

<header class="head">
  <p class="eyebrow">Async federated learning · vehicle GPU tiers</p>
  <h1>Does the selector trade data quality for speed?</h1>
  <p class="lede measure">Two sweeps, 57 finished runs, one question: as the fleet's hardware
  gets more uneven, does client selection shift its preference from data quality
  (<code>G_i</code>) toward training speed (<code>τ̂</code>)? Every time axis below is the
  <b>reconstructed simulated clock</b> — tier sleep only, no host cost.</p>
  <div class="meta">
    <span><b>H-sweep</b> 3 arms × 5 gaps × 2 seeds = 30 runs</span>
    <span><b>Fleet mix</b> 3 arms × 4 fleets × 1 seed = 12 runs</span>
    <span><b>Target</b> 55 % top-1, CIFAR-10 / ResNet-18</span>
  </div>
</header>

<div class="stack-lg">

<section>
  <div class="sec-head">
    <p class="eyebrow">Before the figures</p>
    <h2>The time axis was rebuilt, not reported</h2>
  </div>
  <div class="stack">
    <p class="measure">PLATO advances its simulated clock to
    <code>training_time + processing_time + comm_time</code>. Only the first term is
    simulated; the other two are <code>time.time()</code> deltas for serialising and
    shipping a 42.69&nbsp;MB payload — real work, on whichever GPU happened to run that
    cell, contended or not. Since these runs did not share a machine and some ran two
    lanes in parallel, the reported clock is not comparable across arms.</p>
    <p class="measure">Every figure here therefore reads
    <code>recompute_wall_time.py --floor 0</code>: the recorded dispatch/merge schedule
    replayed with tier sleep as the only cost. The seed-1234/5678 runs had never been
    reconstructed before — that ran for this report.</p>
    <div class="keys">
      <div class="key"><b>host cost removed</b><span>0.393 – 0.439 s per report</span></div>
      <div class="key"><b>reported ÷ clean</b><span>1.02 – 1.03× uniformly</span></div>
      <div class="key"><b>reconstruction error</b><span>0.02 – 0.29 % on 29 of 30</span></div>
      <div class="key"><b>conclusions changed</b><span>none — only the axis</span></div>
    </div>
    <p class="note measure">The correction is small and nearly uniform, so it moves levels
    rather than orderings. That is the good outcome: it means the host contamination was
    never carrying a result, and now it demonstrably is not.</p>
  </div>
</section>

<section>
  <div class="sec-head">
    <p class="eyebrow">Topic 1 — heterogeneity sweep</p>
    <h2>Widening the tier gap</h2>
    <p class="lede measure">Fleet held at 50 / 35 / 15. The straggler is anchored at 20 s in
    every setting, so only the fast tiers move. Seed 8888 is excluded — seeds 1234 and 5678
    only, both read off the same reconstructed clock.</p>
  </div>
  <div class="stack">
    <div class="card">
      <div class="fig-head">
        <p class="fig-n">Figure 1</p>
        <h3>Time to 55 % accuracy</h3>
        <p class="fig-sub">Mean of two seeds; bars span the two. Lower is better.
        <code>G=1</code> is the control — all three tiers sleep 20 s, so there is no speed
        signal to exploit and <code>gpu</code> should <em>not</em> win. It doesn't.</p>
      </div>
      <div class="plot">{fig1()}</div>
      {arm_leg}
      <details><summary>Data table</summary><div class="tw">{table_h_time()}</div></details>
    </div>

    <div class="card">
      <div class="fig-head">
        <p class="fig-n">Figure 2</p>
        <h3>Which vehicle tier gets picked</h3>
        <p class="fig-sub">Percent of all dispatches going to each tier, counted from the
        client-side dispatch log so the predictor-less <code>baseline</code> is measurable
        too. Dashed lines are the pool's own 50 / 35 / 15.</p>
      </div>
      <div class="plot">{fig2()}</div>
      {tier_leg}
      <details><summary>Data table</summary><div class="tw">{table_h_share()}</div></details>
    </div>

    <div class="stack">
      <p class="measure"><b>The mechanism is unambiguous.</b> As the gap widens,
      <code>gpu</code> walks its xavier share from 49.3 % down to 36.3 % and lifts thor from
      13.7 % to 20.7 % — monotone at every step. <code>cpu</code> moves the opposite way
      (48.8 → 52.8 % xavier): its belief cannot separate xavier from orin, since both clock
      2.00 GHz, so it steers slightly <em>toward</em> the straggler. <code>baseline</code>
      sits flat near the pool.</p>
      <p class="measure"><b>The end-to-end timing follows, but with less room.</b>
      <code>gpu</code> beats <code>cpu</code> at every gap (0.72–0.83×) and the margin widens
      with G. Against <code>baseline</code> it is 0.82–0.97× for G ≥ 10 and 1.12× at the
      control — the right sign everywhere, but the two-seed spread is ±0.02–0.16 h, so
      individual gaps are not separately significant.</p>
    </div>
  </div>
</section>

<section>
  <div class="sec-head">
    <p class="eyebrow">Topic 2 — fleet-mix sweep</p>
    <h2>Holding the gap, moving the ratio</h2>
    <p class="lede measure">Tier speeds pinned at G=30; only the proportions change. Capacity
    alone shrinks E[T] by 3.2× from 70/20/10 to 15/35/50, so raw hours are mostly free
    speed-up — the ratio to each fleet's own baseline is the part that is about selection.</p>
  </div>
  <div class="stack">
    <div class="card">
      <div class="fig-head">
        <p class="fig-n">Figure 3</p>
        <h3>Time to 55 % accuracy by fleet</h3>
        <p class="fig-sub">The three arms compared directly, same layout as Figure 1. Lower is
        better. Read the <code>capacity</code> row before the lines: E[T] falls from 14.8 s to
        4.6 s across this axis, so most of the drop is free speed-up that any selector gets —
        Figure 4 divides it out.</p>
      </div>
      <div class="plot">{fig3()}</div>
      {arm_leg}
      <p class="note">No direct labels here: <code>gpu</code> and <code>baseline</code> really
      do coincide on the two straggler-heavy fleets (0.807 vs 0.835 h, then 0.597 vs 0.607 h).
      That closeness is the finding, so the legend and table carry identity instead.</p>
      <details><summary>Data table</summary><div class="tw">{table_mix_time()}</div></details>
    </div>

    <div class="card">
      <div class="fig-head">
        <p class="fig-n">Figure 4</p>
        <h3>The same runs, capacity divided out</h3>
        <p class="fig-sub">Each arm ÷ that fleet's own baseline. This is the only view on this
        page that is about <em>selection</em> rather than hardware.</p>
      </div>
      <div class="plot">{fig4()}</div>
      <p class="measure"><b>The advantage peaks in the middle, as pre-registered.</b>
      <code>gpu ÷ baseline</code> runs 1.03 → 1.02 → <b>0.64</b> → 0.80. At 70/20/10 there are
      only ten thor clients and <code>max_concurrency = 10</code> cannot keep picking them; at
      15/35/50 half the pool is already fast, so preferring thor is nearly free. The middle is
      where a selector has something to do. This was the sweep's riskiest prediction and it
      held — though with n = 1 it is a single observation, not an interval.</p>
    </div>

    <div class="card">
      <div class="fig-head">
        <p class="fig-n">Figure 5</p>
        <h3>Which vehicle tier gets picked, by fleet</h3>
        <p class="fig-sub">Thin dashed lines are each fleet's own pool share, which moves with
        the mix. Distance from the dashed line is steering.</p>
      </div>
      <div class="plot">{fig5()}</div>
      {tier_leg}
      <details><summary>Data table</summary><div class="tw">{table_mix()}</div></details>
    </div>
  </div>
</section>

<section>
  <div class="sec-head">
    <p class="eyebrow">Read before writing</p>
    <h2>What these figures do not support</h2>
  </div>
  <div class="stack">
    <div class="flag">
      <h3>The two topics are not the same experiment</h3>
      <p>They differ in two rendered keys, on purpose:
      <code>t_round_model</code> is <code>"min"</code> in the H-sweep and <code>"queue"</code>
      in the fleet-mix sweep, and <code>target_accuracy</code> is 0.6 vs 1.01. Everything else
      — dataset, model, sampler, concentration, seed mechanism, staleness bound, concurrency,
      anchor — is identical across all 72 configs. The G=30 / 50-35-15 cell exists in both
      sweeps and <b>must not be pooled</b>; it was deliberately re-run for topic 2.</p>
    </div>
    <div class="flag">
      <h3>The baseline is not a perfectly neutral control</h3>
      <p>Its dispatch lift drifts to 0.89 and 0.80 on xavier at the two fast-heavy fleets,
      against a pre-registered "~1.00 on every fleet". This looks like availability bias in
      random-skipping-busy — slow clients are occupied longer, so random sampling finds them
      less often — not a bug. But it means <code>gpu</code>'s steering should be read against
      the baseline's lift, not against 1.00.</p>
    </div>
    <div class="flag">
      <h3>Topic 2 has n = 1</h3>
      <p>The fleet-mix sweep ran one seed, so Figures 3 and 4 carry no error bars and the
      0.64 minimum has nothing separating it from run-to-run variance. Two more seeds would
      cost about the same as the H-sweep's second and third did.</p>
    </div>
    <p class="note measure">Also worth carrying into the write-up: first-crossing times inherit
    roughly 10 % of evaluation noise, so any ratio inside about 1.15× is not resolvable from a
    single crossing. All curves here are smoothed with a 25-evaluation rolling median before
    the crossing is read.</p>
  </div>
</section>

</div>
</div>
<script>{JS}</script>
"""
    out = HERE / "report_hsweep_mix.html"
    out.write_text(html)
    print(f"wrote {out}  ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
