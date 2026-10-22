"""
Prompt K Step 2: the final Scenario 1 report, 3 seeds per method (mechanical).

    PYTHONPATH=sumo_close_loop:<repo> python wen_env/s1_final_report.py \
        --out wen_env/evidence/s1_final.md

Sections 2.1-2.4 are generated; 2.5 (run health) reads the watchdog alerts,
the Prompt K launcher log and each run's wen_fresh_eval.csv.
"""

from __future__ import annotations

import argparse
import re
import statistics as st
from pathlib import Path

import numpy as np

import wen_main_report as R
import wen_study1 as S1m
from polaris_report import READING_B, coverage, fmt, rows
from wen_faithful_report import circ
from wen_ptable import run_dir

HERE = Path(__file__).resolve().parent
SEEDS = (4126, 4127, 4128)
THR = (0.80, 0.85, 0.90)
TIERS = ("xavier", "orin", "thor")
ARMS = [("Random", "S1_random"), ("Ours (gpu)", "S1_gpu"), ("Polaris", "S1_polaris"),
        ("Wen et al. (wen_fresh)", "S1_wen_fresh"),
        ("Wen et al. + GPU (wen_fresh_gpu)", "S1_wen_fresh_gpu"),
        ("Wen stale (wen), reference", "S1_wen")]
PROMPT_K_RUNS = [f"S1_{a}_s{s}" for s in (4127, 4128) for a in ("wen_fresh", "wen_fresh_gpu")]


# ---------------------------------------------------------------------------
# per run
# ---------------------------------------------------------------------------

def metrics(cell: str, seed: int) -> dict:
    run = run_dir(cell, seed)
    m = S1m.one(cell, seed)
    curve = R.accuracy_curve(run)
    disp = rows(run / "dispatch_log.csv")
    m["max_acc"] = max(a for _, _, a in curve)
    m["hit"] = {}
    for thr in THR:
        hit = next(((rnd, t) for rnd, t, a in curve if a >= thr), None)
        if hit is None and thr == 0.90 and (cell, seed) in READING_B:
            hit = next((rnd, t) for rnd, t, a in curve if a == m["max_acc"])
        energy = None if hit is None else sum(
            float(r["E_train"]) + float(r["E_up"]) for r in disp
            if float(r["t_finish_clean"]) <= hit[1]) / 1000
        m["hit"][thr] = (hit[1], hit[0], energy) if hit else (None, None, None)
    if m["xhat_err"] is None and (run / "wen_decisions.csv").exists():
        dec = rows(run / "wen_decisions.csv")
        if len(dec) != len(disp) or any(a["pick"] != b["client_id"] for a, b in zip(dec, disp)):
            raise SystemExit(f"{run}: decisions do not line up with dispatches")
        m["xhat_err"] = st.fmean(circ(float(a["x_hat_upload"]), float(b["x_finish"]))
                                 for a, b in zip(dec, disp))
    m["distinct"] = len({r["client_id"] for r in disp})
    m["cov"] = coverage(cell.split("_", 1)[1], seed)
    m["curve"] = curve
    m["energy_curve"] = energy_curve(curve, disp)
    return m


def energy_curve(curve, disp) -> list[tuple[float, float]]:
    """(cumulative ground-truth energy in kJ, accuracy) at each evaluated round."""
    fin = sorted((float(r["t_finish_clean"]), float(r["E_train"]) + float(r["E_up"]))
                 for r in disp)
    t_arr = np.array([f for f, _ in fin])
    e_cum = np.cumsum([e for _, e in fin]) / 1000
    out = []
    for _, t, a in curve:
        k = int(np.searchsorted(t_arr, t, side="right"))
        out.append((float(e_cum[k - 1]) if k else 0.0, a))
    return out


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def ms(vals, f):
    if len(vals) == 1:
        return f.format(vals[0])
    return f"{f.format(st.fmean(vals))} ± {f.format(st.stdev(vals))}"


def thr_cell(ms_list, idx, thr, f):
    vals = [m["hit"][thr][idx] for m in ms_list]
    hit = [v for v in vals if v is not None]
    if len(hit) == len(vals):
        return ms(hit, f)
    mx = [m["max_acc"] for m in ms_list]
    return f"not reached ({len(hit)}/{len(vals)}; max {ms(mx, '{:.4f}')})"


def one_cell(m, idx, thr, f):
    v = m["hit"][thr][idx]
    return f.format(v) if v is not None else f"not reached (max {m['max_acc']:.4f})"


HEAD = ("| method | seeds | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | energy to 80% (kJ) | "
        "energy to 85% (kJ) | energy to 90% (kJ) | rounds to 80% | rounds to 85% | "
        "rounds to 90% | s/round whole / <80% / ≥80% | max accuracy |")


def table_time_energy(data) -> list[str]:
    out = [HEAD, "|" + "---|" * 13]
    for name, cell in ARMS:
        ml = [data[(cell, s)] for s in SEEDS if (cell, s) in data]
        spr = (f"{ms([m['spr'] for m in ml], '{:.3f}')} / "
               f"{ms([m['before'] for m in ml if m['before']], '{:.3f}')} / "
               f"{ms([m['after'] for m in ml if m['after']], '{:.3f}')}")
        out.append(f"| {name} | {len(ml)} | "
                   + " | ".join(thr_cell(ml, 0, t, "{:,.0f}") for t in THR) + " | "
                   + " | ".join(thr_cell(ml, 2, t, "{:.1f}") for t in THR) + " | "
                   + " | ".join(thr_cell(ml, 1, t, "{:,.0f}") for t in THR)
                   + f" | {spr} | {ms([m['max_acc'] for m in ml], '{:.4f}')} |")
    return out


def table_behaviour(data) -> list[str]:
    out = ["| method | seeds | share X / O / T r1–500 | share X / O / T last 500 | Gini | "
           "distinct vehicles | image coverage r500 / r1000 / end | "
           "mean \\|x_upload − 1000\\| (m) | mean \\|x̂ − x\\| (m) |", "|" + "---|" * 9]
    for name, cell in ARMS:
        ml = [data[(cell, s)] for s in SEEDS if (cell, s) in data]
        first = " / ".join(f"{st.fmean(m[f'first_{t}'] for m in ml):.2f}" for t in TIERS)
        last = " / ".join(f"{st.fmean(m[f'last_{t}'] for m in ml):.2f}" for t in TIERS)
        cov = " / ".join(f"{st.fmean(m['cov'][k] for m in ml):.3f}"
                         for k in ("r500", "r1000", "end"))
        xh = [m["xhat_err"] for m in ml if m["xhat_err"] is not None]
        out.append(f"| {name} | {len(ml)} | {first} | {last} | "
                   f"{ms([m['gini'] for m in ml], '{:.3f}')} | "
                   f"{ms([m['distinct'] for m in ml], '{:.1f}')} | {cov} | "
                   f"{ms([m['upload_dist'] for m in ml], '{:.0f}')} | "
                   f"{ms(xh, '{:.0f}') if xh else '—'} |")
    return out


def per_seed_rows(data) -> list[str]:
    out = ["| method | seed | TTA@80 / 85 / 90% (s) | energy to 80 / 85 / 90% (kJ) | "
           "rounds to 80 / 85 / 90% | s/round | max accuracy | Gini | Thor share | "
           "mean \\|x_upload − 1000\\| (m) | mean \\|x̂ − x\\| (m) |", "|" + "---|" * 11]
    for name, cell in ARMS:
        for s in SEEDS:
            if (cell, s) not in data:
                continue
            m = data[(cell, s)]
            out.append(f"| {name} | {s} | "
                       + " / ".join(one_cell(m, 0, t, "{:,.0f}") for t in THR) + " | "
                       + " / ".join(one_cell(m, 2, t, "{:.1f}") for t in THR) + " | "
                       + " / ".join(one_cell(m, 1, t, "{:,.0f}") for t in THR)
                       + f" | {m['spr']:.3f} | {m['max_acc']:.4f} | {m['gini']:.3f} | "
                       f"{m['thor_whole']:.3f} | {m['upload_dist']:.0f} | "
                       f"{fmt(m['xhat_err'], '{:.0f}')} |")
    return out


# ---------------------------------------------------------------------------
# 2.2 verdicts, 2.3 pairwise
# ---------------------------------------------------------------------------

def _lt(a, b):
    return S1m.lower(a, b)


VERDICTS = [
    ("P22", "wen_fresh Gini < wen (S1 part)", "S1_wen_fresh", "S1_wen",
     lambda m: m["gini"], _lt, "{:.3f}"),
    ("P23", "wen_fresh Gini > random (S1 part)", "S1_wen_fresh", "S1_random",
     lambda m: m["gini"], lambda a, b: a > b, "{:.3f}"),
    ("P24", "gpu TTA@80% < wen_fresh", "S1_gpu", "S1_wen_fresh",
     lambda m: m["hit"][0.80][0], _lt, "{:,.0f}"),
    ("P29", "wen_fresh_gpu mean |x̂ − x| < wen_fresh", "S1_wen_fresh_gpu", "S1_wen_fresh",
     lambda m: m["xhat_err"], _lt, "{:.0f}"),
    ("P30", "wen_fresh_gpu mean |x_upload − 1000| < wen_fresh", "S1_wen_fresh_gpu",
     "S1_wen_fresh", lambda m: m["upload_dist"], _lt, "{:.0f}"),
    ("P31", "wen_fresh_gpu s/round < wen_fresh", "S1_wen_fresh_gpu", "S1_wen_fresh",
     lambda m: m["spr"], _lt, "{:.3f}"),
    ("P32", "wen_fresh_gpu Thor share (whole run) > wen_fresh", "S1_wen_fresh_gpu",
     "S1_wen_fresh", lambda m: m["thor_whole"], lambda a, b: a > b, "{:.3f}"),
]


def verdicts(data) -> list[str]:
    out = ["| P | prediction | " + " | ".join(f"seed {s}" for s in SEEDS)
           + " | seeds as predicted | verdict (v3 rule) |", "|" + "---|" * (len(SEEDS) + 4)]
    for p, text, a, b, get, cmp, f in VERDICTS:
        cells, ok = [], 0
        for s in SEEDS:
            va, vb = get(data[(a, s)]), get(data[(b, s)])
            good = cmp(va, vb)
            ok += int(good)
            cells.append(f"{fmt(va, f)} vs {fmt(vb, f)} ({'as predicted' if good else 'not'})")
        verdict = "held" if ok == len(SEEDS) else "not held"
        out.append(f"| {p} | {text.replace('|', chr(92) + '|')} | " + " | ".join(cells)
                   + f" | {ok}/{len(SEEDS)} | {verdict} |")
    return out


def pairwise(data) -> list[str]:
    others = [(n, c) for n, c in ARMS if c != "S1_gpu"]
    out = ["| Ours vs | metric | " + " | ".join(f"{int(t * 100)}%: seeds 4126 / 4127 / 4128 "
                                                  f"(Ours lower)" for t in THR) + " |",
           "|---|---|" + "---|" * len(THR)]
    for name, cell in others:
        for label, idx in (("TTA", 0), ("energy", 2)):
            parts = []
            for thr in THR:
                signs, lower = [], 0
                for s in SEEDS:
                    o, x = data[("S1_gpu", s)]["hit"][thr][idx], data[(cell, s)]["hit"][thr][idx]
                    if o is None and x is None:
                        signs.append("both n/r")
                        continue
                    is_lower = _lt(o, x)
                    lower += int(is_lower)
                    signs.append("<" if is_lower else ">")
                parts.append(f"{' / '.join(signs)} ({lower}/{len(SEEDS)})")
            out.append(f"| {name} | {label} | " + " | ".join(parts) + " |")
    return out


# ---------------------------------------------------------------------------
# 2.4 curves
# ---------------------------------------------------------------------------

def _band(ax, series, label, grid):
    """Mean with min-max band; each seed is interpolated on `grid` within its own
    range, and the band is drawn only where every seed has data."""
    ys = np.array([np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y in series])
    ok = ~np.isnan(ys).any(axis=0)
    line, = ax.plot(grid[ok], ys[:, ok].mean(axis=0), lw=1.3, label=label)
    ax.fill_between(grid[ok], ys[:, ok].min(axis=0), ys[:, ok].max(axis=0),
                    color=line.get_color(), alpha=0.15, lw=0)


def plot(data, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for name, cell in ARMS[:5]:
        ml = [data[(cell, s)] for s in SEEDS if (cell, s) in data]
        tc = [([t for _, t, _ in m["curve"]], [a for _, _, a in m["curve"]]) for m in ml]
        _band(a1, tc, name, np.linspace(0, max(max(x) for x, _ in tc), 600))
        ec = [([e for e, _ in m["energy_curve"]], [a for _, a in m["energy_curve"]]) for m in ml]
        _band(a2, ec, name, np.linspace(0, max(max(x) for x, _ in ec), 600))
    for ax, xl in ((a1, "online clock (s)"), (a2, "cumulative ground-truth energy (kJ)")):
        for thr in THR:
            ax.axhline(thr, color="grey", lw=0.6, ls="--")
        ax.set_xlabel(xl)
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0.3, 0.95)
        ax.legend(fontsize=8, loc="lower right")
    a1.set_title("S1: accuracy vs online clock (mean, min–max over 3 seeds)")
    a2.set_title("S1: accuracy vs cumulative energy (mean, min–max over 3 seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


# ---------------------------------------------------------------------------
# 2.5 run health
# ---------------------------------------------------------------------------

def run_health() -> list[str]:
    logs = HERE / "logs"
    started = {}
    out_file = logs / "promptK_runs.out"
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            m = re.match(r"(\S+) started (\S+) ", line)
            if m:
                started[m.group(2)] = np.datetime64(m.group(1)[:19])
    out = ["| run | wall time (h) | rounds | reached 90% | fresh-evaluation time (h) | "
           "evaluations |", "|---|---|---|---|---|---|"]
    for name in PROMPT_K_RUNS:
        run = HERE.parent / "runs" / "wen_env" / name
        log = logs / f"{name}.log"
        end = np.datetime64(int(log.stat().st_mtime), "s")
        wall = (end - started[name]) / np.timedelta64(1, "h") if name in started else None
        ev = rows(run / "wen_fresh_eval.csv")
        reached = "Target accuracy reached" in log.read_text(errors="replace")
        last = max(int(r["round"]) for r in rows(run / "dispatch_log.csv"))
        out.append(f"| {name} | {fmt(wall, '{:.2f}')} | {last} | {'yes' if reached else 'no'} | "
                   f"{sum(float(e['fresh_eval_s']) for e in ev) / 3600:.2f} | {len(ev)} |")
    alerts = logs / "watchdog_alerts.log"
    text = alerts.read_text().strip() if alerts.exists() else ""
    out += ["", "Watchdog alerts (`wen_env/logs/watchdog_alerts.log`):", "",
            "```", text or "none", "```"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "evidence" / "s1_final.md")
    args = parser.parse_args()
    data = {(c, s): metrics(c, s) for _, c in ARMS for s in SEEDS
            if any((run_dir(c, s) / "results").glob("*.csv"))}
    curves = args.out.with_name(args.out.stem + "_curves.png")
    plot(data, curves)
    lines = ["## 2.1 Main S1 table (mean ± sd over seeds; per-seed rows in 2.1c)", "",
             "### 2.1a Time, energy and rounds to accuracy", "", *table_time_energy(data), "",
             "### 2.1b Selection behaviour", "", *table_behaviour(data), "",
             "### 2.1c Per seed", "", *per_seed_rows(data), "",
             "## 2.2 Verdicts (v3 rule: the same direction in all 3 seeds)", "",
             *verdicts(data), "",
             "## 2.3 Pairwise directions, Ours vs each method (reported, not predictions)", "",
             "`<` = Ours lower (faster or less energy); a threshold Ours reaches and the other "
             "does not counts as Ours lower; `both n/r` = neither reached it.", "",
             *pairwise(data), "",
             "## 2.4 Curves", "", f"![S1 curves]({curves.name})", "",
             "## 2.5 Run health (Prompt K runs)", "", *run_health(), ""]
    args.out.write_text("\n".join(lines))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
