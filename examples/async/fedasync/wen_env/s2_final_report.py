"""
Scenario 2, 3 seeds per arm (mechanical): the S2-B arms of Prompt L at seeds
4126 / 4127 / 4128, v3 verdicts for P25-P28, P33, P38-P43.

    PYTHONPATH=sumo_close_loop:<repo> python wen_env/s2_final_report.py \
        --out wen_env/evidence/s2_final.md

Per-run quantities are exactly those of s2_report.py (energy includes failed
dispatches; STR = merged / dispatched; airtime charges a lost upload its full
T_up; staleness = merge round - dispatch round of merged updates).
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import numpy as np

import s1_final_report as F
import s2_report as S2
import wen_constants as K
import wen_study1 as S1m
from polaris_report import fmt, rows
from wen_ptable import run_dir

HERE = Path(__file__).resolve().parent
SEEDS = (4126, 4127, 4128)
THR = S2.THR
TIERS = S2.TIERS
ARMS = S2.ARMS
NAME = S2.NAME
OURS = "S2_gpu_stage0"


def ms(vals, f):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return f.format(vals[0])
    return f"{f.format(st.fmean(vals))} ± {f.replace('+', '').format(st.stdev(vals))}"


def thr_cell(ml, key, thr, f):
    vals = [m["hit"][thr][key] for m in ml]
    hit = [v for v in vals if v is not None]
    if len(hit) == len(vals):
        return ms(hit, f)
    return f"not reached ({len(hit)}/{len(vals)})"


def coverage_check(data) -> list[str]:
    """Every dispatch of every run used is inside the S2 coverage."""
    lo, hi = K.S2_COVERAGE_M
    bad = []
    for (cell, seed) in data:
        xs = [float(r["x_dispatch"]) for r in rows(run_dir(cell, seed) / "dispatch_log.csv")]
        if not all(lo <= x <= hi for x in xs):
            bad.append(f"{cell}_s{seed}")
    return bad


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def table_main(data) -> list[str]:
    out = ["| arm | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | energy to 80% (kJ) | "
           "energy to 85% (kJ) | energy to 90% (kJ) | **STR** | failed kJ (whole run) | "
           "uplink airtime to 80% (s) | rounds to 80 / 85 / 90% | max accuracy | "
           "mean staleness | s/round |", "|" + "---|" * 14]
    for name, cell in ARMS:
        ml = [data[(cell, s)] for s in SEEDS]
        rounds = " / ".join(thr_cell(ml, "rounds", t, "{:,.0f}") for t in THR)
        out.append(
            f"| {name} | " + " | ".join(thr_cell(ml, "t", t, "{:,.0f}") for t in THR) + " | "
            + " | ".join(thr_cell(ml, "e", t, "{:.1f}") for t in THR)
            + f" | **{ms([m['str'] for m in ml], '{:.3f}')}** | "
            f"{ms([m['fail_kj'] for m in ml], '{:.1f}')} | {thr_cell(ml, 'air', 0.80, '{:,.0f}')} | "
            f"{rounds} | {ms([m['max_acc'] for m in ml], '{:.4f}')} | "
            f"{ms([m['staleness'] for m in ml], '{:.2f}')} | {ms([m['spr'] for m in ml], '{:.3f}')} |")
    return out


def table_behaviour(data) -> list[str]:
    out = ["| arm | share X / O / T r1–500 | share X / O / T last 500 | Gini | distinct | "
           "image coverage r500 / r1000 / end | mean \\|x_upload − 1000\\| (m) | "
           "Stage 0 rejected (of eligible) |", "|" + "---|" * 8]
    for name, cell in ARMS:
        ml = [data[(cell, s)] for s in SEEDS]
        sh = lambda k: " / ".join(f"{st.fmean(m[f'{k}_{t}'] for m in ml):.2f}" for t in TIERS)
        cov = " / ".join(f"{st.fmean(m['cov'][k] for m in ml):.3f}" for k in ("r500", "r1000", "end"))
        s0 = [m["s0"]["rate"] for m in ml if m["s0"]]
        out.append(f"| {name} | {sh('first')} | {sh('last')} | "
                   f"{ms([m['gini'] for m in ml], '{:.3f}')} | "
                   f"{ms([m['distinct'] for m in ml], '{:.1f}')} | {cov} | "
                   f"{ms([m['upload_dist'] for m in ml], '{:.0f}')} | {ms(s0, '{:.3f}')} |")
    return out


def per_seed(data) -> list[str]:
    out = ["| arm | seed | TTA@80 / 85 / 90% (s) | energy to 80 / 85 / 90% (kJ) | STR | "
           "failed kJ | max accuracy | Gini |", "|" + "---|" * 8]
    for name, cell in ARMS:
        for s in SEEDS:
            m = data[(cell, s)]
            t = " / ".join(fmt(m["hit"][x]["t"], "{:,.0f}") for x in THR)
            e = " / ".join(fmt(m["hit"][x]["e"], "{:.1f}") for x in THR)
            out.append(f"| {name} | {s} | {t} | {e} | {m['str']:.3f} | {m['fail_kj']:.1f} | "
                       f"{m['max_acc']:.4f} | {m['gini']:.3f} |")
    return out


# ---------------------------------------------------------------------------
# verdicts, pairwise, plug-in pairs
# ---------------------------------------------------------------------------

def _tta(m):
    return m["hit"][0.80]["t"]


def _e80(m):
    return m["hit"][0.80]["e"]


VERDICTS = [
    ("P25", "gpu (Stage 0 off) has lower TTA@80% than wen_fresh", "S2_gpu", "S2_wen_fresh",
     _tta, S1m.lower, "{:,.0f}"),
    ("P26", "random + Stage 0 has a lower waste fraction than random (STR higher)",
     "S2_random_s0", "S2_random", lambda m: m["str"], lambda a, b: a > b, "{:.3f}"),
    ("P27", "random + Stage 0 has a lower TTA@80% than random", "S2_random_s0", "S2_random",
     _tta, S1m.lower, "{:,.0f}"),
    ("P28", "wen_fresh + Stage 0 has a lower waste fraction than wen_fresh (STR higher)",
     "S2_wen_fresh_s0", "S2_wen_fresh", lambda m: m["str"], lambda a, b: a > b, "{:.3f}"),
    ("P33", "wen_fresh + GPU + Stage 0 waste no higher than wen_fresh + GPU (STR no lower)",
     "S2_wen_fresh_gpu_s0", "S2_wen_fresh_gpu", lambda m: m["str"], lambda a, b: a >= b,
     "{:.3f}"),
    ("P38", "Ours has lower TTA@80% than Polaris", OURS, "S2_polaris", _tta, S1m.lower,
     "{:,.0f}"),
    ("P39", "Ours has a higher STR than Polaris", OURS, "S2_polaris", lambda m: m["str"],
     lambda a, b: a > b, "{:.3f}"),
    ("P40", "Ours uses less energy to reach 80% than Polaris (failed included)", OURS,
     "S2_polaris", _e80, S1m.lower, "{:.1f}"),
    ("P41", "Ours has lower TTA@80% than wen_fresh", OURS, "S2_wen_fresh", _tta, S1m.lower,
     "{:,.0f}"),
    ("P42", "Polaris + Stage 0 has a higher STR than Polaris", "S2_polaris_s0", "S2_polaris",
     lambda m: m["str"], lambda a, b: a > b, "{:.3f}"),
    ("P43", "Polaris + Stage 0 has lower TTA@80% than Polaris", "S2_polaris_s0", "S2_polaris",
     _tta, S1m.lower, "{:,.0f}"),
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
        out.append(f"| {p} | {text} | " + " | ".join(cells)
                   + f" | {ok}/{len(SEEDS)} | {'held' if ok == len(SEEDS) else 'not held'} |")
    return out


def pairwise(data) -> list[str]:
    out = ["| Ours vs | TTA 80 / 85 / 90% (seeds Ours lower) | energy 80 / 85 / 90% "
           "(seeds Ours lower) | STR (seeds Ours higher) |", "|---|---|---|---|"]
    for name, cell in ARMS:
        if cell == OURS:
            continue
        parts = []
        for key in ("t", "e"):
            counts = []
            for t in THR:
                n, both = 0, 0
                for s in SEEDS:
                    a, b = data[(OURS, s)]["hit"][t][key], data[(cell, s)]["hit"][t][key]
                    if a is None and b is None:
                        both += 1
                    elif S1m.lower(a, b):
                        n += 1
                counts.append(f"{n}/{len(SEEDS)}" + (f" ({both} both n/r)" if both else ""))
            parts.append(" / ".join(counts))
        n_str = sum(data[(OURS, s)]["str"] > data[(cell, s)]["str"] for s in SEEDS)
        out.append(f"| {name} | {parts[0]} | {parts[1]} | {n_str}/{len(SEEDS)} |")
    return out


def plugin_pairs(data) -> list[str]:
    out = ["| A → B | mean ΔTTA@80% (s) | seeds B faster at 80% | mean Δenergy to 80% (kJ) | "
           "mean ΔSTR | seeds B higher STR | mean Δfailed kJ | mean Δs/round |",
           "|" + "---|" * 8]
    for a_cell, b_cell in S2.PAIRS:
        A = [data[(a_cell, s)] for s in SEEDS]
        B = [data[(b_cell, s)] for s in SEEDS]
        dt = [_tta(b) - _tta(a) for a, b in zip(A, B) if _tta(a) and _tta(b)]
        de = [_e80(b) - _e80(a) for a, b in zip(A, B) if _e80(a) is not None and _e80(b) is not None]
        out.append(
            f"| {NAME[a_cell]} → {NAME[b_cell]} | {ms(dt, '{:+,.0f}')} | "
            f"{sum(S1m.lower(_tta(b), _tta(a)) for a, b in zip(A, B))}/3 | {ms(de, '{:+.1f}')} | "
            f"{ms([b['str'] - a['str'] for a, b in zip(A, B)], '{:+.3f}')} | "
            f"{sum(b['str'] > a['str'] for a, b in zip(A, B))}/3 | "
            f"{ms([b['fail_kj'] - a['fail_kj'] for a, b in zip(A, B)], '{:+.1f}')} | "
            f"{ms([b['spr'] - a['spr'] for a, b in zip(A, B)], '{:+.3f}')} |")
    return out


def plot(data, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("main arms", [c for _, c in S2.MAIN] + ["S2_gpu"]),
              ("plug-in pairs", ["S2_random", "S2_random_s0", "S2_wen_fresh", "S2_wen_fresh_s0",
                                 "S2_wen_fresh_gpu", "S2_wen_fresh_gpu_s0", "S2_polaris",
                                 "S2_polaris_s0"])]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for row, (title, cells) in zip(axes, groups):
        for cell in cells:
            ml = [data[(cell, s)] for s in SEEDS]
            tc = [([t for _, t, _ in m["curve"]], [a for _, _, a in m["curve"]]) for m in ml]
            F._band(row[0], tc, NAME[cell], np.linspace(0, max(max(x) for x, _ in tc), 600))
            ec = [([e for e, _ in m["energy_curve"]], [a for _, a in m["energy_curve"]])
                  for m in ml]
            F._band(row[1], ec, NAME[cell], np.linspace(0, max(max(x) for x, _ in ec), 600))
        for ax, xl in ((row[0], "online clock (s)"),
                       (row[1], "cumulative ground-truth energy, failed included (kJ)")):
            for thr in THR:
                ax.axhline(thr, color="grey", lw=0.6, ls=":")
            ax.set_xlabel(xl)
            ax.set_ylabel("test accuracy")
            ax.set_ylim(0.3, 0.95)
            ax.set_title(f"S2, 3 seeds (mean, min–max): {title}")
            ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "evidence" / "s2_final.md")
    args = parser.parse_args()
    data = {(cell, s): S2.metrics(cell, s) for _, cell in ARMS for s in SEEDS}
    bad = coverage_check(data)
    curves = args.out.with_name(args.out.stem + "_curves.png")
    plot(data, curves)
    lines = ["## 1. Main S2 table (mean ± sd over 3 seeds)", "", *table_main(data), "",
             "## 2. Selection behaviour (means over 3 seeds)", "", *table_behaviour(data), "",
             "## 3. Per seed", "", *per_seed(data), "",
             "## 4. Verdicts (v3 rule: the same direction in all 3 seeds; waste lower = STR "
             "higher, v3.6)", "", *verdicts(data), "",
             "## 5. Pairwise directions, Ours (gpu + Stage 0) vs each arm (reported, not "
             "predictions)", "",
             "A threshold Ours reaches and the other does not counts as Ours lower.", "",
             *pairwise(data), "",
             "## 6. Plug-in pairs, 3 seeds (B − A)", "", *plugin_pairs(data), "",
             "## 7. Curves", "", f"![S2 curves]({curves.name})", "",
             f"Coverage check: every dispatch of all {len(data)} runs inside "
             f"[{K.S2_COVERAGE_M[0]:.0f}, {K.S2_COVERAGE_M[1]:.0f}] m"
             + (": yes." if not bad else f": NO for {bad}."), ""]
    args.out.write_text("\n".join(lines))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
