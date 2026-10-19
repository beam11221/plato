"""
Prompt E, Step 7: Study 1 (ours vs Wen-faithful vs random). Mechanical only.

Writes the Study 1 table, the constraint table, the participation table, the
P15-P21 table and the accuracy-vs-time curves.

Usage:
    python wen_env/wen_study1.py --out wen_env/evidence/wen_faithful_v1_tables.md \
        --plot wen_env/evidence/wen_faithful_v1_curves.png
"""

from __future__ import annotations

import argparse
import statistics as st
from collections import Counter
from pathlib import Path

import wen_constants as K
import wen_main_report as R
from wen_faithful_report import circ, gini
from wen_ptable import SEEDS, run_dir

TIERS = ("xavier", "orin", "thor")
THRESHOLDS = (0.80, 0.85, 0.90)
S1_ROWS = ("S1_random", "S1_gpu", "S1_wen", "S1_location_only", "S1_cpu")
S2_ROWS = ("S2_random", "S2_gpu", "S2_wen")


def _f(r, k):
    return float(r[k])


def x_hat(r, dec_row=None):
    """The arm's own predicted upload position, or None (random / observed)."""
    if dec_row is not None:
        return float(dec_row["x_hat_upload"])
    if r["T_hat_train"] == "":
        return None
    return (_f(r, "x_dispatch") + _f(r, "V_k") * _f(r, "T_hat_train")) % K.ROAD_LENGTH_M


def one(cell: str, seed: int) -> dict:
    run = run_dir(cell, seed)
    m = R.one_run(run)
    d = R._rows(run / "dispatch_log.csv")
    dec = R._rows(run / "wen_decisions.csv") if cell.endswith("_wen") else None
    curve = m["curve"]
    tta = {thr: next((t for _, t, a in curve if a >= thr), None) for thr in THRESHOLDS}
    last_round = max(int(r["round"]) for r in d)
    first = [r for r in d if int(r["round"]) <= 500]
    last = [r for r in d if int(r["round"]) > last_round - 500]
    errs = [circ(xh, _f(r, "x_finish"))
            for i, r in enumerate(d)
            if (xh := x_hat(r, dec[i] if dec else None)) is not None]
    wasted = [r for r in d if r["accepted"] == "0"]
    per_vehicle = Counter(int(r["client_id"]) for r in d)
    counts = [per_vehicle.get(c, 0) for c in range(1, K.N_CLIENTS + 1)]
    return {
        "tta": tta, "last_acc": curve[-1][2],
        "spr": m["s_per_round"], "before": m["s_before"], "after": m["s_after"],
        "e80": m["energy_to_80"],
        **{f"first_{t}": sum(r["tier"] == t for r in first) / len(first) for t in TIERS},
        **{f"last_{t}": sum(r["tier"] == t for r in last) / len(last) for t in TIERS},
        "thor_whole": sum(r["tier"] == "thor" for r in d) / len(d),
        "waste": len(wasted) / len(d),
        "waste_gpu_s": sum(_f(r, "T_train_true") for r in wasted),
        "waste_j": sum(_f(r, "E_train") + _f(r, "E_up") for r in wasted),
        "upload_dist": st.fmean(abs(_f(r, "x_finish") - K.RSU_X_M) for r in d),
        "xhat_err": st.fmean(errs) if errs else None,
        "pmin": min(counts), "pmax": max(counts), "pzero": counts.count(0),
        "gini": gini(counts),
        "dec": dec,
    }


def agg(vals, fmt="{:.3g}"):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return fmt.format(vals[0])
    return f"{fmt.format(st.fmean(vals))} ± {fmt.format(st.stdev(vals))}"


def tta_cell(rs, thr):
    hit = [r["tta"][thr] for r in rs if r["tta"][thr] is not None]
    miss = len(rs) - len(hit)
    txt = agg(hit, "{:.0f}") if hit else "—"
    if miss:
        lasts = ", ".join(f"{r['last_acc']:.3f}" for r in rs if r["tta"][thr] is None)
        txt += f" ({miss}/{len(rs)} not reached; last {lasts})"
    return txt


def study_table(data, rows) -> list[str]:
    L = ["| arm | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | s/round | s/round before / after 80% "
         "| energy to 80% (kJ) | share X/O/T rounds 1–500 | share X/O/T last 500 "
         "| waste (disp / GPU-s / kJ) | mean \\|x_upload − 1000\\| (m) | mean \\|x̂ − x\\| (m) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell in rows:
        rs = data[cell]
        share = lambda pre: " / ".join(agg([r[f"{pre}_{t}"] for r in rs], "{:.2f}")  # noqa: E731
                                       for t in TIERS)
        waste = ("—" if cell.startswith("S1") else
                 f"{agg([r['waste'] for r in rs], '{:.3f}')} / "
                 f"{agg([r['waste_gpu_s'] for r in rs], '{:.0f}')} / "
                 f"{agg([r['waste_j'] / 1e3 for r in rs], '{:.1f}')}")
        L.append(
            f"| {cell} | {tta_cell(rs, 0.80)} | {tta_cell(rs, 0.85)} | {tta_cell(rs, 0.90)} | "
            f"{agg([r['spr'] for r in rs], '{:.3f}')} | "
            f"{agg([r['before'] for r in rs], '{:.3f}')} / {agg([r['after'] for r in rs], '{:.3f}')} | "
            f"{agg([r['e80'] / 1e3 if r['e80'] else None for r in rs], '{:.1f}')} | "
            f"{share('first')} | {share('last')} | {waste} | "
            f"{agg([r['upload_dist'] for r in rs], '{:.0f}')} | "
            f"{agg([r['xhat_err'] for r in rs], '{:.0f}')} |")
    return L


def constraint_table(data, cells=("S1_wen", "S2_wen")) -> list[str]:
    L = ["| scenario | tier | cut by T only | cut by E only | cut by both | fallback rate "
         "| mean n_feasible | mean n_pareto |", "|---|---|---|---|---|---|---|---|"]
    for cell in cells:
        decs = [row for r in data[cell] for row in r["dec"]]
        for t in TIERS:
            n = max(1, sum(int(d[f"n_candidates_{t}"]) for d in decs))
            cuts = [sum(int(d[f"n_cut_{k}_{t}"]) for d in decs) / n
                    for k in ("T_only", "E_only", "both")]
            L.append(f"| {cell[:2]} | {t} | {cuts[0]:.3f} | {cuts[1]:.3f} | {cuts[2]:.3f} | "
                     f"{st.fmean(int(d['fallback']) for d in decs):.4f} | "
                     f"{st.fmean(int(d['n_feasible']) for d in decs):.2f} | "
                     f"{st.fmean(int(d['n_pareto']) for d in decs):.2f} |")
    return L


def participation_table(data) -> list[str]:
    L = ["| arm | min dispatches per vehicle | max | vehicles with 0 | Gini |",
         "|---|---|---|---|---|"]
    for cell in ("S1_random", "S1_gpu", "S1_wen", "S2_random", "S2_gpu", "S2_wen"):
        rs = data[cell]
        L.append(f"| {cell} | {agg([r['pmin'] for r in rs], '{:.0f}')} | "
                 f"{agg([r['pmax'] for r in rs], '{:.0f}')} | "
                 f"{agg([r['pzero'] for r in rs], '{:.1f}')} | "
                 f"{agg([r['gini'] for r in rs], '{:.3f}')} |")
    return L


def lower(a, b):
    """a < b with 'not reached' (None) treated as +infinity."""
    inf = float("inf")
    return (inf if a is None else a) < (inf if b is None else b)


def p_table(data) -> list[str]:
    checks = [
        ("P15", "wen mean |x_upload − 1000| > location_only's (S1)",
         lambda i: (data["S1_wen"][i]["upload_dist"] > data["S1_location_only"][i]["upload_dist"],
                    f"{data['S1_wen'][i]['upload_dist']:.0f} vs {data['S1_location_only'][i]['upload_dist']:.0f} m")),
        ("P16", "gpu s/round < wen (S1)",
         lambda i: (data["S1_gpu"][i]["spr"] < data["S1_wen"][i]["spr"],
                    f"{data['S1_gpu'][i]['spr']:.3f} vs {data['S1_wen'][i]['spr']:.3f}")),
        ("P17", "gpu TTA@80% < wen (S1)",
         lambda i: (lower(data["S1_gpu"][i]["tta"][0.80], data["S1_wen"][i]["tta"][0.80]),
                    f"{data['S1_gpu'][i]['tta'][0.80]:.0f} vs {data['S1_wen'][i]['tta'][0.80]:.0f} s")),
        ("P18", "energy to 80%: gpu < wen (S1)",
         lambda i: (lower(data["S1_gpu"][i]["e80"], data["S1_wen"][i]["e80"]),
                    f"{data['S1_gpu'][i]['e80'] / 1e3:.1f} vs {data['S1_wen'][i]['e80'] / 1e3:.1f} kJ")),
        ("P19", "wen Thor share (whole run) < gpu's (S1)",
         lambda i: (data["S1_wen"][i]["thor_whole"] < data["S1_gpu"][i]["thor_whole"],
                    f"{data['S1_wen'][i]['thor_whole']:.3f} vs {data['S1_gpu'][i]['thor_whole']:.3f}")),
        ("P20", "wen waste fraction < random's (S2)",
         lambda i: (data["S2_wen"][i]["waste"] < data["S2_random"][i]["waste"],
                    f"{data['S2_wen'][i]['waste']:.3f} vs {data['S2_random'][i]['waste']:.3f}")),
        ("P21", "gpu (Stage 0 off) TTA@80% < wen (S2)",
         lambda i: (lower(data["S2_gpu"][i]["tta"][0.80], data["S2_wen"][i]["tta"][0.80]),
                    f"{data['S2_gpu'][i]['tta'][0.80]:.0f} vs {data['S2_wen'][i]['tta'][0.80]:.0f} s")),
    ]
    L = ["| P | comparison | seed 4126 | seed 4127 | seed 4128 | verdict |",
         "|---|---|---|---|---|---|"]
    for p, txt, fn in checks:
        res = [fn(i) for i in range(len(SEEDS))]
        cells = [("✓ " if ok else "✗ ") + t for ok, t in res]
        verdict = "**held**" if all(ok for ok, _ in res) else "**not held**"
        L.append(f"| {p} | {txt} | " + " | ".join(cells) + f" | {verdict} |")
    return L


def plot(data, path: Path, arms=("random", "gpu", "wen")) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, sc in zip(axes, ("S1", "S2")):
        for arm in arms:
            curves = [R.accuracy_curve(run_dir(f"{sc}_{arm}", s)) for s in SEEDS]
            end = min(c[-1][1] for c in curves)
            grid = np.linspace(0, end, 500)
            ys = np.array([np.interp(grid, [p[1] for p in c], [p[2] for p in c]) for c in curves])
            ax.plot(grid, ys.mean(axis=0), label=arm)
            ax.fill_between(grid, ys.min(axis=0), ys.max(axis=0), alpha=0.2)
        for thr in THRESHOLDS:
            ax.axhline(thr, color="grey", lw=0.6, ls="--")
        ax.set_title(sc)
        ax.set_xlabel("online clock (s)")
        ax.set_ylabel("test accuracy")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    data = {c: [one(c, s) for s in SEEDS] for c in S1_ROWS + S2_ROWS}
    text = "\n".join(
        ["## 1. Study 1 — S1 (mean ± SD over 3 seeds)", "", *study_table(data, S1_ROWS), "",
         "## 1. Study 1 — S2 (Stage 0 off for every arm)", "", *study_table(data, S2_ROWS), "",
         "## 2. Constraint table (all wen decisions per scenario, 3 seeds pooled)", "",
         *constraint_table(data), "",
         "## 3. Participation (per-vehicle dispatch counts; mean ± SD over seeds)", "",
         *participation_table(data), "",
         "## 4. P15–P21 (v3 rule: same direction in all 3 seeds)", "", *p_table(data), ""])
    print(text)
    args.out.write_text(text)
    plot(data, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
