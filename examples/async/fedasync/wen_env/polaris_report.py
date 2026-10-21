"""
Prompt J (Polaris in S1, seed 4126): mechanical report.

    --smoke     Step 4 table and per-50-round q statistics (accuracy not read)
    --study     Step 6: S1 table, Polaris internals, P34-P37, curves

Usage:
    python wen_env/polaris_report.py --smoke
    python wen_env/polaris_report.py --study --seed 4126 --out wen_env/evidence/polaris_s1_s4126.md
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import Counter
from pathlib import Path

import wen_concentration as C
import wen_constants as K
import wen_main_report as R
import wen_study1 as S1m
from wen_faithful_report import gini
from wen_ptable import run_dir

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs" / "wen_env"
TIERS = ("xavier", "orin", "thor")


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def dispatch_summary(disp: list[dict]) -> dict:
    rounds = max(int(r["round"]) for r in disp)
    counts = Counter(int(r["client_id"]) for r in disp)
    return {
        "s_per_round": max(float(r["t_finish_clean"]) for r in disp) / rounds,
        "share": {t: sum(r["tier"] == t for r in disp) / len(disp) for t in TIERS},
        "distinct": len(counts),
        "gini": gini([counts.get(c, 0) for c in range(1, K.N_CLIENTS + 1)]),
        "upload_dist": st.fmean(abs(float(r["x_finish"]) - K.RSU_X_M) for r in disp),
    }


def q_windows(dec: list[dict], width: int) -> list[dict]:
    out = []
    last = max(int(d["round"]) for d in dec)
    for lo in range(1, last + 1, width):
        w = [d for d in dec if lo <= int(d["round"]) < lo + width]
        if not w:
            continue
        G = [g for d in w for g in json.loads(d["G"])]
        tau = [t for d in w for t in json.loads(d["tau"])]
        out.append({"rounds": f"{lo}–{lo + width - 1}", "calls": len(w),
                    "max_q": st.median(float(d["max_q"]) for d in w),
                    "eff_pool": st.median(float(d["eff_pool"]) for d in w),
                    "G": (min(G), max(G)), "tau": (min(tau), max(tau))})
    return out


def render_smoke(run: Path) -> list[str]:
    disp = rows(run / "dispatch_log.csv")
    dec = rows(run / "polaris_decisions.csv")
    m = dispatch_summary(disp)
    share = " / ".join(f"{m['share'][t]:.2f}" for t in TIERS)
    out = [f"## Step 4 — smoke ({run.name}; accuracy not read)", "",
           "| s/round | share X / O / T | distinct vehicles | Gini | "
           "mean \\|x_upload − 1000\\| (m) | dispatches | selection calls |",
           "|---|---|---|---|---|---|---|",
           f"| {m['s_per_round']:.3f} | {share} | {m['distinct']} | {m['gini']:.3f} | "
           f"{m['upload_dist']:.0f} | {len(disp)} | {len(dec)} |", "",
           "| rounds | calls | median max q | median effective pool 1/Σq² | G range | τ range |",
           "|---|---|---|---|---|---|"]
    for w in q_windows(dec, 50):
        out.append(f"| {w['rounds']} | {w['calls']} | {w['max_q']:.4f} | {w['eff_pool']:.2f} | "
                   f"{w['G'][0]:.3f}–{w['G'][1]:.3f} | {w['tau'][0]:.2f}–{w['tau'][1]:.2f} |")
    return out


# ---------------------------------------------------------------------------
# Step 6
# ---------------------------------------------------------------------------

CELLS = ("S1_random", "S1_gpu", "S1_observed", "S1_location_only", "S1_wen",
         "S1_wen_fresh", "S1_polaris")
THRESHOLDS = (0.80, 0.85, 0.90)
TIER_RANK = {"xavier": 0, "orin": 1, "thor": 2}
# S1 gpu / cpu at seed 4126 hung at the disk-full incident; from Prompt F on they
# count as reaching 90% at their maximum accuracy (wen_faithful_v1_step6_footnote.md).
READING_B = {("S1_gpu", 4126), ("S1_cpu", 4126)}


def fmt(v, f="{:.3f}"):
    return "—" if v is None else f.format(v)


def coverage(arm: str, seed: int) -> dict:
    """Prompt F Step 4: share of the 50,000 images in aggregated updates."""
    by_check = {check: cov for check, _upto, _kl, cov, _n in C.step4("S1", arm, seed)}
    return {"r500": by_check[500], "r1000": by_check[1000], "end": by_check[4000]}


def s1_table(seed: int) -> tuple[list[str], dict]:
    out = ["| arm | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | s/round whole / <80% / ≥80% | "
           "energy to 80% (kJ) | share X / O / T r1–500 | share X / O / T last 500 | "
           "mean \\|x_upload − 1000\\| (m) | Gini | distinct | image coverage r500 / r1000 / end |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    data = {}
    for cell in CELLS:
        m = S1m.one(cell, seed)
        if (cell, seed) in READING_B and m["tta"][0.90] is None:
            curve = R.accuracy_curve(run_dir(cell, seed))
            best = max(a for _, _, a in curve)
            m["tta"][0.90] = next(t for _, t, a in curve if a == best)
        disp = rows(run_dir(cell, seed) / "dispatch_log.csv")
        m["distinct"] = len({r["client_id"] for r in disp})
        m["cov"] = coverage(cell.split("_", 1)[1], seed)
        data[cell] = m
        first = " / ".join(f"{m[f'first_{t}']:.2f}" for t in TIERS)
        last = " / ".join(f"{m[f'last_{t}']:.2f}" for t in TIERS)
        e80 = None if m["e80"] is None else m["e80"] / 1000
        out.append(
            f"| {cell.split('_', 1)[1]} | "
            + " | ".join(fmt(m["tta"][t], "{:.0f}") for t in THRESHOLDS)
            + f" | {m['spr']:.3f} / {fmt(m['before'])} / {fmt(m['after'])} | {fmt(e80, '{:.1f}')}"
            f" | {first} | {last} | {m['upload_dist']:.0f} | {m['gini']:.3f} | {m['distinct']}"
            f" | {m['cov']['r500']:.3f} / {m['cov']['r1000']:.3f} / {m['cov']['end']:.3f} |")
    return out, data


def internals(seed: int) -> list[str]:
    from scipy.stats import spearmanr

    run = run_dir("S1_polaris", seed)
    dec = rows(run / "polaris_decisions.csv")
    disp = rows(run / "dispatch_log.csv")
    out = ["| rounds | calls | median max q | median effective pool 1/Σq² | G range | τ range |",
           "|---|---|---|---|---|---|"]
    for w in q_windows(dec, 500):
        out.append(f"| {w['rounds']} | {w['calls']} | {w['max_q']:.4f} | {w['eff_pool']:.2f} | "
                   f"{w['G'][0]:.3f}–{w['G'][1]:.3f} | {w['tau'][0]:.2f}–{w['tau'][1]:.2f} |")
    # Per-vehicle means over every selection call in which the vehicle was a candidate.
    sums = {c: [0.0, 0.0, 0.0, 0] for c in range(1, K.N_CLIENTS + 1)}
    for d in dec:
        for c, g, t in zip(json.loads(d["pool"]), json.loads(d["G"]), json.loads(d["tau"])):
            s = sums[c]
            s[0] += g
            s[1] += t
            s[2] += g / (1.0 + t) ** 0.5
            s[3] += 1
    counts = Counter(int(r["client_id"]) for r in disp)
    ids = range(1, K.N_CLIENTS + 1)
    n = [counts.get(c, 0) for c in ids]
    tier = {int(r["client_id"]): r["tier"] for r in disp}
    cols = {"mean G": [sums[c][0] / sums[c][3] for c in ids],
            "mean τ": [sums[c][1] / sums[c][3] for c in ids],
            "tier (X=0, O=1, T=2)": [TIER_RANK[tier[c]] for c in ids],
            "mean V = G/√(1+τ)": [sums[c][2] / sums[c][3] for c in ids]}
    out += ["", "Spearman correlation of per-vehicle dispatch count (30 vehicles) with:", "",
            "| quantity | ρ | p |", "|---|---|---|"]
    for name, v in cols.items():
        r = spearmanr(n, v)
        out.append(f"| {name} | {r.statistic:.3f} | {r.pvalue:.3g} |")
    return out


def p_table(d) -> list[str]:
    def row(p, txt, ok, a, b):
        return (f"| {p} | {txt} | {a} vs {b} | {'as predicted' if ok else 'not as predicted'} | "
                "single seed — not evaluated |")
    g, q = d["S1_gpu"], d["S1_polaris"]
    tta = lambda m: fmt(m["tta"][0.80], "{:.0f}")
    return ["| P | comparison | seed (gpu vs polaris) | direction | status |",
            "|---|---|---|---|---|",
            row("P34", "gpu has lower s/round than polaris", g["spr"] < q["spr"],
                f"{g['spr']:.3f}", f"{q['spr']:.3f}"),
            row("P35", "gpu has lower TTA@80% than polaris",
                S1m.lower(g["tta"][0.80], q["tta"][0.80]), tta(g), tta(q)),
            row("P36", "gpu uses less ground-truth energy to reach 80% than polaris",
                S1m.lower(g["e80"], q["e80"]), fmt(g["e80"] and g["e80"] / 1000, "{:.1f} kJ"),
                fmt(q["e80"] and q["e80"] / 1000, "{:.1f} kJ")),
            row("P37", "polaris has a lower whole-run dispatch Gini than gpu",
                q["gini"] < g["gini"], f"{g['gini']:.3f}", f"{q['gini']:.3f}")]


def plot(seed: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for cell in CELLS:
        curve = R.accuracy_curve(run_dir(cell, seed))
        ax.plot([p[1] for p in curve], [p[2] for p in curve],
                lw=1.6 if cell == "S1_polaris" else 0.9, label=cell)
    for thr in THRESHOLDS:
        ax.axhline(thr, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("online clock (s)")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"S1, seed {seed}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def render_study(seed: int, out_path: Path) -> list[str]:
    table, data = s1_table(seed)
    curves = out_path.with_name(out_path.stem + "_curves.png")
    plot(seed, curves)
    return [f"## 1. S1 table, seed {seed}", "", *table, "",
            f"## 2. Polaris internals over time (seed {seed})", "", *internals(seed), "",
            "## 3. P34–P37", "", *p_table(data), "",
            "## 4. Curves", "", f"![accuracy vs online clock]({curves.name})", ""]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", default="S1_polaris_s4126_smoke")
    parser.add_argument("--study", action="store_true")
    parser.add_argument("--seed", type=int, default=4126)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.smoke:
        print("\n".join(render_smoke(RUNS / args.run)))
    if args.study:
        out = args.out or HERE / "evidence" / f"polaris_s1_s{args.seed}_tables.md"
        out.write_text("\n".join(render_study(args.seed, out)) + "\n")
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
