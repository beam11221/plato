"""
Prompt F: Wen-faithful participation collapse, log-only diagnosis. No training.

Inputs (read only): the 6 wen runs (dispatch_log, wen_decisions, wen_candidates,
merge_log) and the dispatch / merge / results logs of the S1 and S2 random and
gpu runs. Outputs: one markdown file with all tables, and two plots.

Usage:
    python wen_env/wen_concentration.py --out wen_env/evidence/wen_concentration_v1.md \
        --plot-gini wen_env/evidence/wen_concentration_gini.png \
        --plot-f3 wen_env/evidence/wen_concentration_f3.png
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import wen_constants as K
import wen_faithful as W
import wen_partition
from wen_faithful_report import gini
from wen_ptable import SEEDS, run_dir

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SCENARIOS = ("S1", "S2")
ARMS = ("wen", "random", "gpu")
TIER_CODE = {"xavier": 0, "orin": 1, "thor": 2}
CHECK_ROUNDS = (250, 500, 1000, 2000, 4000)
T_MAX, E_MAX = 34.334345905647005, 36.423846640394224


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def counts_of(dispatch) -> Counter:
    return Counter(int(r["client_id"]) for r in dispatch)


# ---------------------------------------------------------------------------
# Step 1: per-vehicle table and Spearman correlations
# ---------------------------------------------------------------------------

def step1(sc, seed, arm="wen"):
    run = run_dir(f"{sc}_{arm}", seed)
    disp, cand, dec = (rows(run / f) for f in
                       ("dispatch_log.csv", "wen_candidates.csv", "wen_decisions.csv"))
    n = counts_of(disp)
    n_rand = counts_of(rows(run_dir(f"{sc}_random", seed) / "dispatch_log.csv"))
    n_gpu = counts_of(rows(run_dir(f"{sc}_gpu", seed) / "dispatch_log.csv"))
    as_cand = defaultdict(list)
    for r in cand:
        as_cand[int(r["client_id"])].append(r)
    picked = defaultdict(list)
    for d in dec:
        picked[int(d["pick"])].append(d)
    speed = {int(r["client_id"]): float(r["V_k"]) for r in disp}
    f4 = {c: float(rs[0]["F4"]) for c, rs in as_cand.items()}
    f4_rank = {c: i + 1 for i, c in enumerate(sorted(f4, key=f4.get))}
    by_count = sorted(range(1, 31), key=lambda c: -n.get(c, 0))
    count_rank = {c: i + 1 for i, c in enumerate(by_count)}
    table = []
    for c in range(1, K.N_CLIENTS + 1):
        cr = as_cand.get(c, [])
        pk = picked.get(c, [])
        table.append({
            "client_id": c, "tier": cr[0]["tier"] if cr else "?", "S": K.partition_size(c),
            "V": speed.get(c), "F4": f4.get(c), "F4_rank": f4_rank.get(c),
            "n": n.get(c, 0), "share": n.get(c, 0) / len(disp), "rank": count_rank[c],
            "mF1": st.fmean(float(r["F1"]) for r in cr) if cr else None,
            "mF2": st.fmean(float(r["F2"]) for r in cr) if cr else None,
            "mF3": st.fmean(float(r["F3"]) for r in cr) if cr else None,
            **{f"pick_nF{i}": (st.fmean(float(d[f"nF{i}"]) for d in pk) if pk else None)
               for i in range(1, 5)},
            "n_random": n_rand.get(c, 0), "n_gpu": n_gpu.get(c, 0),
        })
    rho = {}
    count = [t["n"] for t in table]
    for key, vals in (("F4 rank", [t["F4_rank"] for t in table]),
                      ("mean F3", [t["mF3"] for t in table]),
                      ("mean F1", [t["mF1"] for t in table]),
                      ("mean F2", [t["mF2"] for t in table]),
                      ("V_i", [t["V"] if t["V"] is not None else np.nan for t in table]),
                      ("tier", [TIER_CODE.get(t["tier"], np.nan) for t in table])):
        rho[key] = float(spearmanr(count, vals, nan_policy="omit").statistic)
    return table, rho


# ---------------------------------------------------------------------------
# Step 2: time course
# ---------------------------------------------------------------------------

def rolling_gini(dispatch, window=250):
    last = max(int(r["round"]) for r in dispatch)
    out = []
    for lo in range(1, last + 1, window):
        sub = [r for r in dispatch if lo <= int(r["round"]) < lo + window]
        c = counts_of(sub)
        out.append((lo + window - 1, gini([c.get(v, 0) for v in range(1, 31)])))
    return out


def onset(dispatch, threshold):
    """(first crossing, sustained crossing): rounds at which the run's final top
    vehicle's cumulative dispatch share first exceeds `threshold`, and from which
    it stays above it until the end."""
    top = counts_of(dispatch).most_common(1)[0][0]
    hits, total, first, sustained = 0, 0, None, None
    for r in dispatch:
        total += 1
        hits += int(r["client_id"]) == top
        above = hits / total > threshold
        rnd = int(r["round"])
        if above and first is None:
            first = rnd
        if above and sustained is None:
            sustained = rnd
        if not above:
            sustained = None
    return first, sustained


def f3_over_time(sc, seed, window=100, arm="wen"):
    run = run_dir(f"{sc}_{arm}", seed)
    disp = rows(run / "dispatch_log.csv")
    top5 = {c for c, _ in counts_of(disp).most_common(5)}
    acc = defaultdict(lambda: ([], []))
    for r in rows(run / "wen_candidates.csv"):
        w = (int(r["round"]) - 1) // window
        acc[w][0 if int(r["client_id"]) in top5 else 1].append(float(r["F3"]))
    return [((w + 1) * window, st.fmean(a) if a else None, st.fmean(b) if b else None)
            for w, (a, b) in sorted(acc.items())]


def round_loss(sc, arm, seed, window=100):
    """random / gpu: the round-level train_loss in results/*.csv (mean of the losses
    reported by that round's merged updates) -- per-vehicle losses are not logged."""
    f = next((run_dir(f"{sc}_{arm}", seed) / "results").glob("[0-9]*.csv"))
    acc = defaultdict(list)
    for r in rows(f):
        if r["train_loss"] not in ("", "None"):
            acc[(int(r["round"]) - 1) // window].append(float(r["train_loss"]))
    return [((w + 1) * window, st.fmean(v)) for w, v in sorted(acc.items())]


def min_objective_shares(sc, seed, window=500, arm="wen"):
    run = run_dir(f"{sc}_{arm}", seed)
    per_dec = defaultdict(list)
    for r in rows(run / "wen_candidates.csv"):
        per_dec[r["decision"]].append(r)
    acc = defaultdict(lambda: [0, 0, 0, 0, 0])
    for cands in per_dec.values():
        pick = next(c for c in cands if c["picked"] == "1")
        w = (int(pick["round"]) - 1) // window
        acc[w][4] += 1
        for i in range(4):
            if float(pick[f"F{i + 1}"]) == min(float(c[f"F{i + 1}"]) for c in cands):
                acc[w][i] += 1
    return [((w + 1) * window, [v[i] / v[4] for i in range(4)]) for w, v in sorted(acc.items())]


# ---------------------------------------------------------------------------
# Step 3: static counterfactual replay on logged decisions
# ---------------------------------------------------------------------------

def replay(obj: dict[int, tuple], keep: tuple[int, ...]) -> int:
    """W.decide with only the objectives in `keep`; constraints still use F1 / F2."""
    feasible = sorted(c for c, f in obj.items() if f[0] <= T_MAX and f[1] <= E_MAX)
    pool = {c: tuple(obj[c][i] for i in keep) for c in (feasible or sorted(obj))}
    pareto = W.non_dominated(pool)
    norm = W.normalise({c: pool[c] for c in pareto})
    score = {c: sum(v) / len(v) for c, v in norm.items()}
    return min(pareto, key=lambda c: (score[c], c))


class ReplayMismatch(SystemExit):
    pass


def step3(sc, seed, arm="wen"):
    run = run_dir(f"{sc}_{arm}", seed)
    per_dec = defaultdict(list)
    for r in rows(run / "wen_candidates.csv"):
        per_dec[int(r["decision"])].append(r)
    variants = {"all four": (0, 1, 2, 3), "drop F1": (1, 2, 3), "drop F2": (0, 2, 3),
                "drop F3": (0, 1, 3), "drop F4": (0, 1, 2)}
    picks = {k: Counter() for k in variants}
    changed = Counter()
    for dec_id in sorted(per_dec):
        cands = per_dec[dec_id]
        obj = {int(c["client_id"]): tuple(float(c[f"F{i}"]) for i in range(1, 5)) for c in cands}
        logged = next(int(c["client_id"]) for c in cands if c["picked"] == "1")
        full = replay(obj, variants["all four"])
        if full != logged or W.decide(obj, T_MAX, E_MAX).pick != logged:
            raise ReplayMismatch(f"{run.name} decision {dec_id}: logged {logged}, replay {full}")
        for name, keep in variants.items():
            p = replay(obj, keep)
            picks[name][p] += 1
            changed[name] += p != logged
    n = len(per_dec)
    return {name: (changed[name] / n, gini([picks[name].get(v, 0) for v in range(1, 31)]))
            for name in variants}, n


# ---------------------------------------------------------------------------
# Step 4: label coverage of the aggregated updates
# ---------------------------------------------------------------------------

_PARTS: dict[int, wen_partition.Partition] = {}


def partition(seed):
    if seed not in _PARTS:
        _PARTS[seed] = wen_partition.build(W.cifar10_train_targets(DATA), seed)
    return _PARTS[seed]


def step4(sc, arm, seed):
    run = run_dir(f"{sc}_{arm}", seed)
    part = partition(seed)
    merged = [r for r in rows(run / "merge_log.csv") if r["accepted"] == "1"]
    last = max(int(r["round"]) for r in merged)
    out = []
    for check in CHECK_ROUNDS:
        upto = min(check, last)
        sub = [r for r in merged if int(r["round"]) <= upto]
        hist = np.zeros(10)
        seen = set()
        for r in sub:
            c = int(r["client_id"])
            hist += part.counts[c - 1]
            seen.add(c)
        images = sum(len(part.indices[c]) for c in seen)
        out.append((check, upto, W.kl_to_uniform(hist), images / K.TRAIN_IMAGES, len(seen)))
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(v, f="{:.3f}"):
    return "—" if v is None else f.format(v)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot-gini", type=Path, required=True)
    parser.add_argument("--plot-f3", type=Path, required=True)
    args = parser.parse_args()
    L = []

    # Step 3 first: it is the stop condition.
    cf = {}
    for sc in SCENARIOS:
        for seed in SEEDS:
            cf[(sc, seed)] = step3(sc, seed)
    n_dec = sum(n for _, n in cf.values())
    L += ["## Replay test (Step 3 stop condition)", "",
          f"Replaying all {n_dec:,} logged wen decisions with all four objectives reproduces "
          "every logged pick exactly (and `wen_faithful.decide` agrees). **PASS.**", ""]

    L += ["## Step 1 — Spearman correlation of dispatch count with … (per seed)", "",
          "| scenario | seed | F4 rank | mean F3 | mean F1 | mean F2 | V_i | tier |",
          "|---|---|---|---|---|---|---|---|"]
    tables = {}
    for sc in SCENARIOS:
        for seed in SEEDS:
            t, rho = step1(sc, seed)
            tables[(sc, seed)] = t
            L.append(f"| {sc} | {seed} | " + " | ".join(f"{rho[k]:+.2f}" for k in
                     ("F4 rank", "mean F3", "mean F1", "mean F2", "V_i", "tier")) + " |")
    L += ["", "Tier coded xavier = 0, orin = 1, thor = 2; F4 rank 1 = lowest KL.", ""]
    for (sc, seed), t in tables.items():
        L += [f"### Per-vehicle table — {sc}, seed {seed}", "",
              "| client | tier | \\|S\\| | V (m/s) | F4 | F4 rank | dispatches | share | rank "
              "| mean F1 | mean F2 | mean F3 | pick nF1 | nF2 | nF3 | nF4 | random | gpu |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(t, key=lambda r: r["rank"]):
            L.append(f"| {r['client_id']} | {r['tier']} | {r['S']} | {fmt(r['V'], '{:.1f}')} | "
                     f"{fmt(r['F4'], '{:.4f}')} | {r['F4_rank']} | {r['n']} | {r['share']:.3f} | "
                     f"{r['rank']} | {fmt(r['mF1'], '{:.2f}')} | {fmt(r['mF2'], '{:.2f}')} | "
                     f"{fmt(r['mF3'], '{:.3f}')} | {fmt(r['pick_nF1'], '{:.2f}')} | "
                     f"{fmt(r['pick_nF2'], '{:.2f}')} | {fmt(r['pick_nF3'], '{:.2f}')} | "
                     f"{fmt(r['pick_nF4'], '{:.2f}')} | {r['n_random']} | {r['n_gpu']} |")
        L.append("")

    L += ["## Step 2.1 — onset of concentration (wen; random / gpu for reference)", "",
          "Top vehicle = the run's final top vehicle by dispatch count. 'First' = first round its "
          "cumulative share exceeds the threshold; 'sustained' = first round from which it stays "
          "above until the end (early rounds have tiny denominators, so 'first' is often round 1).",
          "", "| scenario | arm | seed | top vehicle share (final) | >10% first / sustained "
          "| >20% first / sustained |", "|---|---|---|---|---|---|"]
    gini_series = defaultdict(list)
    for sc in SCENARIOS:
        for arm in ARMS:
            for seed in SEEDS:
                d = rows(run_dir(f"{sc}_{arm}", seed) / "dispatch_log.csv")
                gini_series[(sc, arm)].append(rolling_gini(d))
                top = counts_of(d).most_common(1)[0][1] / len(d)
                o10, o20 = onset(d, 0.10), onset(d, 0.20)
                L.append(f"| {sc} | {arm} | {seed} | {top:.3f} | {o10[0]} / {o10[1] or '—'} | "
                         f"{o20[0] or '—'} / {o20[1] or '—'} |")
    L += ["", "### Rolling Gini (250-round windows; mean over 3 seeds)", "",
          "| scenario | arm | " + " | ".join(f"r≤{w}" for w in range(250, 4001, 250)) + " |",
          "|---|---|" + "---|" * 16]
    for (sc, arm), series in gini_series.items():
        by_w = defaultdict(list)
        for s in series:
            for w, g in s:
                by_w[w].append(g)
        L.append(f"| {sc} | {arm} | " + " | ".join(
            fmt(st.fmean(by_w[w]) if by_w.get(w) else None, "{:.2f}")
            for w in range(250, 4001, 250)) + " |")

    L += ["", "## Step 2.2 — F3 over time, wen: top-5 vehicles vs the rest "
          "(mean candidate F3 per 100-round window, mean over seeds; every 500 rounds shown)", "",
          "| scenario | round ≤ | top-5 F3 | rest F3 | random round-level loss | "
          "gpu round-level loss |", "|---|---|---|---|---|---|"]
    f3_plot = {}
    for sc in SCENARIOS:
        series = [f3_over_time(sc, s) for s in SEEDS]
        rl = {arm: [dict(round_loss(sc, arm, s)) for s in SEEDS] for arm in ("random", "gpu")}
        f3_plot[sc] = (series, rl)
        for w in range(500, 4001, 500):
            top = [v for s in series for (r, a, _b) in s if r == w and a is not None for v in [a]]
            rest = [v for s in series for (r, _a, b) in s if r == w and b is not None for v in [b]]
            ra = [d[w] for d in rl["random"] if w in d]
            ga = [d[w] for d in rl["gpu"] if w in d]
            L.append(f"| {sc} | {w} | {fmt(st.fmean(top) if top else None)} | "
                     f"{fmt(st.fmean(rest) if rest else None)} | "
                     f"{fmt(st.fmean(ra) if ra else None)} | {fmt(st.fmean(ga) if ga else None)} |")

    L += ["", "## Step 2.3 — share of wen picks that are the minimum of each objective "
          "among all candidates (500-round windows, mean over seeds)", "",
          "| scenario | round ≤ | F1-min | F2-min | F3-min | F4-min |", "|---|---|---|---|---|---|"]
    for sc in SCENARIOS:
        per = [dict(min_objective_shares(sc, s)) for s in SEEDS]
        for w in range(500, 4001, 500):
            v = [p[w] for p in per if w in p]
            if v:
                L.append(f"| {sc} | {w} | " + " | ".join(
                    f"{st.fmean(x[i] for x in v):.3f}" for i in range(4)) + " |")

    L += ["", "## Step 3 — static counterfactual on logged decisions", "",
          "Static replay on the logged candidate sets and logged F values: it shows which "
          "objective decides each pick, **not** what a rerun would do (it ignores feedback — "
          "e.g. F3 values would differ after a different pick).", "",
          "| scenario | seed | decisions | pick changes: drop F1 | drop F2 | drop F3 | drop F4 | "
          "Gini: logged (all four) | drop F1 | drop F2 | drop F3 | drop F4 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (sc, seed), (res, n) in cf.items():
        L.append(f"| {sc} | {seed} | {n} | " +
                 " | ".join(f"{res[k][0]:.3f}" for k in ("drop F1", "drop F2", "drop F3", "drop F4")) +
                 " | " + " | ".join(f"{res[k][1]:.3f}" for k in
                                     ("all four", "drop F1", "drop F2", "drop F3", "drop F4")) + " |")

    L += ["", "## Step 4 — label coverage of the aggregated (accepted, merged) updates", "",
          "Count-weighted by |S_i| (staleness ignored). KL is to uniform 0.1. Coverage = share of "
          "the 50,000 training images in ≥ 1 aggregated update. 'at round' = min(check, run's "
          "last merge round).", "",
          "| scenario | arm | seed | " + " | ".join(f"r{c}: KL / coverage / vehicles"
                                                    for c in CHECK_ROUNDS) + " |",
          "|---|---|---|" + "---|" * len(CHECK_ROUNDS)]
    for sc in SCENARIOS:
        for arm in ARMS:
            for seed in SEEDS:
                cells = [f"{kl:.4f} / {cov:.3f} / {nv}" + ("" if upto == c else f" (r{upto})")
                         for c, upto, kl, cov, nv in step4(sc, arm, seed)]
                L.append(f"| {sc} | {arm} | {seed} | " + " | ".join(cells) + " |")
    args.out.write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, sc in zip(axes, SCENARIOS):
        for arm in ARMS:
            by_w = defaultdict(list)
            for s in gini_series[(sc, arm)]:
                for w, g in s:
                    by_w[w].append(g)
            ws = sorted(by_w)
            ax.plot(ws, [st.fmean(by_w[w]) for w in ws], marker="o", ms=3, label=arm)
        ax.set_title(f"{sc}: rolling Gini of dispatch counts (250-round windows)")
        ax.set_xlabel("round")
        ax.set_ylabel("Gini")
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.plot_gini, dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, sc in zip(axes, SCENARIOS):
        series, rl = f3_plot[sc]
        for idx, label in ((1, "wen top-5 (candidate F3)"), (2, "wen rest (candidate F3)")):
            by_w = defaultdict(list)
            for s in series:
                for row in s:
                    if row[idx] is not None:
                        by_w[row[0]].append(row[idx])
            ws = sorted(by_w)
            ax.plot(ws, [st.fmean(by_w[w]) for w in ws], label=label)
        for arm in ("random", "gpu"):
            by_w = defaultdict(list)
            for d in rl[arm]:
                for w, v in d.items():
                    by_w[w].append(v)
            ws = sorted(by_w)
            ax.plot(ws, [st.fmean(by_w[w]) for w in ws], ls="--", label=f"{arm} (round-level loss)")
        ax.set_title(f"{sc}: reported training loss over rounds")
        ax.set_xlabel("round")
        ax.set_ylabel("loss")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.plot_f3, dpi=130)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
