"""
Prompt H: log-only parts of the wen_fresh report (no new runs).

Step 0  Replays the `wen` arm's F3 prior from its own logs. Each accepted
        report's loss is recovered as the reporting vehicle's F3 at the first
        decision after its merge (wen_candidates.csv); the prior at a decision
        is then the mean of every loss merged before it. The script checks
        that this equals the logged F3 of every never-trained candidate,
        bit for bit.
Step 6  Extends the Prompt F static counterfactual (wen_concentration.replay) with
        "F1 + F2 only" and "F3 + F4 only", on the logged wen decisions.

Usage:
    python wen_env/wen_fresh_report.py --step0 --step6 --out /tmp/x.md
"""

from __future__ import annotations

import argparse
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import wen_faithful as W
import wen_main_report as R
import wen_study1 as S1m
from wen_concentration import (CHECK_ROUNDS, SCENARIOS, T_MAX, E_MAX, ReplayMismatch,
                               counts_of, f3_over_time, fmt, onset, replay, rolling_gini,
                               rows, step1, step3, step4)
from wen_faithful_report import circ, gini, render_smoke, smoke_row
from wen_ptable import SEEDS, run_dir


# ---------------------------------------------------------------------------
# Step 0: the wen F3 prior, replayed from the logs
# ---------------------------------------------------------------------------

def prior_replay(run: Path) -> dict:
    cands = defaultdict(list)
    for r in rows(run / "wen_candidates.csv"):
        cands[int(r["decision"])].append(r)
    merges = rows(run / "merge_log.csv")
    dec_cycle = [int(d["cycle"]) for d in rows(run / "dispatch_log.csv")]

    losses, pending, mi = [], [], 0          # losses: (merge cycle, client, loss)
    for i, cyc in enumerate(dec_cycle, 1):
        while mi < len(merges) and int(merges[mi]["cycle"]) <= cyc:
            if merges[mi]["accepted"] == "1":
                pending.append(int(merges[mi]["client_id"]))
            mi += 1
        f3 = {int(c["client_id"]): float(c["F3"]) for c in cands[i]}
        losses += [(cyc, cid, f3[cid]) for cid in pending if cid in f3]
        pending = [cid for cid in pending if cid not in f3]

    seen, total, n, li = set(), 0.0, 0, 0
    checked = mismatched = 0
    per_vehicle = defaultdict(list)
    trajectory = {}
    for i, cyc in enumerate(dec_cycle, 1):
        while li < len(losses) and losses[li][0] <= cyc:
            total += losses[li][2]
            n += 1
            seen.add(losses[li][1])
            li += 1
        prior = total / n if n else 0.0
        for c in cands[i]:
            cid = int(c["client_id"])
            per_vehicle[cid].append(float(c["F3"]))
            if cid not in seen:
                checked += 1
                mismatched += float(c["F3"]) != prior
        trajectory[i] = (cyc, n, prior)
    return {"run": run.name, "losses": losses, "unrecovered": len(pending),
            "checked": checked, "mismatched": mismatched, "per_vehicle": per_vehicle,
            "trajectory": trajectory, "dispatch": rows(run / "dispatch_log.csv")}


def vehicle_line(r: dict, cid: int) -> str:
    vals = r["per_vehicle"][cid]
    picks = [int(d["round"]) for d in r["dispatch"] if int(d["client_id"]) == cid]
    own = [round(loss, 4) for _, c, loss in r["losses"] if c == cid]
    return (f"| {cid} | {r['dispatch'][0]['scenario']} | {len(vals)} | {st.fmean(vals):.4f} | "
            f"{len(picks)} | {picks[:4]} | {own[:4]} |")


def render_step0(example_run: Path, vehicles: tuple[int, ...]) -> list[str]:
    out = ["## 0. The `wen` F3 prior (Step 0)", "",
           "| run | accepted reports recovered | unrecovered | never-trained candidate rows "
           "checked | rows where logged F3 ≠ replayed prior |", "|---|---|---|---|---|"]
    for sc in SCENARIOS:
        for seed in SEEDS:
            r = prior_replay(run_dir(f"{sc}_wen", seed))
            out.append(f"| {r['run']} | {len(r['losses'])} | {r['unrecovered']} | "
                       f"{r['checked']} | {r['mismatched']} |")
    r = prior_replay(example_run)
    out += ["", f"Per-vehicle example, `{r['run']}`:", "",
            "| client | scenario | candidate rows | mean F3 as a candidate | dispatches | "
            "first dispatch rounds | its own reported losses |", "|---|---|---|---|---|---|---|"]
    out += [vehicle_line(r, c) for c in vehicles]
    marks = (10, 50, 100, 500, 1000, 2000, 3000, len(r["trajectory"]))
    out += ["", f"Prior over the run, `{r['run']}` (decision → round, losses averaged, prior):", "",
            "| decision | round | losses averaged | prior |", "|---|---|---|---|"]
    for i in marks:
        cyc, n, prior = r["trajectory"][i]
        out.append(f"| {i} | {r['dispatch'][i - 1]['round']} | {n} | {prior:.4f} |")
    return out


# ---------------------------------------------------------------------------
# Step 6: two more static counterfactuals on the logged wen decisions
# ---------------------------------------------------------------------------

VARIANTS = {"all four": (0, 1, 2, 3), "F1 + F2 only": (0, 1), "F3 + F4 only": (2, 3)}


def step6(sc: str, seed: int):
    run = run_dir(f"{sc}_wen", seed)
    per_dec = defaultdict(list)
    for r in rows(run / "wen_candidates.csv"):
        per_dec[int(r["decision"])].append(r)
    picks = {k: Counter() for k in VARIANTS}
    changed = Counter()
    for dec_id in sorted(per_dec):
        cands = per_dec[dec_id]
        obj = {int(c["client_id"]): tuple(float(c[f"F{i}"]) for i in range(1, 5))
               for c in cands}
        logged = next(int(c["client_id"]) for c in cands if c["picked"] == "1")
        if replay(obj, VARIANTS["all four"]) != logged or \
                W.decide(obj, T_MAX, E_MAX).pick != logged:
            raise ReplayMismatch(f"{run.name} decision {dec_id}: replay != logged {logged}")
        for name, keep in VARIANTS.items():
            p = replay(obj, keep)
            picks[name][p] += 1
            changed[name] += p != logged
    n = len(per_dec)
    return {name: (changed[name] / n, gini([picks[name].get(v, 0) for v in range(1, 31)]))
            for name in VARIANTS}, n


def render_step6() -> list[str]:
    out = ["## 6. Static counterfactual: F1 + F2 only, F3 + F4 only (Step 6)", "",
           "Static replay on the logged candidate sets and logged F values (constraints "
           "unchanged; ignores feedback, as in Prompt F Step 3). The all-four replay "
           "reproduced every logged pick.", "",
           "| scenario | seed | decisions | pick changes: F1 + F2 only | F3 + F4 only | "
           "Gini: logged (all four) | F1 + F2 only | F3 + F4 only |",
           "|---|---|---|---|---|---|---|---|"]
    for sc in SCENARIOS:
        for seed in SEEDS:
            res, n = step6(sc, seed)
            out.append(f"| {sc} | {seed} | {n} | {res['F1 + F2 only'][0]:.3f} | "
                       f"{res['F3 + F4 only'][0]:.3f} | {res['all four'][1]:.3f} | "
                       f"{res['F1 + F2 only'][1]:.3f} | {res['F3 + F4 only'][1]:.3f} |")
    return out


# ---------------------------------------------------------------------------
# Step 3 test 6 and Step 4 smoke
# ---------------------------------------------------------------------------

def eval_times(run: Path) -> list[float]:
    return [float(r["fresh_eval_s"]) for r in rows(run / "wen_fresh_eval.csv")]


def render_eval_time(runs: list[Path], full_rounds: int = 4000) -> list[str]:
    out = ["| run | evaluated versions | mean fresh_eval_s | max | merges | "
           "projected extra wall time per 4,000-round run (h) |", "|---|---|---|---|---|---|"]
    for run in runs:
        ts = eval_times(run)
        merges = len(rows(run / "merge_log.csv"))
        out.append(f"| {run.name} | {len(ts)} | {st.fmean(ts):.3f} | {max(ts):.3f} | {merges} | "
                   f"{full_rounds * st.fmean(ts) / 3600:.2f} |")
    return out


def f3_range(run: Path, at_round: int) -> tuple[int, float, float]:
    """Fleet F3 range of the last evaluated version at or before `at_round`."""
    ev = [r for r in rows(run / "wen_fresh_eval.csv") if int(r["round"]) <= at_round][-1]
    vals = [float(ev[f"F3_{c}"]) for c in range(1, 31)]
    return int(ev["global_version"]), min(vals), max(vals)


def render_smoke_fresh(runs: list[Path]) -> list[str]:
    out = [render_smoke([smoke_row(r) for r in runs]).replace(
        "# Prompt E, Step 4: Wen smoke", "### Prompt H, Step 4: wen_fresh smoke"), "",
        "Rolling Gini of dispatch counts (50-round windows, rounds 1–200):", "",
        "| run | 1–50 | 51–100 | 101–150 | 151–200 |", "|---|---|---|---|---|"]
    for run in runs:
        g = rolling_gini(rows(run / "dispatch_log.csv"), window=50)[:4]
        out.append(f"| {run.name} | " + " | ".join(f"{v:.3f}" for _, v in g) + " |")
    out += ["", "Fleet F3 range (fresh loss of the global model, all 30 vehicles):", "",
            "| run | round 50 (version) | round 100 | round 200 |", "|---|---|---|---|"]
    for run in runs:
        cells = [f"{lo:.3f}–{hi:.3f} (v{v})" for v, lo, hi in
                 (f3_range(run, r) for r in (50, 100, 200))]
        out.append(f"| {run.name} | " + " | ".join(cells) + " |")
    out += ["", "Fresh-evaluation time:", ""] + render_eval_time(runs)
    return out


# ---------------------------------------------------------------------------
# Step 7: study, concentration, constraint, P22-P25, curves, eval time
# ---------------------------------------------------------------------------

S1_ROWS = ("S1_random", "S1_gpu", "S1_wen", "S1_wen_fresh", "S1_location_only", "S1_cpu")
S2_ROWS = ("S2_random", "S2_gpu", "S2_wen", "S2_wen_fresh")
CONC_ARMS = ("wen_fresh", "wen", "random", "gpu")
# Disk-full hangs counted as reaching 90% (Reading B, wen_faithful_v1_step6_footnote.md).
READING_B = {("S1_gpu", 4126), ("S1_cpu", 4126)}


def one(cell: str, seed: int) -> dict:
    m = S1m.one(cell, seed)
    run = run_dir(cell, seed)
    if cell.endswith("_wen_fresh"):
        disp, dec = rows(run / "dispatch_log.csv"), rows(run / "wen_decisions.csv")
        if len(dec) != len(disp) or any(a["pick"] != b["client_id"] for a, b in zip(dec, disp)):
            raise SystemExit(f"{run}: decisions do not line up with dispatches")
        m["xhat_err"] = st.fmean(circ(float(a["x_hat_upload"]), float(b["x_finish"]))
                                 for a, b in zip(dec, disp))
        m["dec"] = dec
    if (cell, seed) in READING_B and m["tta"][0.90] is None:
        curve = R.accuracy_curve(run)
        best = max(a for _, _, a in curve)
        m["tta"][0.90] = next(t for _, t, a in curve if a == best)
    return m


def p_table(data) -> list[str]:
    def g(c, i):
        return data[c][i]["gini"]

    def tta(c, i):
        return data[c][i]["tta"][0.80]

    checks = []
    for sc in SCENARIOS:
        checks.append(("P22", sc, "wen_fresh Gini < wen",
                       lambda i, sc=sc: (g(f"{sc}_wen_fresh", i) < g(f"{sc}_wen", i),
                                         f"{g(f'{sc}_wen_fresh', i):.3f} vs {g(f'{sc}_wen', i):.3f}")))
    for sc in SCENARIOS:
        checks.append(("P23", sc, "wen_fresh Gini > random",
                       lambda i, sc=sc: (g(f"{sc}_wen_fresh", i) > g(f"{sc}_random", i),
                                         f"{g(f'{sc}_wen_fresh', i):.3f} vs {g(f'{sc}_random', i):.3f}")))
    for p, sc in (("P24", "S1"), ("P25", "S2")):
        checks.append((p, sc, "gpu TTA@80% < wen_fresh" + (" (Stage 0 off)" if sc == "S2" else ""),
                       lambda i, sc=sc: (S1m.lower(tta(f"{sc}_gpu", i), tta(f"{sc}_wen_fresh", i)),
                                         f"{fmt(tta(f'{sc}_gpu', i), '{:.0f}')} vs "
                                         f"{fmt(tta(f'{sc}_wen_fresh', i), '{:.0f}')} s")))
    out = ["| P | scenario | comparison | " + " | ".join(f"seed {s}" for s in SEEDS)
           + f" | all {len(SEEDS)} seed(s) |", "|---|---|---|" + "---|" * (len(SEEDS) + 1)]
    held = defaultdict(list)
    for p, sc, txt, fn in checks:
        res = [fn(i) for i in range(len(SEEDS))]
        ok = all(r[0] for r in res)
        held[p].append(ok)
        out.append(f"| {p} | {sc} | {txt} | " + " | ".join(("✓ " if r[0] else "✗ ") + r[1]
                                                          for r in res) + f" | {'yes' if ok else 'no'} |")
    out += ["", "| P | verdict |", "|---|---|"]
    if len(SEEDS) < 3:
        out += [f"| {p} | direction {'as predicted' if all(v) else 'not as predicted'} in seed "
                f"{', '.join(map(str, SEEDS))}; **no v3 verdict** (the rule needs 3 seeds) |"
                for p, v in held.items()]
    else:
        out += [f"| {p} | {'**held**' if all(v) else '**not held**'} |" for p, v in held.items()]
    return out


def concentration(data) -> tuple[list[str], dict]:
    out = ["### Whole-run dispatch Gini (per seed) and per-vehicle extremes", "",
           "| scenario | arm | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean | min / max "
           "dispatches (mean) | vehicles with 0 (mean) |", "|---|---|" + "---|" * (len(SEEDS) + 3)]
    for sc in SCENARIOS:
        for arm in CONC_ARMS:
            rs = data[f"{sc}_{arm}"]
            out.append(f"| {sc} | {arm} | " + " | ".join(f"{r['gini']:.3f}" for r in rs) +
                       f" | {st.fmean(r['gini'] for r in rs):.3f} | "
                       f"{st.fmean(r['pmin'] for r in rs):.1f} / {st.fmean(r['pmax'] for r in rs):.1f} | "
                       f"{st.fmean(r['pzero'] for r in rs):.1f} |")
    out += ["", "### Onset (final top vehicle's cumulative share; first / sustained round)", "",
            "| scenario | arm | seed | top share | >10% | >20% |", "|---|---|---|---|---|---|"]
    gini_series = defaultdict(list)
    for sc in SCENARIOS:
        for arm in CONC_ARMS:
            for seed in SEEDS:
                d = rows(run_dir(f"{sc}_{arm}", seed) / "dispatch_log.csv")
                gini_series[(sc, arm)].append(rolling_gini(d))
                o10, o20 = onset(d, 0.10), onset(d, 0.20)
                top = counts_of(d).most_common(1)[0][1] / len(d)
                out.append(f"| {sc} | {arm} | {seed} | {top:.3f} | {o10[0]} / {o10[1] or '—'} | "
                           f"{o20[0] or '—'} / {o20[1] or '—'} |")
    out += ["", f"### Rolling Gini (250-round windows; mean over {len(SEEDS)} seed(s))", "",
            "| scenario | arm | " + " | ".join(f"r≤{w}" for w in range(250, 4001, 250)) + " |",
            "|---|---|" + "---|" * 16]
    for (sc, arm), series in gini_series.items():
        by_w = defaultdict(list)
        for srs in series:
            for w, gv in srs:
                by_w[w].append(gv)
        out.append(f"| {sc} | {arm} | " + " | ".join(
            fmt(st.fmean(by_w[w]) if by_w.get(w) else None, "{:.2f}")
            for w in range(250, 4001, 250)) + " |")

    out += ["", "### Spearman correlation of dispatch count with … (wen_fresh, per seed)", "",
            "| scenario | seed | F4 rank | mean F3 | mean F1 | mean F2 | V_i | tier |",
            "|---|---|---|---|---|---|---|---|"]
    tables = {}
    for sc in SCENARIOS:
        for seed in SEEDS:
            t, rho = step1(sc, seed, arm="wen_fresh")
            tables[(sc, seed)] = t
            out.append(f"| {sc} | {seed} | " + " | ".join(f"{rho[k]:+.2f}" for k in
                       ("F4 rank", "mean F3", "mean F1", "mean F2", "V_i", "tier")) + " |")

    out += ["", "### F3 over time, wen_fresh vs wen: top-5 vehicles vs the rest "
            "(mean candidate F3 per 100-round window, mean over seeds; every 500 rounds)", "",
            "| scenario | round ≤ | wen_fresh top-5 | wen_fresh rest | wen top-5 | wen rest |",
            "|---|---|---|---|---|---|"]
    f3_plot = {}
    for sc in SCENARIOS:
        ser = {arm: [f3_over_time(sc, s, arm=arm) for s in SEEDS] for arm in ("wen_fresh", "wen")}
        f3_plot[sc] = ser
        for w in range(500, 4001, 500):
            cells = []
            for arm in ("wen_fresh", "wen"):
                for idx in (1, 2):
                    v = [row[idx] for srs in ser[arm] for row in srs
                         if row[0] == w and row[idx] is not None]
                    cells.append(fmt(st.fmean(v) if v else None))
            out.append(f"| {sc} | {w} | " + " | ".join(cells) + " |")

    out += ["", "### Label coverage of the aggregated (accepted, merged) updates", "",
            "Same definitions as Prompt F Step 4.", "",
            "| scenario | arm | seed | " + " | ".join(f"r{c}: KL / coverage / vehicles"
                                                      for c in CHECK_ROUNDS) + " |",
            "|---|---|---|" + "---|" * len(CHECK_ROUNDS)]
    for sc in SCENARIOS:
        for arm in CONC_ARMS:
            for seed in SEEDS:
                cells = [f"{kl:.4f} / {cov:.3f} / {nv}" + ("" if upto == c else f" (r{upto})")
                         for c, upto, kl, cov, nv in step4(sc, arm, seed)]
                out.append(f"| {sc} | {arm} | {seed} | " + " | ".join(cells) + " |")
    return out, {"tables": tables, "gini": gini_series, "f3": f3_plot}


def per_vehicle_table(t) -> list[str]:
    out = ["| client | tier | \\|S\\| | V (m/s) | F4 | F4 rank | dispatches | share | rank "
           "| mean F1 | mean F2 | mean F3 | pick nF1 | nF2 | nF3 | nF4 | random | gpu |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(t, key=lambda r: r["rank"]):
        out.append(f"| {r['client_id']} | {r['tier']} | {r['S']} | {fmt(r['V'], '{:.1f}')} | "
                   f"{fmt(r['F4'], '{:.4f}')} | {r['F4_rank']} | {r['n']} | {r['share']:.3f} | "
                   f"{r['rank']} | {fmt(r['mF1'], '{:.2f}')} | {fmt(r['mF2'], '{:.2f}')} | "
                   f"{fmt(r['mF3'], '{:.3f}')} | {fmt(r['pick_nF1'], '{:.2f}')} | "
                   f"{fmt(r['pick_nF2'], '{:.2f}')} | {fmt(r['pick_nF3'], '{:.2f}')} | "
                   f"{fmt(r['pick_nF4'], '{:.2f}')} | {r['n_random']} | {r['n_gpu']} |")
    return out


def plots(data, extra, curves: Path, gini_png: Path, f3_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S1m.plot(data, curves, arms=("random", "gpu", "wen", "wen_fresh"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, sc in zip(axes, SCENARIOS):
        for arm in CONC_ARMS:
            by_w = defaultdict(list)
            for srs in extra["gini"][(sc, arm)]:
                for w, gv in srs:
                    by_w[w].append(gv)
            ws = sorted(by_w)
            ax.plot(ws, [st.fmean(by_w[w]) for w in ws], marker="o", ms=3, label=arm)
        ax.set_title(f"{sc}: rolling Gini of dispatch counts (250-round windows)")
        ax.set_xlabel("round")
        ax.set_ylabel("Gini")
        ax.legend()
    fig.tight_layout()
    fig.savefig(gini_png, dpi=130)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, sc in zip(axes, SCENARIOS):
        for arm in ("wen_fresh", "wen"):
            for idx, part in ((1, "top-5"), (2, "rest")):
                by_w = defaultdict(list)
                for srs in extra["f3"][sc][arm]:
                    for row in srs:
                        if row[idx] is not None:
                            by_w[row[0]].append(row[idx])
                ws = sorted(by_w)
                ax.plot(ws, [st.fmean(by_w[w]) for w in ws], ls="-" if arm == "wen_fresh" else "--",
                        label=f"{arm} {part}")
        ax.set_title(f"{sc}: candidate F3 over rounds")
        ax.set_xlabel("round")
        ax.set_ylabel("F3")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f3_png, dpi=130)


def render_study(evidence: Path) -> list[str]:
    data = {c: [one(c, s) for s in SEEDS] for c in S1_ROWS + S2_ROWS}
    n_rep = sum(step3(sc, s, arm="wen_fresh")[1] for sc in SCENARIOS for s in SEEDS)
    conc, extra = concentration(data)
    out = [f"## 1. Study table — S1 (seeds {', '.join(map(str, SEEDS))}; mean ± SD when > 1)", "",
           *S1m.study_table(data, S1_ROWS), "",
           "S1 gpu / cpu at seed 4126 count as reaching 90% at their maximum accuracy "
           "(Reading B, `wen_faithful_v1_step6_footnote.md`).", "",
           "## 1. Study table — S2 (Stage 0 off for every arm)", "",
           *S1m.study_table(data, S2_ROWS), "",
           "## 2. Concentration (wen_fresh next to wen, random, gpu)", "",
           f"Replay check: all {n_rep:,} logged wen_fresh decisions are reproduced by "
           "`wen_faithful.decide` on the logged candidates and F values.", "",
           *conc, "",
           "### Per-vehicle table — wen_fresh, S1, seed 4126", "",
           *per_vehicle_table(extra["tables"][("S1", 4126)]), "",
           "### Per-vehicle table — wen_fresh, S2, seed 4126", "",
           *per_vehicle_table(extra["tables"][("S2", 4126)]), "",
           f"## 3. Constraint table (wen_fresh; all decisions per scenario, {len(SEEDS)} seed(s) pooled)", "",
           *S1m.constraint_table(data, cells=("S1_wen_fresh", "S2_wen_fresh")), "",
           "## 4. P22–P25 (v3 rule: same direction in all 3 seeds; no re-wording)", "",
           *p_table(data), "",
           "## 5. Curves", "",
           "`wen_fresh_v1_curves.png` (accuracy vs online clock; random / gpu / wen / wen_fresh; "
           "mean with min–max band), `wen_fresh_v1_gini.png`, `wen_fresh_v1_f3.png`.", ""]
    appendix = ["## Appendix — per-vehicle tables, wen_fresh, all seeds", ""]
    for (sc, seed), t in extra["tables"].items():
        appendix += [f"### {sc}, seed {seed}", "", *per_vehicle_table(t), ""]
    plots(data, extra, evidence / "wen_fresh_v1_curves.png", evidence / "wen_fresh_v1_gini.png",
          evidence / "wen_fresh_v1_f3.png")
    return out, appendix


def render_extra_time() -> list[str]:
    out = ["## 7. Extra wall time spent on fresh evaluation", "",
           "| run | evaluated versions | mean fresh_eval_s | total fresh evaluation (h) | "
           "run wall time (h) |", "|---|---|---|---|---|"]
    for sc in SCENARIOS:
        for seed in SEEDS:
            run = run_dir(f"{sc}_wen_fresh", seed)
            ts = eval_times(run)
            wall = ((run / "dispatch_log.csv").stat().st_mtime
                    - (run / "config_used.toml").stat().st_mtime) / 3600
            out.append(f"| {run.name} | {len(ts)} | {st.fmean(ts):.3f} | {sum(ts) / 3600:.2f} | "
                       f"{wall:.2f} |")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step0", action="store_true")
    parser.add_argument("--step6", action="store_true")
    parser.add_argument("--smoke", nargs="*", type=Path, default=[])
    parser.add_argument("--eval-time", nargs="*", type=Path, default=[])
    parser.add_argument("--study", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", help="restrict to these seeds")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds:
        global SEEDS
        SEEDS = tuple(args.seeds)
        S1m.SEEDS = SEEDS
    lines = []
    if args.step0:
        lines += render_step0(run_dir("S1_wen", 4127), (26, 30, 20, 8)) + [""]
    if args.step6:
        lines += render_step6() + [""]
    if args.eval_time:
        lines += render_eval_time(args.eval_time) + [""]
    if args.smoke:
        lines += render_smoke_fresh(args.smoke) + [""]
    if args.study:
        main_part, appendix = render_study(args.out.parent)
        lines = main_part + lines + render_extra_time() + [""] + appendix
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
