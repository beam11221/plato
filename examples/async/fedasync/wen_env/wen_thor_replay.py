"""
Prompt N Part 1: static replay of Wen's selection on the logged S1 candidates.

Why does `wen_fresh` exclude Thor? Each replay re-decides every logged decision
of the S1 `wen_fresh` runs (seeds 4126 / 4127 / 4128) with the same candidate
sets, the same decision order and the logged F3 / F4, and changes only the time
model (F1), the energy model (F2), the limits T_max / E_max or the objectives.
No training: feedback is ignored (the model trajectory, and so F3, would differ
in a real run).

F1 / F2 are recomputed from the logged dispatch position `x` with the exact
estimators of `wen_fresh` (CpuEstimator) and `wen_fresh_gpu` (GpuEstimator),
each with its own coupling (upload position from its own predicted finish; GPU
energy = P_tier * T_hat_train(GPU) + p * T_up). The selection rule is
`wen_faithful.decide`, unchanged. Limits follow the W5 rule (wen_limits.py)
over the F1 / F2 used in that replay; "inf" switches a constraint off.

Gates (stop on the first failure, Prompt N Step 0 / X3):
    G3   recomputed CPU F1 / F2 equal the logged values of every `wen_fresh`
         row, and GPU F1 / F2 those of every `wen_fresh_gpu` row (1e-9 rel.)
    G4   R0 reproduces every logged pick of the three `wen_fresh` runs
    GX3  recomputed gpu/gpu q90 limits equal the `wen_fresh_gpu` manifest

Usage (from examples/async/fedasync):
    PYTHONPATH=<repo>:wen_env:sumo_close_loop python wen_env/wen_thor_replay.py \
        --out wen_env/evidence/wen_thor_replay_v1.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import wen_constants as K  # noqa: E402
import wen_limits  # noqa: E402
from wen_estimators import CpuEstimator, GpuEstimator  # noqa: E402
from wen_faithful import decide  # noqa: E402
from wen_fleet import Fleet, mix_fleet_dir  # noqa: E402
from wen_reported import ReportedVehicle  # noqa: E402
from wen_trace import load_speeds, veh_id  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs" / "wen_env"
SEEDS = (4126, 4127, 4128)
TIERS = ("xavier", "orin", "thor")
REL_TOL = 1e-9
GPU_MANIFEST = (13.564085609369194, 113.50862433236982)
FOCUS_CLIENT = 26

# id: (time family, energy family, T_max rule, E_max rule, objectives)
REPLAYS = {
    "R0": ("cpu", "cpu", "q90", "q90", 4),
    "R1": ("cpu", "cpu", "q90", "q99", 4),
    "R2": ("cpu", "cpu", "q90", "inf", 4),
    "R3": ("cpu", "cpu", "inf", "inf", 4),
    "R4": ("cpu", "cpu", "inf", "inf", 2),
    "X1": ("cpu", "gpu", "q90", "q90", 4),
    "X2": ("gpu", "cpu", "q90", "q90", 4),
    "X3": ("gpu", "gpu", "q90", "q90", 4),
    "X1inf": ("cpu", "gpu", "inf", "inf", 4),
    "X2inf": ("gpu", "cpu", "inf", "inf", 4),
}
THOR_DIAG = ("R0", "R1", "R2", "R3", "R4", "X2", "X2inf")
XAVIER_DIAG = ("X1", "X1inf")


class GateFailure(RuntimeError):
    pass


def rel_close(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b), 1e-300)


def gini(counts) -> float:
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

class Env:
    """Reported attributes and both estimator families, as the selector sees them."""

    def __init__(self) -> None:
        self.speeds = load_speeds(wen_limits.BUNDLE)
        self.fleet = Fleet.load(mix_fleet_dir(wen_limits.FLEETS, "M0"), wen_limits.TIERS,
                                mix="M0")
        bits = wen_limits.MODEL_BITS
        self.est = {"cpu": CpuEstimator(K.FLOPS_FWD_PER_IMAGE, bits),
                    "gpu": GpuEstimator(K.FLOPS_FWD_PER_IMAGE, bits)}

    def estimates(self, cid: int, t: float, x: float):
        spec = self.fleet.spec(cid)
        rv = ReportedVehicle(cid, t, x, self.speeds[veh_id(cid)], spec.tier_id,
                             spec.tflops_fp16, spec.cpu_ghz, K.partition_size(cid))
        return {fam: e.estimate(rv) for fam, e in self.est.items()}


def load_candidates(run: Path, env: Env, family: str) -> list[dict]:
    """Decisions in log order: {decision, round, logged pick, rows}. Gate G3 inline."""
    decisions: dict[int, dict] = {}
    with (run / "wen_candidates.csv").open(newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh), start=2):
            cid, t, x = int(r["client_id"]), float(r["t"]), float(r["x"])
            est = env.estimates(cid, t, x)
            own = est[family]
            for name, mine, logged in (("F1", own.total, float(r["F1"])),
                                       ("F2", own.E, float(r["F2"]))):
                if not rel_close(mine, logged):
                    raise GateFailure(f"G3 {run.name} line {i}: recomputed {name} "
                                      f"{mine!r} != logged {logged!r} (client {cid}, t {t})")
            d = decisions.setdefault(int(r["decision"]), {
                "decision": int(r["decision"]), "round": int(r["round"]), "t": t,
                "logged_pick": None, "rows": []})
            if int(r["picked"]):
                d["logged_pick"] = cid
            d["rows"].append({
                "cid": cid, "tier": r["tier"],
                "F1": {f: e.total for f, e in est.items()},
                "F2": {f: e.E for f, e in est.items()},
                "F3": float(r["F3"]), "F4": float(r["F4"]),
                "logged_feasible": int(r["feasible"])})
    return [decisions[k] for k in sorted(decisions)]


def inventory(run: Path) -> dict:
    with (run / "wen_candidates.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames
        per_dec, feas = Counter(), Counter()
        raw_infeasible = 0
        for r in reader:
            per_dec[r["decision"]] += 1
            feas[r["decision"]] += int(r["feasible"])
            if not int(r["feasible"]) and all(r[f"F{i}"] not in ("", "nan")
                                              for i in range(1, 5)):
                raw_infeasible += 1
    n = len(per_dec)
    return {"run": run.name, "decisions": n,
            "rows_per_decision": sum(per_dec.values()) / n,
            "feasible_per_decision": sum(feas.values()) / n,
            "infeasible_rows_with_raw_F": raw_infeasible,
            "infeasible_rows": sum(per_dec.values()) - sum(feas.values()),
            "columns": cols}


def random_tier_shares(seed: int) -> dict[str, float]:
    with (RUNS / f"S1_random_s{seed}" / "dispatch_log.csv").open(newline="") as fh:
        tiers = Counter(r["tier"] for r in csv.DictReader(fh))
    n = sum(tiers.values())
    return {t: tiers[t] / n for t in TIERS}


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

def limit_table() -> tuple[dict, dict]:
    """{(time fam, energy fam): {'T': {q: v}, 'E': {q: v}}} and the populations."""
    pops = {fam: wen_limits.population(fam, "M0") for fam in ("cpu", "gpu")}
    if not np.array_equal(pops["cpu"]["x"], pops["gpu"]["x"]):
        raise GateFailure("limit populations are not aligned between families")
    out = {}
    for tf in ("cpu", "gpu"):
        for ef in ("cpu", "gpu"):
            out[(tf, ef)] = {
                "T": {q: float(np.percentile(pops[tf]["F1"], q)) for q in (90, 99)},
                "E": {q: float(np.percentile(pops[ef]["F2"], q)) for q in (90, 99)}}
    return out, pops


def limit_of(limits, tf, ef, kind, rule) -> float:
    return math.inf if rule == "inf" else limits[(tf, ef)][kind][int(rule[1:])]


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay(decisions, tf, ef, t_max, e_max, n_obj):
    out = []
    for d in decisions:
        obj = {}
        for r in d["rows"]:
            full = (r["F1"][tf], r["F2"][ef], r["F3"], r["F4"])
            obj[r["cid"]] = full[:n_obj]
        # decide() tests f[0] <= T_max and f[1] <= E_max, so F1, F2 stay first.
        out.append((d, decide(obj, t_max, e_max)))
    return out


def tier_rates(results) -> dict:
    rows, feas, par, pick = Counter(), Counter(), Counter(), Counter()
    for d, dec in results:
        fs, ps = set(dec.feasible), set(dec.pareto)
        for r in d["rows"]:
            t = r["tier"]
            rows[t] += 1
            feas[t] += r["cid"] in fs
            par[t] += r["cid"] in ps
            pick[t] += r["cid"] == dec.pick
    n_pick = sum(pick.values())
    return {t: {"feasible": feas[t] / rows[t], "pareto": par[t] / rows[t],
                "picked": pick[t] / rows[t], "share": pick[t] / n_pick} for t in TIERS}


def pick_stats(results) -> dict:
    counts = Counter(dec.pick for _, dec in results)
    return {"gini": gini([counts.get(c, 0) for c in range(1, K.N_CLIENTS + 1)]),
            "distinct": len(counts),
            "fallback": sum(dec.fallback for _, dec in results),
            "decisions": len(results)}


def focus_diag(results, focus: str, tf: str, n_obj: int) -> dict:
    """Decisions with >= 1 `focus`-tier vehicle in the Pareto set."""
    n, picked, ineq = 0, 0, 0
    best_f, best_o = [], []
    gap, spread, rng = [], [], []
    for d, dec in results:
        tier = {r["cid"]: r["tier"] for r in d["rows"]}
        f1 = {r["cid"]: r["F1"][tf] for r in d["rows"]}
        foc = [c for c in dec.pareto if tier[c] == focus]
        oth = [c for c in dec.pareto if tier[c] != focus]
        if not foc:
            continue
        n += 1
        picked += tier[dec.pick] == focus
        if not oth:
            continue
        bf = min(foc, key=lambda c: (dec.score[c], c))
        bo = min(oth, key=lambda c: (dec.score[c], c))
        nf, no = dec.normalised[bf], dec.normalised[bo]
        best_f.append(nf)
        best_o.append(no)
        ineq += (nf[1] - no[1]) > (no[0] - nf[0])
        par_f1 = [f1[c] for c in dec.pareto]
        rng.append(max(par_f1) - min(par_f1))
        gap.append(min(f1[c] for c in oth) - min(f1[c] for c in foc))
        spread.append(max(f1[c] for c in oth) - min(f1[c] for c in oth))
    m = len(best_f)
    mean = (lambda v: [float(np.mean([x[i] for x in v])) for i in range(n_obj)]) if m else None
    return {"decisions": n, "share_picked": picked / n if n else math.nan,
            "with_both": m,
            "best_focus_nF": mean(best_f) if m else None,
            "best_other_nF": mean(best_o) if m else None,
            "ineq_share": ineq / m if m else math.nan,
            "F1_range": float(np.mean(rng)) if m else math.nan,
            "tier_gap": float(np.mean(gap)) if m else math.nan,
            "within_spread": float(np.mean(spread)) if m else math.nan}


def client_diag(results, cid: int, n_obj: int) -> dict:
    rows = feas = par = pick = 0
    norm = []
    for d, dec in results:
        if not any(r["cid"] == cid for r in d["rows"]):
            continue
        rows += 1
        feas += cid in dec.feasible
        pick += dec.pick == cid
        if cid in dec.normalised:
            par += 1
            norm.append(dec.normalised[cid])
    return {"rows": rows, "feasible": feas, "pareto": par, "picked": pick,
            "nF": [float(np.mean([v[i] for v in norm])) for i in range(n_obj)] if norm
            else None}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{nd}f}"


def nf_cells(v, n_obj=4) -> list[str]:
    if v is None:
        return ["—"] * 4
    return [fmt(x, 2) for x in v] + ["—"] * (4 - n_obj)


def box_plot(pops, limits, cfg_e, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, fam in zip(axes, ("cpu", "gpu")):
        data = [pops[fam]["F2"][pops[fam]["tier"] == t] for t in TIERS]
        ax.boxplot(data, whis=(1, 99), showfliers=False)
        ax.set_xticks(range(1, 4), [t.capitalize() for t in TIERS])
        e = limits[(fam, fam)]["E"]
        ax.axhline(e[90], color="tab:red", ls="--", lw=1, label=f"q90 = {e[90]:.2f} J")
        ax.axhline(e[99], color="tab:orange", ls=":", lw=1.2, label=f"q99 = {e[99]:.2f} J")
        ax.axhline(cfg_e[fam], color="black", lw=0.6, alpha=0.6,
                   label=f"logged E_max = {cfg_e[fam]:.2f} J")
        ax.set_title(f"F2 by tier, {fam.upper()} family (whiskers 1–99 %)")
        ax.set_ylabel("F2 (J)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=HERE, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def verdict(dirs: list[bool]) -> str:
    if all(dirs):
        return "supported (3 / 3 seeds)"
    if not any(dirs):
        return "not supported (0 / 3 seeds)"
    return f"mixed ({sum(dirs)} / 3 seeds) — not supported under v3"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lock", default="6feeb8e7", help="v3.8 lock commit")
    ap.add_argument("--tag", default="", help="suffix of the wen_fresh run dirs")
    args = ap.parse_args()
    start = time.time()
    fig_dir = args.out.with_suffix("")
    fig_dir.mkdir(parents=True, exist_ok=True)
    L = ["# Why Wen et al. excludes Thor — static replay on logged candidates "
         "(Prompt N Part 1)", "",
         f"v3.8 lock commit `{args.lock}`; code at `{git_head()}`; "
         f"script `wen_env/wen_thor_replay.py`.", "",
         "Static replay: same candidate sets, decision order and logged F3 / F4 as the "
         "S1 `wen_fresh` runs; only the replay's F1 / F2 model, limits or objectives "
         "change. It ignores feedback: in a real run the model trajectory, and so F3, "
         "would differ. Candidate sets are the logged ones also within a round, so a "
         "slot-2 / slot-3 set excludes the logged (not the replayed) earlier pick.", ""]

    # -- Step 0 inventory --------------------------------------------------------------
    env = Env()
    L += ["## Step 0 — inventory of `wen_candidates.csv`", "",
          "| run | decisions | rows / decision | n_feasible / decision | infeasible rows | "
          "of which with raw F1–F4 |", "|---|---|---|---|---|---|"]
    cols = None
    for arm in ("wen_fresh", "wen_fresh_gpu"):
        for s in SEEDS:
            inv = inventory(RUNS / f"S1_{arm}_s{s}{args.tag if arm == 'wen_fresh' else ''}")
            cols = cols or inv["columns"]
            L.append(f"| {inv['run']} | {inv['decisions']:,} | {inv['rows_per_decision']:.2f} "
                     f"| {inv['feasible_per_decision']:.2f} | {inv['infeasible_rows']:,} | "
                     f"{inv['infeasible_rows_with_raw_F']:,} |")
    have = {c: (c in cols) for c in ("x", "t", "tier", "feasible", "pareto", "picked",
                                      "x_dispatch", "V_k", "feasible_T", "feasible_E")}
    L += ["", f"Columns: `{', '.join(cols)}`.", "",
          "Requested columns: " + "; ".join(f"`{c}` {'yes' if v else 'no'}"
                                             for c, v in have.items()) + ".",
          "Candidate position is logged as `x` (the reported position at decision time "
          "`t`), so no SUMO reconstruction is needed. `V_k` is not logged; it is the "
          "constant reported speed from `fleet_speeds.csv`. `feasible_T` / `feasible_E` "
          "are not logged in these runs; they are recomputed from F1, F2 and the limits.", ""]

    # -- gates ---------------------------------------------------------------------------
    gates = []
    try:
        for s in SEEDS:
            load_candidates(RUNS / f"S1_wen_fresh_gpu_s{s}", env, "gpu")
        runs = {s: load_candidates(RUNS / f"S1_wen_fresh_s{s}{args.tag}", env, "cpu")
                for s in SEEDS}
        gates.append("G3 PASS: recomputed CPU F1 / F2 equal every logged `wen_fresh` value "
                     "and GPU F1 / F2 every logged `wen_fresh_gpu` value (1e-9 rel.), "
                     f"{sum(len(d['rows']) for v in runs.values() for d in v):,} + GPU rows.")
        limits, pops = limit_table()
        cfg = {"cpu": (34.334345905647005, 36.423846640394224), "gpu": GPU_MANIFEST}
        for fam in ("cpu", "gpu"):
            lt = limits[(fam, fam)]
            if not (rel_close(lt["T"][90], cfg[fam][0]) and rel_close(lt["E"][90], cfg[fam][1])):
                raise GateFailure(f"{'GX3' if fam == 'gpu' else 'limits'} {fam}: recomputed "
                                  f"q90 {lt['T'][90]!r} / {lt['E'][90]!r} != manifest "
                                  f"{cfg[fam]}")
        gates.append("GX3 PASS: recomputed GPU/GPU q90 limits equal the `wen_fresh_gpu` "
                     f"manifest ({limits[('gpu', 'gpu')]['T'][90]!r} s, "
                     f"{limits[('gpu', 'gpu')]['E'][90]!r} J); CPU/CPU q90 equal the "
                     "`wen_fresh` config.")
        results = {}
        for rid, (tf, ef, tr, er, n_obj) in REPLAYS.items():
            t_max = limit_of(limits, tf, ef, "T", tr)
            e_max = limit_of(limits, tf, ef, "E", er)
            for s in SEEDS:
                results[(rid, s)] = replay(runs[s], tf, ef, t_max, e_max, n_obj)
        for s in SEEDS:
            for d, dec in results[("R0", s)]:
                if dec.pick != d["logged_pick"]:
                    raise GateFailure(f"G4 seed {s} decision {d['decision']} (round "
                                      f"{d['round']}): replayed {dec.pick} != logged "
                                      f"{d['logged_pick']}")
                logged_f = sorted(r["cid"] for r in d["rows"] if r["logged_feasible"])
                if logged_f != dec.feasible:
                    raise GateFailure(f"G4 seed {s} decision {d['decision']}: feasible set "
                                      "differs from the log")
        gates.append("G4 PASS: R0 reproduces every logged pick and feasible set of the "
                     f"three `wen_fresh` runs ({sum(len(runs[s]) for s in SEEDS):,} decisions).")
    except GateFailure as exc:
        gates.append(f"**FAIL — {exc}**")
        L += ["## Gates", "", *[f"- {g}" for g in gates], "", "Stopped at the first failure."]
        args.out.write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return 1
    L += ["## Gates", "", *[f"- {g}" for g in gates], ""]

    # -- limits ---------------------------------------------------------------------------
    L += ["## Limits per replay (W5 rule over 30 vehicles × trace samples, "
          f"{len(pops['cpu']['F1']):,} pairs)", "",
          "| replay | F1 model | F2 model | T_max rule | T_max (s) | E_max rule | E_max (J) | "
          "objectives |", "|---|---|---|---|---|---|---|---|"]
    for rid, (tf, ef, tr, er, n_obj) in REPLAYS.items():
        L.append(f"| {rid} | {tf.upper()} | {ef.upper()} | {tr} | "
                 f"{limit_of(limits, tf, ef, 'T', tr)!r} | {er} | "
                 f"{limit_of(limits, tf, ef, 'E', er)!r} | "
                 f"{'F1–F4' if n_obj == 4 else 'F1 + F2'} |")
    L.append("")

    # -- 1 per-tier rates, 2 Gini ----------------------------------------------------------
    rnd = {s: random_tier_shares(s) for s in SEEDS}
    L += ["## 1. Per-tier rates and 2. concentration", "",
          "Rates are shares of that tier's candidate rows (feasible / in the Pareto set / "
          "picked); \"share\" is the tier's share of all replayed picks. Random = the "
          "realised whole-run tier share of `S1_random_s<seed>`.", "",
          "| replay | seed | X feas | O feas | T feas | X Pareto | O Pareto | T Pareto | "
          "X picked | O picked | T picked | X share | O share | **T share** | Gini | "
          "distinct | fallback |", "|" + "---|" * 17]
    stats = {}
    for rid in REPLAYS:
        for s in SEEDS:
            tr_ = tier_rates(results[(rid, s)])
            ps = pick_stats(results[(rid, s)])
            stats[(rid, s)] = (tr_, ps)
            L.append(f"| {rid} | {s} | " + " | ".join(
                fmt(tr_[t]["feasible"], 2) for t in TIERS) + " | " + " | ".join(
                fmt(tr_[t]["pareto"], 2) for t in TIERS) + " | " + " | ".join(
                fmt(tr_[t]["picked"], 3) for t in TIERS) + " | " + " | ".join(
                fmt(tr_[t]["share"], 3) for t in TIERS[:2]) +
                f" | **{fmt(tr_['thor']['share'], 3)}** | {ps['gini']:.3f} | "
                f"{ps['distinct']} | {ps['fallback']} |")
    for s in SEEDS:
        L.append(f"| Random (realised) | {s} | | | | | | | | | | "
                 f"{rnd[s]['xavier']:.3f} | {rnd[s]['orin']:.3f} | "
                 f"**{rnd[s]['thor']:.3f}** | | | |")
    L.append("")

    # -- 3 / 4 focus-tier diagnostics -------------------------------------------------------
    def diag_table(focus, ids):
        name = focus.capitalize()
        out = [f"Over decisions with ≥ 1 {name} in the Pareto set. \"best\" = lowest Wen "
               f"score within the group. Inequality (P51 form): nF2(best {name}) − "
               f"nF2(best other) > nF1(best other) − nF1(best {name}). F1 range of the "
               f"Pareto set vs tier gap = min F1(other) − min F1({name}) and "
               "within-other spread = max − min F1 among the others (means, seconds).", "",
               f"| replay | seed | decisions | {name} picked | best {name} nF1–nF4 | "
               f"best other nF1–nF4 | inequality share | Pareto F1 range | tier gap | "
               "within-other spread |", "|---|---|---|---|---|---|---|---|---|---|"]
        diags = {}
        for rid in ids:
            tf, _, _, _, n_obj = REPLAYS[rid]
            for s in SEEDS:
                g = focus_diag(results[(rid, s)], focus, tf, n_obj)
                diags[(rid, s)] = g
                out.append(
                    f"| {rid} | {s} | {g['decisions']:,} | {fmt(g['share_picked'])} | "
                    f"{' / '.join(nf_cells(g['best_focus_nF'], n_obj))} | "
                    f"{' / '.join(nf_cells(g['best_other_nF'], n_obj))} | "
                    f"{fmt(g['ineq_share'])} | {fmt(g['F1_range'], 2)} | "
                    f"{fmt(g['tier_gap'], 2)} | {fmt(g['within_spread'], 2)} |")
        return out, diags

    t_lines, thor = diag_table("thor", THOR_DIAG)
    x_lines, xav = diag_table("xavier", XAVIER_DIAG)
    L += ["## 3. Thor diagnostics", "", *t_lines, "", "## 4. Xavier diagnostics", "",
          *x_lines, ""]

    # -- 5 client 26 --------------------------------------------------------------------------
    L += [f"## 5. Client {FOCUS_CLIENT} (Thor)", ""]
    for s in SEEDS:
        f4 = {r["cid"]: r["F4"] for d in runs[s] for r in d["rows"]}
        rank = 1 + sum(v < f4[FOCUS_CLIENT] for v in f4.values())
        L.append(f"- seed {s}: F4 = {f4[FOCUS_CLIENT]:.4f}, F4 rank {rank} of 30 "
                 "(1 = lowest KL).")
    L += ["", "| replay | seed | candidate rows | feasible | in Pareto | picked | "
          "mean nF1–nF4 when in Pareto |", "|---|---|---|---|---|---|---|"]
    for rid, (_, _, _, _, n_obj) in REPLAYS.items():
        for s in SEEDS:
            c = client_diag(results[(rid, s)], FOCUS_CLIENT, n_obj)
            L.append(f"| {rid} | {s} | {c['rows']:,} | {c['feasible']:,} | {c['pareto']:,} "
                     f"| {c['picked']:,} | {' / '.join(nf_cells(c['nF'], n_obj))} |")
    L.append("")

    # -- 6 box plot ---------------------------------------------------------------------------
    box = fig_dir / "f2_by_tier.png"
    box_plot(pops, limits, {"cpu": 36.423846640394224, "gpu": GPU_MANIFEST[1]}, box)
    L += ["## 6. F2 by tier", "",
          f"![F2 by tier]({fig_dir.name}/{box.name})", "",
          "| family | tier | F2 min | F2 median | F2 max | share > q90 | share > q99 |",
          "|---|---|---|---|---|---|---|"]
    for fam in ("cpu", "gpu"):
        e = limits[(fam, fam)]["E"]
        for t in TIERS:
            v = pops[fam]["F2"][pops[fam]["tier"] == t]
            L.append(f"| {fam.upper()} | {t.capitalize()} | {v.min():.3f} | "
                     f"{np.median(v):.3f} | {v.max():.3f} | {np.mean(v > e[90]):.1%} | "
                     f"{np.mean(v > e[99]):.1%} |")
    L.append("")

    # -- P49-P53 ----------------------------------------------------------------------------
    def share(rid, s, tier):
        return stats[(rid, s)][0][tier]["share"]

    preds = [
        ("P49", "R1: Thor share of picks < 0.05",
         [(share("R1", s, "thor"), share("R1", s, "thor") < 0.05) for s in SEEDS]),
        ("P50", "R2: Thor share < Random's realised Thor share",
         [(share("R2", s, "thor"), share("R2", s, "thor") < rnd[s]["thor"]) for s in SEEDS]),
        ("P51", "R2: inequality holds in > 1/2 of decisions with Thor in Pareto",
         [(thor[("R2", s)]["ineq_share"], thor[("R2", s)]["ineq_share"] > 0.5)
          for s in SEEDS]),
        ("P52", "X1: Xavier share of picks < 0.05",
         [(share("X1", s, "xavier"), share("X1", s, "xavier") < 0.05) for s in SEEDS]),
        ("P53", "X2: Thor share of picks < 0.05",
         [(share("X2", s, "thor"), share("X2", s, "thor") < 0.05) for s in SEEDS]),
    ]
    L += ["## P49–P53 (v3 rule: same direction in all three seeds)", "",
          "| id | prediction | " + " | ".join(f"seed {s}" for s in SEEDS) + " | v3 verdict |",
          "|---|---|---|---|---|---|"]
    for pid, text, vals in preds:
        L.append(f"| {pid} | {text} | " + " | ".join(
            f"{v:.3f} ({'yes' if ok else 'no'})" for v, ok in vals) +
            f" | {verdict([ok for _, ok in vals])} |")
    L.append("")

    # -- conditional run --------------------------------------------------------------------
    r1 = {s: share("R1", s, "thor") for s in SEEDS}
    run_q99 = any(v >= 0.05 for v in r1.values())
    L += ["## Decision rule for Part 2 `wen_fresh_q99`", "",
          "R1 Thor share: " + ", ".join(f"seed {s} {v:.3f}" for s, v in r1.items()) + ". "
          + ("At least one seed ≥ 0.05 → **run `wen_fresh_q99`** in Part 2." if run_q99
             else "No seed ≥ 0.05 → **skip `wen_fresh_q99`**."), ""]

    # -- layers ------------------------------------------------------------------------------
    def consistent(cond: bool) -> str:
        return "consistent" if cond else "not consistent"

    L += ["## Layer check (mechanical)", "",
          f"- Layer 2 (our E_max calibration) as the sole cause: Thor share rises to "
          f"≥ Random's under R2 in every seed → "
          f"{consistent(all(share('R2', s, 'thor') >= rnd[s]['thor'] for s in SEEDS))}.",
          f"- Layer 1 + 3 (Wen's CPU energy model and normalisation) excluding Thor without "
          f"constraints: R3 Thor share < Random's in every seed → "
          f"{consistent(all(share('R3', s, 'thor') < rnd[s]['thor'] for s in SEEDS))}.",
          f"- Layer 3 alone (R4, F1 + F2, no limits): Thor share < Random's in every seed → "
          f"{consistent(all(share('R4', s, 'thor') < rnd[s]['thor'] for s in SEEDS))}.",
          f"- Energy model removes the tier it prices highest (P52 and P53 both hold in "
          f"every seed) → {consistent(all(ok for p in preds[3:] for _, ok in p[2]))}.", "",
          f"Wall time {time.time() - start:.0f} s.", ""]
    args.out.write_text("\n".join(L) + "\n")
    (fig_dir / "replay_summary.json").write_text(json.dumps({
        f"{rid}_s{s}": {"tier": stats[(rid, s)][0], **stats[(rid, s)][1]}
        for rid in REPLAYS for s in SEEDS}, indent=1) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
