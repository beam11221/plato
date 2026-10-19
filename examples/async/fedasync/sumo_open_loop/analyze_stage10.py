"""
Stage 10, Step 4: the full method (Stage 0 filter + scoring) in the open population.

Per run: the Stage 9 set (via analyze_stage9.per_run, unchanged) plus the
Stage 5-6 filter set, from the [Stage0] cand / round / selected lines:
  dispatches attempted, candidates rejected, fallback rate
  pool sizes after each gate (median / p10 / p90) per window
  rejection accuracy per tier: fraction of rejected candidates that would in
      fact have completed in range (t + occupant sleep <= its vehicle's
      window exit; T_comm = 0; the ~0.1 s host processing time is ignored)
  of admitted dispatches, fraction wasted, per tier
  candidate pass rate per tier (the eligibility fraction as the gate saw it)
  waste / mix decomposition against baseline/none of the same seed and regime
      (conservation law, empirical tier sleeps 14.37 / 1.78 / 1.07 s)
  observed: the fleet running-mean T_hat over time; share of its dispatches
      whose T_hat came from the never-seen rule, per tier
  G_i xavier : thor (rounds 2,001-4,000, seen) next to the no-filter partner

Comparisons A-I (STAGE10_filter_open.md), paired by seed, sign checked per
seed; a sign that flips in any available seed is "undecided".

Prefixes (seed ledger):
  stage10f1_s<seed>    filter, alpha = 1     baseline gpu cpu observed
  stage10f10_s<seed>   filter, alpha = 10    gpu cpu observed          (Wave B)
  stage10nf_s<seed>    no filter, random     baseline
  stage9_s<seed>       no filter, alpha = 1  gpu cpu observed          (partners)
  7b / 8               no filter, alpha = 10 (and the 7b/8 baseline cells, for H)

Usage: python3 analyze_stage10.py [--wave A|B] [--out ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_stage7 as a7  # noqa: E402
import analyze_stage9 as a9  # noqa: E402
from occupant_identity import load_fleet  # noqa: E402
from stage7_population import vehicle_sleep_s  # noqa: E402

FLEET = load_fleet(HERE.parent / "sumo_close_loop" / "input" / "clients_n20r400.csv",
                   HERE.parent / "sumo_close_loop" / "output" / "client_sleep_times_n20r400.csv")


def occupant_sleep(regime: str, slot: int, veh: str, tier: str) -> float:
    """The identity's true sleep, independent of the log (physics seed 4126)."""
    return FLEET[slot][1] if regime == "closed" else vehicle_sleep_s(4126, veh, tier)

SEEDS = (4126, 4127, 4128)
REGIMES = ("open", "closed")
TIERS = ("xavier", "orin", "thor")
SLEEP = {"xavier": 14.37, "orin": 1.78, "thor": 1.07}   # empirical tier sleeps (spec)
C = 4
LOGS = HERE / "logs"
EVIDENCE = HERE.parent / "gpu_tiers_dataset" / "stage10_evidence"
TRACE = HERE / "sumo_trace" / "e1_open_poisson_seed4126_range400"
WINDOWS = ((1, 1000), (1001, 2000), (2001, 4000))

CAND = re.compile(r"\[Stage0\] cand round=(-?\d+) t=(\S+) slot=(\d+) veh_id=(\S+) identity_id=(\S+) "
                  r"tier=(\S+) t_hat=(\S+) t_hat_basis=(\S+) r_i=(\S+) pass=([01])")
ROUND = re.compile(r"\[Stage0\] round=(-?\d+) .* eligible_after_range_gate=(\d+) "
                   r"eligible_after_stage0=(\d+) .*stage0_fallback=([01])")
SEL = re.compile(r"\[Stage0\] round=(-?\d+) selected_id=\d+ selected_tier=(\S+) t_hat_used=(\S+) "
                 r"R_i_selected=\S+ t_hat_basis=(\S+)")

EXIT = {}
with (TRACE / "coverage_windows.csv").open() as fh:
    for r in csv.DictReader(fh):
        EXIT[r["veh_id"]] = float(r["t_exit_s"])


def pfx_of(kind: str, seed: int) -> str:
    return {"f1": f"stage10f1_s{seed}", "f10": f"stage10f10_s{seed}", "nf": f"stage10nf_s{seed}",
            "s9": f"stage9_s{seed}", "a10": "stage7b" if seed == 4126 else f"stage8_s{seed}"}[kind]


def available(pfx, cell):
    return a9.available(pfx, cell)


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def filter_stats(pfx, cell):
    log = LOGS / f"{pfx}_{cell}.log"
    regime = cell.split("_")[1]
    pools = defaultdict(lambda: {"range": [], "after": []})
    rounds = fb = 0
    cand_pass = defaultdict(lambda: [0, 0])
    rej_ok = defaultdict(lambda: [0, 0])     # tier -> [would complete, rejected]
    basis_sel = defaultdict(Counter)          # tier -> basis counts among dispatches
    fleet_t_hat = defaultdict(list)
    for line in log.open(errors="ignore"):
        m = CAND.search(line)
        if m:
            rnd, t, slot, veh, ident, tier, t_hat, basis, r_i, ok = m.groups()
            rnd, t = int(rnd), float(t)
            cand_pass[tier][0] += ok == "1"
            cand_pass[tier][1] += 1
            if basis == "fleet":
                w = next((f"{lo}-{hi}" for lo, hi in WINDOWS if lo <= rnd <= hi), None)
                if w:
                    fleet_t_hat[w].append(float(t_hat))
            if ok == "0":
                sl = occupant_sleep(regime, int(slot), veh, tier)
                rej_ok[tier][0] += (t + sl) <= EXIT[veh]
                rej_ok[tier][1] += 1
            continue
        m = ROUND.search(line)
        if m:
            rnd = int(m.group(1))
            w = next((f"{lo}-{hi}" for lo, hi in WINDOWS if lo <= rnd <= hi), "4000+")
            pools[w]["range"].append(int(m.group(2)))
            pools[w]["after"].append(int(m.group(3)))
            rounds += 1
            fb += m.group(4) == "1"
            continue
        m = SEL.search(line)
        if m:
            basis_sel[m.group(2)][m.group(4)] += 1
    allr = [x for w in pools.values() for x in w["range"]]
    alla = [x for w in pools.values() for x in w["after"]]
    return {
        "gated_rounds": rounds, "fallback_rate": fb / rounds if rounds else None,
        "candidates": len(allr) and sum(allr), "rejected": sum(allr) - sum(alla),
        "reject_rate": 1 - sum(alla) / sum(allr) if sum(allr) else None,
        "pools": {w: {g: {"median": st.median(v[g]) if v[g] else None, "p10": q(v[g], .1), "p90": q(v[g], .9)}
                      for g in ("range", "after")} for w, v in pools.items()},
        "pool_after_median": st.median(alla) if alla else None,
        "candidate_pass_rate": {t: v[0] / v[1] for t, v in cand_pass.items() if v[1]},
        "rejection_would_complete": {t: {"frac": v[0] / v[1], "n": v[1]} for t, v in rej_ok.items() if v[1]},
        "dispatch_t_hat_basis": {t: dict(c) for t, c in basis_sel.items()},
        "fleet_running_mean_t_hat": {w: st.mean(v) for w, v in fleet_t_hat.items()},
    }


def admitted_waste_per_tier(run):
    rep = [d for d in run.dispatches if d.raw_in_range is not None]
    out = {}
    for t in TIERS:
        ds = [d for d in rep if d.tier == t]
        out[t] = sum(not d.raw_in_range for d in ds) / len(ds) if ds else None
    return out


def conservation_split(run, ref):
    """ln(T/T_ref) = mix (fixed tier sleeps) - ln((1-w)/(1-w_ref)) + residual."""
    def terms(r):
        rep = [d for d in r.dispatches if d.raw_in_range is not None]
        n = len(rep)
        share = {t: sum(d.tier == t for d in rep) / n for t in TIERS}
        ms = sum(share[t] * SLEEP[t] for t in TIERS)
        w = sum(not d.raw_in_range for d in rep) / n
        return ms, w, a7.clock_at(r, 4000)
    ms, w, T = terms(run)
    ms0, w0, T0 = terms(ref)
    total = math.log(T / T0)
    mix = math.log(ms / ms0)
    waste = -(math.log(1 - w) - math.log(1 - w0))
    return {"total_pct": 100 * total, "mix_pct": 100 * mix, "waste_pct": 100 * waste,
            "residual_pct": 100 * (total - mix - waste), "measured_pct": 100 * (T / T0 - 1)}


def load(pfx, cell):
    a7.PREFIX = pfx
    return a7.parse(cell)


def pct(a, b):
    return 100.0 * (a - b) / b


def verdict(vals, want_negative=True):
    if not vals:
        return "no data"
    s = [(v < 0) if want_negative else (v > 0) for v in vals]
    return "held" if all(s) else ("contradicted" if not any(s) else "undecided")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", choices=["A", "B"], default="B")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_path = args.out or EVIDENCE / f"stage10_analysis_wave{args.wave}.json"

    kinds = ["f1", "nf", "s9", "a10"] + (["f10"] if args.wave == "B" else [])
    arms_of = {"f1": ("baseline", "gpu", "cpu", "observed"), "nf": ("baseline",),
               "s9": ("gpu", "cpu", "observed"), "a10": ("baseline", "gpu", "cpu", "observed"),
               "f10": ("gpu", "cpu", "observed")}
    runs, parsed = {}, {}
    for kind in kinds:
        for seed in SEEDS:
            for arm in arms_of[kind]:
                for reg in REGIMES:
                    pfx, cell = pfx_of(kind, seed), f"{arm}_{reg}"
                    if not available(pfx, cell):
                        continue
                    key = f"{kind}/{seed}/{cell}"
                    res = a9.per_run(pfx, cell, arm)
                    res.pop("_recycle_seq", None)
                    run = load(pfx, cell)
                    parsed[key] = run
                    if kind in ("f1", "f10"):
                        res["filter"] = filter_stats(pfx, cell)
                        res["admitted_waste_per_tier"] = admitted_waste_per_tier(run)
                    runs[key] = res

    T = {k: a7.clock_at(r, 4000) for k, r in parsed.items()}

    # occupancy: common prefix over every parsed run
    n = min(len(r.recycle_seq) for r in parsed.values())
    digests = {repr(r.recycle_seq[:n]) for r in parsed.values()}
    occupancy = {"runs": len(parsed), "common_prefix": n, "identical": len(digests) == 1}

    # decomposition of every filtered run vs baseline/none (Stage 10 re-run)
    for k, res in runs.items():
        kind, seed, cell = k.split("/")
        ref = f"nf/{seed}/baseline_{cell.split('_')[1]}"
        if kind in ("f1", "f10") and ref in parsed:
            res["vs_baseline_none"] = conservation_split(parsed[k], parsed[ref])

    def g(a, b):
        return pct(T[a], T[b]) if a in T and b in T else None

    def per_seed(fa, fb):
        return {s: v for s in SEEDS if (v := g(fa(s), fb(s))) is not None}

    comps = {}
    for reg, key in (("open", "A"), ("closed", "B")):
        v = per_seed(lambda s: f"f1/{s}/gpu_{reg}", lambda s: f"f1/{s}/baseline_{reg}")
        comps[key] = {"what": f"gpu+F vs baseline+F, alpha=1, {reg}", "per_seed": v,
                      "verdict_gpu_faster": verdict(list(v.values()), True)}
    comps["C"] = {}
    for reg in REGIMES:
        v = per_seed(lambda s, r=reg: f"f1/{s}/gpu_{r}", lambda s, r=reg: f"f1/{s}/observed_{r}")
        comps["C"][reg] = {"per_seed": v, "verdict_gpu_faster": verdict(list(v.values()), True),
                           "waste_gpu_minus_observed_pts": {
                               s: 100 * (runs[f"f1/{s}/gpu_{reg}"]["waste"] - runs[f"f1/{s}/observed_{reg}"]["waste"])
                               for s in v}}
    comps["D"] = {}
    for arm in ("baseline", "gpu", "observed", "cpu"):
        for reg in REGIMES:
            v = per_seed(lambda s, a=arm, r=reg: f"f1/{s}/{a}_{r}", lambda s, r=reg: f"nf/{s}/baseline_{r}")
            comps["D"][f"{arm}+F_{reg}"] = {
                "per_seed": v, "verdict_faster": verdict(list(v.values()), True),
                "split": {s: runs[f"f1/{s}/{arm}_{reg}"].get("vs_baseline_none") for s in v}}
    comps["E"] = {reg: comps["D"][f"baseline+F_{reg}"] for reg in REGIMES}
    comps["F"] = {}
    for arm in ("gpu", "observed", "cpu"):
        for reg in REGIMES:
            rows = {}
            for s in SEEDS:
                base = f"nf/{s}/baseline_{reg}"
                need = (f"f1/{s}/baseline_{reg}", f"s9/{s}/{arm}_{reg}", f"f1/{s}/{arm}_{reg}", base)
                if all(k in T for k in need):
                    gf = T[need[0]] / T[base] - 1
                    gs = T[need[1]] / T[base] - 1
                    gc = T[need[2]] / T[base] - 1
                    rows[s] = {"filter_only_pct": 100 * gf, "scoring_only_pct": 100 * gs,
                               "additive_pred_pct": 100 * (gf + gs),
                               "multiplicative_pred_pct": 100 * ((1 + gf) * (1 + gs) - 1),
                               "combined_pct": 100 * gc,
                               "vs_filter_off_pct": pct(T[need[2]], T[need[1]])}
            comps["F"][f"{arm}_{reg}"] = rows
    comps["G"] = {reg: {"per_seed": (v := per_seed(lambda s, r=reg: f"f1/{s}/cpu_{r}", lambda s, r=reg: f"f1/{s}/gpu_{r}")),
                        "verdict_cpu_slower": verdict(list(v.values()), False)} for reg in REGIMES}
    comps["H"] = {reg: {"per_seed": (v := per_seed(lambda s, r=reg: f"nf/{s}/baseline_{r}", lambda s, r=reg: f"a10/{s}/baseline_{r}")),
                        "within_2pct": {s: abs(x) <= 2 for s, x in v.items()},
                        "within_3pct_stop": {s: abs(x) <= 3 for s, x in v.items()}} for reg in REGIMES}
    if args.wave == "B":
        comps["I"] = {}
        for arm in ("gpu", "observed", "cpu"):
            for reg in REGIMES:
                v = per_seed(lambda s, a=arm, r=reg: f"f1/{s}/{a}_{r}", lambda s, a=arm, r=reg: f"f10/{s}/{a}_{r}")
                comps["I"][f"{arm}_{reg}"] = {
                    "per_seed": v, "verdict_alpha1_faster": verdict(list(v.values()), True),
                    "waste_alpha1_minus_alpha10_pts": {
                        s: 100 * (runs[f"f1/{s}/{arm}_{reg}"]["waste"] - runs[f"f10/{s}/{arm}_{reg}"]["waste"])
                        for s in v}}

    out = {"wave": args.wave, "runs": runs, "occupancy_invariant": occupancy, "comparisons": comps}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"runs analysed: {len(runs)}; occupancy identical over {n}: {occupancy['identical']}; wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
