"""
Stage 9, Step 0: why is `gpu - observed` negative in all six Stage 7b/8 cells?
Log-only; reads the existing runs through analyze_stage7 unchanged.

Hypothesis (STAGE9_neutral_prior.md): `observed`'s learned staleness cannot
separate orin from thor (both settle at tau = 1) while the spec can (0.721 vs
0.822), so `gpu`'s edge is signal precision among the fast tiers.

Reports, per seed x regime, for gpu / observed / baseline:
  1. dispatch share per tier, overall and >= 1 observation, per window
  2. thor : orin dispatch ratio, and relative to baseline's
  3. observed's tau_hat by tier at dispatch (seen vehicles): median, IQR, and
     how often an orin and a thor get the same tau_hat
  4. gpu - observed through the conservation law
         T = R * mean_sleep / (C * (1 - waste))
     split into a waste channel and a mix channel, the mix channel into
     xavier share, the orin/thor split, and within-tier composition
     (Shapley average over the orders, so the parts sum exactly).
"""

from __future__ import annotations

import itertools
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_stage7 as a7  # noqa: E402

PREFIX_OF = {4126: "stage7b", 4127: "stage8_s4127", 4128: "stage8_s4128"}
SEEDS = (4126, 4127, 4128)
WINDOWS = ((1, 1000), (1001, 2000), (2001, 4000))
OUT = HERE.parent / "gpu_tiers_dataset" / "stage9_evidence" / "step0_log_check.json"


def load(seed, cell):
    a7.PREFIX = PREFIX_OF[seed]
    return a7.parse(cell)


def share(ds):
    c = Counter(d.tier for d in ds)
    n = len(ds)
    return {t: c[t] / n for t in a7.TIERS} if n else {}


def shares(run):
    rep = [d for d in run.dispatches if d.raw_in_range is not None]
    out = {"overall": share(rep), "ge1_obs": share([d for d in rep if d.observations >= 1])}
    for lo, hi in WINDOWS:
        w = [d for d in rep if lo <= d.round <= hi]
        out[f"{lo}-{hi}"] = share(w)
        out[f"{lo}-{hi}_ge1"] = share([d for d in w if d.observations >= 1])
    return out


def iqr(v):
    q = st.quantiles(v, n=4)
    return [q[0], q[2]]


def tau_profile(run):
    seen = [d for d in run.dispatches if d.observations >= 1 and not math.isnan(d.tau_hat)]
    by = {t: [d.tau_hat for d in seen if d.tier == t] for t in ("orin", "thor")}
    out = {t: {"n": len(v), "median": st.median(v), "iqr": iqr(v),
               "dist": dict(sorted(Counter(round(x) for x in v).items())[:8])}
           for t, v in by.items()}
    # Random orin x thor pairs with the same tau_hat (exact, tau is an integer count).
    co, ct = Counter(by["orin"]), Counter(by["thor"])
    out["p_same_random_pair"] = sum(co[k] * ct[k] for k in co) / (len(by["orin"]) * len(by["thor"]))
    # Pairs dispatched in the SAME selection round (the only co-candidates the log shows).
    per_round = {}
    for d in seen:
        per_round.setdefault(d.round, []).append(d)
    same = tot = 0
    for ds in per_round.values():
        for a, b in itertools.product([d for d in ds if d.tier == "orin"],
                                      [d for d in ds if d.tier == "thor"]):
            tot += 1
            same += a.tau_hat == b.tau_hat
    out["same_round_pairs"] = {"pairs": tot, "same_tau": same,
                               "frac": same / tot if tot else None}
    return out


def mix_terms(run):
    rep = [d for d in run.dispatches if d.raw_in_range is not None]
    n = len(rep)
    x = sum(d.tier == "xavier" for d in rep) / n
    o = sum(d.tier == "orin" for d in rep)
    t = sum(d.tier == "thor" for d in rep)
    q = t / (o + t)
    m = {tr: st.mean([d.sleep for d in rep if d.tier == tr]) for tr in a7.TIERS}
    w = sum(not d.raw_in_range for d in rep) / n
    return {"x": x, "q": q, "m": m, "waste": w,
            "mean_sleep": st.mean([d.sleep for d in rep]), "T": run.clock[max(run.clock)]}


def ms(x, q, m):
    return x * m["xavier"] + (1 - x) * ((1 - q) * m["orin"] + q * m["thor"])


def decompose(g, o):
    """ln T_g - ln T_o = mix - ln((1-w_g)/(1-w_o)); mix split by Shapley."""
    factors = ("x", "q", "m")
    val = {}
    for combo in itertools.product((0, 1), repeat=3):
        pick = {f: (g if c else o)[f] for f, c in zip(factors, combo)}
        val[combo] = math.log(ms(pick["x"], pick["q"], pick["m"]))
    shap = {f: 0.0 for f in factors}
    for perm in itertools.permutations(range(3)):
        cur = [0, 0, 0]
        for i in perm:
            before = val[tuple(cur)]
            cur[i] = 1
            shap[factors[i]] += (val[tuple(cur)] - before) / 6
    waste_ch = -(math.log(1 - g["waste"]) - math.log(1 - o["waste"]))
    total = math.log(g["T"] / o["T"])
    return {
        "measured_gpu_minus_obs_pct": 100 * (g["T"] / o["T"] - 1),
        "log_gap_pct": 100 * total,
        "xavier_share_pct": 100 * shap["x"],
        "orin_thor_split_pct": 100 * shap["q"],
        "within_tier_pct": 100 * shap["m"],
        "mix_total_pct": 100 * sum(shap.values()),
        "waste_pct": 100 * waste_ch,
        "residual_pct": 100 * (total - sum(shap.values()) - waste_ch),
        "inputs": {"gpu": {k: g[k] for k in ("x", "q", "waste")},
                   "observed": {k: o[k] for k in ("x", "q", "waste")}},
    }


def main():
    res = {}
    for seed in SEEDS:
        for regime in ("open", "closed"):
            runs = {arm: load(seed, f"{arm}_{regime}") for arm in ("gpu", "observed", "baseline")}
            sh = {arm: shares(r) for arm, r in runs.items()}
            ratio = {arm: sh[arm]["overall"]["thor"] / sh[arm]["overall"]["orin"] for arm in sh}
            ratio_ge1 = {arm: sh[arm]["ge1_obs"]["thor"] / sh[arm]["ge1_obs"]["orin"] for arm in sh}
            res[f"{seed}/{regime}"] = {
                "shares": sh,
                "thor_orin": ratio,
                "thor_orin_rel_baseline": {a: ratio[a] / ratio["baseline"] for a in ("gpu", "observed")},
                "thor_orin_ge1": ratio_ge1,
                "observed_tau": tau_profile(runs["observed"]),
                "decomposition": decompose(mix_terms(runs["gpu"]), mix_terms(runs["observed"])),
            }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))

    print("thor:orin dispatch ratio (overall) gpu / observed / baseline  |  gpu>obs?")
    for k, v in res.items():
        r = v["thor_orin"]
        print(f"  {k:12s} {r['gpu']:.3f} / {r['observed']:.3f} / {r['baseline']:.3f}   "
              f"{'yes' if r['gpu'] > r['observed'] else 'NO'}   (>=1 obs: "
              f"{v['thor_orin_ge1']['gpu']:.3f} / {v['thor_orin_ge1']['observed']:.3f})")
    print("\nobserved tau_hat at dispatch (seen): orin median [IQR] | thor median [IQR] | P(same) random pair | same-round pairs")
    for k, v in res.items():
        t = v["observed_tau"]
        sr = t["same_round_pairs"]
        print(f"  {k:12s} {t['orin']['median']:.0f} {[round(x, 1) for x in t['orin']['iqr']]} | "
              f"{t['thor']['median']:.0f} {[round(x, 1) for x in t['thor']['iqr']]} | "
              f"{t['p_same_random_pair']:.3f} | {sr['same_tau']}/{sr['pairs']}")
    print("\ngpu - observed via conservation law (% of T, log-points): total = xavier + orin/thor + within-tier + waste (+ residual)")
    for k, v in res.items():
        d = v["decomposition"]
        print(f"  {k:12s} measured {d['measured_gpu_minus_obs_pct']:+6.2f} | log {d['log_gap_pct']:+6.2f} = "
              f"x {d['xavier_share_pct']:+6.2f}  o/t {d['orin_thor_split_pct']:+6.2f}  "
              f"within {d['within_tier_pct']:+6.2f}  waste {d['waste_pct']:+6.2f}  resid {d['residual_pct']:+5.2f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
