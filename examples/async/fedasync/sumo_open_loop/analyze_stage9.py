"""
Stage 9, Step 4: neutral priors (alpha = 1) vs Polaris default (alpha = 10).

Works on whatever runs exist, so it can be run before the queue finishes;
every comparison lists the seeds it actually used.

Run-directory prefixes (seed ledger):
    alpha = 1   stage9_s<seed>                        gpu / cpu / observed
    alpha = 10  4126 -> stage7b, 4127/4128 -> stage8_s<seed>   (cpu 4127/4128 from Stage 9's queue)
    baseline    alpha = 10 prefixes (reused; invariance accepted in distribution)

Per run: the Stage 7b metric set (through analyze_stage7, unchanged) plus
membership, first-dispatch tier mix, wait to first dispatch (PLATO clock vs
coverage entry), per-vehicle fairness (dispatch counts, never-dispatched,
Jain's index), G_i at dispatch by tier x seen/never-seen x window, and the
observed arm's running-mean tau over time.

Comparisons A-E are paired by seed and signed per seed; a sign that flips in
any available seed is "undecided".

Usage:  python3 analyze_stage9.py [--out ...]
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

SEEDS = (4126, 4127, 4128)
ARMS = ("gpu", "cpu", "observed")
REGIMES = ("open", "closed")
RUNS_DIR = HERE.parent / "runs" / "sumo_open_loop"
LOGS = HERE / "logs"
OUT = HERE.parent / "gpu_tiers_dataset" / "stage9_evidence" / "stage9_analysis.json"
EXEC_NOISE_PCT = 1.29  # same-config spread, STAGE9_report.md section 4
WINDOWS = ((1, 1000), (1001, 2000), (2001, 4000))

REPORT = re.compile(r"\[Stage7\] report round=\d+ slot=\d+ veh_id=(\S+) tier=(\S+) "
                    r"dispatch_t=(\S+) finish_t=\S+ uploaded=([01])")
IDENT = re.compile(r"\[Stage7\] report .* identity_id=(\S+)")
TAU_UNSEEN = re.compile(r"\[Stage7\] dispatch round=(\d+) .* tau_unseen=(\S+)")


def prefix(alpha: int, seed: int, arm: str) -> str:
    if alpha == 1:
        return f"stage9_s{seed}"
    return "stage7b" if seed == 4126 else f"stage8_s{seed}"


def available(pfx: str, cell: str) -> bool:
    return (LOGS / f"{pfx}_{cell}.done").exists() and (RUNS_DIR / f"{pfx}_{cell}" / "results").exists()


def windows_of_trace():
    out = {}
    with (HERE / "sumo_trace" / "e1_open_poisson_seed4126_range400" / "coverage_windows.csv").open() as fh:
        for r in csv.DictReader(fh):
            out[r["veh_id"]] = (max(0.0, float(r["t_enter_s"])), float(r["t_exit_s"]), r["tier"])
    return out


TRACE = windows_of_trace()


def jain(xs):
    s, q = sum(xs), sum(x * x for x in xs)
    return (s * s) / (len(xs) * q) if q else None


def membership(pfx: str, cell: str, run) -> dict:
    """Per admitted vehicle (p=0) or identity (p=1): dispatch counts etc."""
    regime = cell.split("_")[1]
    log = LOGS / f"{pfx}_{cell}.log"
    first, count, last_t = {}, Counter(), 0.0
    ident_first_tier = {}
    for line in log.open(errors="ignore"):
        m = REPORT.search(line)
        if not m:
            continue
        veh, tier, t = m.group(1), m.group(2), float(m.group(3))
        last_t = max(last_t, t)
        key = veh
        if regime == "closed":
            mi = IDENT.search(line)
            key = mi.group(1) if mi else veh
        count[key] += 1
        if key not in first:
            first[key] = (t, veh)
            ident_first_tier[key] = tier
    out = {}
    if regime == "open":
        admitted = {v for s in run.recycle_seq for v in [s[2]]}
        horizon = last_t - 30.0
        adm = [v for v in admitted if 0.0 < TRACE[v][0] < horizon]
        by_tier = defaultdict(list)
        for v in adm:
            by_tier[TRACE[v][2]].append(v)
        out["ever_dispatched"] = {t: sum(v in count for v in vs) / len(vs) for t, vs in by_tier.items()}
        out["ever_dispatched_all"] = sum(v in count for v in adm) / len(adm)
        out["admitted"] = {t: len(vs) for t, vs in by_tier.items()}
        waits = {t: [first[v][0] - TRACE[v][0] for v in vs if v in first] for t, vs in by_tier.items()}
        out["wait_first"] = {t: {"mean": st.mean(w), "median": st.median(w)} for t, w in waits.items() if w}
        ft = Counter(TRACE[v][2] for v in adm if v in first)
        n = sum(ft.values())
        out["first_dispatch_mix"] = {t: ft[t] / n for t in a7.TIERS}
        counts = [count.get(v, 0) for v in adm]
    else:
        counts = [count.get(str(k), 0) for k in range(1, 21)]
        ft = Counter(ident_first_tier.values())
        out["first_dispatch_mix"] = {t: ft[t] / max(1, sum(ft.values())) for t in a7.TIERS}
    out["fairness"] = {
        "units": len(counts), "never_dispatched": sum(c == 0 for c in counts) / len(counts),
        "mean": st.mean(counts), "p10": sorted(counts)[len(counts) // 10],
        "p90": sorted(counts)[9 * len(counts) // 10], "max": max(counts), "jain": jain(counts),
    }
    return out


def gi_by_window(run) -> dict:
    out = {}
    for lo, hi in WINDOWS:
        w = {}
        for t in a7.TIERS:
            for seen, keep in (("seen", lambda d: d.observations >= 1), ("unseen", lambda d: d.observations == 0)):
                v = [d.g_i for d in run.dispatches if lo <= d.round <= hi and d.tier == t
                     and keep(d) and not math.isnan(d.g_i)]
                w[f"{t}_{seen}"] = st.median(v) if v else None
        out[f"{lo}-{hi}"] = w
    x = [d.g_i for d in run.dispatches if d.round > 2000 and d.observations >= 1 and d.tier == "xavier"]
    th = [d.g_i for d in run.dispatches if d.round > 2000 and d.observations >= 1 and d.tier == "thor"]
    out["xavier_thor_late_seen"] = st.median(x) / st.median(th) if x and th else None
    return out


def tau_unseen_series(pfx, cell):
    by = defaultdict(list)
    for line in (LOGS / f"{pfx}_{cell}.log").open(errors="ignore"):
        m = TAU_UNSEEN.search(line)
        if m and m.group(2) != "-":
            r = int(m.group(1))
            by[next(f"{lo}-{hi}" for lo, hi in ((1, 250), (251, 1000), (1001, 2000), (2001, 4000)) if lo <= r <= hi)].append(float(m.group(2)))
    return {k: {"mean": st.mean(v), "last": v[-1]} for k, v in by.items()}


def per_run(pfx, cell, arm):
    a7.PREFIX = pfx
    run = a7.parse(cell)
    cons, curve = a7.conservation(run), a7.curve_stats(run)
    res = {
        # clock_at, as in every other analysis: a run whose last merge is
        # logged as round 3,999 has no exact 4,000 key (Stage 7 gpu_open did too).
        "T4000": a7.clock_at(run, 4000), "max_round": max(run.clock) if run.clock else None,
        "clock_at": {str(c): a7.clock_at(run, c) for c in a7.CHECKPOINTS},
        "waste": a7.waste(run)["overall"]["wasted_frac"],
        "dispatch_share": a7.dispatch_share(run),
        "conservation_rel_error": cons["rel_error"], "plateau": curve["plateau"],
        "tta_h": {k: v for k, v in curve.items() if k.startswith("tta_h_")},
        "predictor_tier_agreement": a7.predictor_agreement(run),
        "membership": membership(pfx, cell, run),
        "gi": gi_by_window(run) if arm != "baseline" else None,
        "_recycle_seq": run.recycle_seq,
    }
    if arm == "observed" and pfx.startswith("stage9"):
        res["tau_unseen"] = tau_unseen_series(pfx, cell)
    return res


def pct(a, b):
    return 100.0 * (a - b) / b


def verdict(vals, want_negative=True):
    if not vals:
        return "no data"
    s = [(v < 0) if want_negative else (v > 0) for v in vals]
    return "held" if all(s) else ("contradicted" if not any(s) else "undecided")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    runs = {}
    for seed in SEEDS:
        for regime in REGIMES:
            for arm in ARMS + ("baseline",):
                for alpha in ((1, 10) if arm != "baseline" else (10,)):
                    pfx = prefix(alpha, seed, arm)
                    cell = f"{arm}_{regime}"
                    if available(pfx, cell):
                        runs[(alpha, seed, cell)] = per_run(pfx, cell, arm)
    # Occupancy invariant: compare every run's realised recycle sequence over the
    # COMMON prefix.  A fixed length would be wrong: alpha = 1 gpu_open runs end
    # near t = 5,700 s with ~1,620 recycles, so a 1,800-long cut is a shorter
    # tuple for them and "differs" by construction.
    seqs = {k: v.pop("_recycle_seq") for k, v in runs.items()}
    n_common = min(len(v) for v in seqs.values())
    digests = {f"a{a}/{s}/{c}": __import__("hashlib").md5(repr(v[:n_common]).encode()).hexdigest()
               for (a, s, c), v in seqs.items()}
    occupancy = {"runs": len(digests), "common_prefix_recycles": n_common,
                 "distinct_digests": sorted(set(digests.values())),
                 "identical": len(set(digests.values())) == 1}
    T = {k: v["T4000"] for k, v in runs.items()}

    def gap(k1, k2):
        return pct(T[k1], T[k2]) if k1 in T and k2 in T else None

    comps = {}
    for regime, key in (("open", "A"), ("closed", "B")):
        g = {s: gap((1, s, f"gpu_{regime}"), (1, s, f"observed_{regime}")) for s in SEEDS}
        g = {s: v for s, v in g.items() if v is not None}
        comps[key] = {"what": f"gpu vs observed, alpha=1, {regime} (neg = gpu faster)", "per_seed": g,
                      "verdict": verdict(list(g.values())),
                      "exceeds_exec_noise": {s: abs(v) > EXEC_NOISE_PCT for s, v in g.items()}}
    comps["C"] = {}
    for arm in ARMS:
        for regime in REGIMES:
            g = {s: gap((1, s, f"{arm}_{regime}"), (10, s, f"{arm}_{regime}")) for s in SEEDS}
            comps["C"][f"{arm}_{regime}"] = {k: v for k, v in g.items() if v is not None}
    comps["D"] = {}
    for arm in ("gpu", "observed"):
        for regime in REGIMES:
            g = {s: gap((1, s, f"{arm}_{regime}"), (10, s, f"baseline_{regime}")) for s in SEEDS}
            g = {k: v for k, v in g.items() if v is not None}
            comps["D"][f"{arm}_{regime}"] = {"per_seed": g, "verdict": verdict(list(g.values()), regime == "open")}
    comps["E"] = {}
    for regime in REGIMES:
        g = {s: gap((1, s, f"cpu_{regime}"), (10, s, f"baseline_{regime}")) for s in SEEDS}
        comps["E"][f"cpu_{regime}_alpha1"] = {k: v for k, v in g.items() if v is not None}

    out = {"runs": {f"a{a}/{s}/{c}": v for (a, s, c), v in runs.items()}, "comparisons": comps,
           "occupancy_invariant": occupancy,
           "exec_noise_pct": EXEC_NOISE_PCT}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"runs analysed: {len(runs)}; occupancy identical over common prefix "
          f"{n_common}: {occupancy['identical']}")
    for k in sorted(runs):
        print("  ", k)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
