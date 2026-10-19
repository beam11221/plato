"""
Stage 8, Step 3: the C3 seed analysis over 18 runs (3 arms x 2 regimes x 3 seeds).

Reads the runs through `analyze_stage7` UNCHANGED (its `parse` and metric
functions, with `PREFIX` pointed at each seed's run directories), so every
per-run number is computed exactly as Stage 7b's were.  Nothing here writes to
any run directory.

Seed -> run-directory prefix (the seed ledger in STAGE8_report.md):

    4126  stage7b        Stage 7b, reused as is
    4127  stage8_s4127
    4128  stage8_s4128

Reports: per-run soundness and the Stage 7b metric set; the occupancy
invariant across all 18; per-cell mean and sample SD over seeds; claims
(a)-(g) checked in EVERY seed, a claim whose sign flips in any seed being
"undecided"; and the secondary C2 (gpu - observed) per seed.

Usage:
    python3 analyze_stage8.py [--seeds 4126 4127 4128] [--out ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_stage7 as a7  # noqa: E402

PREFIX_OF = {4126: "stage7b", 4127: "stage8_s4127", 4128: "stage8_s4128"}
ARMS = ("baseline", "gpu", "observed")
REGIMES = ("open", "closed")
PLATEAU_BAND = (0.821, 0.837)
EVIDENCE = HERE.parent / "gpu_tiers_dataset" / "stage8_evidence"


def load(seed: int, cell: str):
    a7.PREFIX = PREFIX_OF[seed]
    return a7.parse(cell)


def gi_median_late(run) -> dict:
    """Median G_i at dispatch per tier, rounds 2,001-4,000, >= 1 observation."""
    out = {}
    for t in a7.TIERS:
        vals = [d.g_i for d in run.dispatches
                if d.tier == t and d.round > 2000 and d.observations >= 1
                and not math.isnan(d.g_i)]
        out[t] = st.median(vals) if vals else None
    x, th = out.get("xavier"), out.get("thor")
    out["xavier_thor"] = x / th if x and th else None
    return out


def per_run(run, arm: str) -> dict:
    cons = a7.conservation(run)
    curve = a7.curve_stats(run)
    w = a7.waste(run)
    share = a7.dispatch_share(run)
    agree = a7.predictor_agreement(run)
    plateau = curve["plateau"]
    lo, hi = PLATEAU_BAND
    return {
        "soundness": {
            "conservation_rel_error": cons["rel_error"],
            "conservation_ok": abs(cons["rel_error"]) < 0.03,
            "plateau": plateau,
            # Stop condition: outside the band by more than 2 points.
            "plateau_ok": (lo - 0.02) <= plateau <= (hi + 0.02),
            "plateau_in_band": lo <= plateau <= hi,
            "predictor_tier_agreement": agree,
            "predictor_ok": agree is None or agree == 1.0,
            "rounds": max(run.clock) if run.clock else 0,
        },
        "clock_at": {str(c): a7.clock_at(run, c) for c in a7.CHECKPOINTS},
        "waste": w["overall"]["wasted_frac"],
        "dispatch_share": share,
        "gi_late": gi_median_late(run) if arm != "baseline" else None,
        "curve": {k: curve[k] for k in curve if k == "plateau" or k.startswith("tta_h_")},
    }


def recycle_digest(run, n: int) -> str:
    return hashlib.md5(repr(run.recycle_seq[:n]).encode()).hexdigest()


def crossing(polaris, baseline) -> dict:
    """Rounds where the closed Polaris arm's cumulative time passes baseline's.

    `first_behind`: first round >= 100 at which Polaris is behind (clock
    larger); rounds < 100 are a few seconds of clock and flip on noise.
    `stays_behind_from`: the round after which it stays behind to 4,000 (None
    if it is not behind at the end) -- the number quoted as "the crossing".
    """
    rounds = sorted(r for r in set(polaris.clock) & set(baseline.clock) if r >= 100)
    behind = [(r, polaris.clock[r] > baseline.clock[r]) for r in rounds]
    first = next((r for r, b in behind if b), None)
    final = None
    if behind and behind[-1][1]:
        final = behind[-1][0]
        for r, b in reversed(behind):
            if not b:
                break
            final = r
    return {"first_behind": first, "stays_behind_from": final}


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else None


def pct(a, b):
    return 100.0 * (a - b) / b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[4126, 4127, 4128])
    ap.add_argument("--out", type=Path, default=EVIDENCE / "stage8_analysis.json")
    args = ap.parse_args()

    runs, report = {}, {"per_run": {}, "seeds": args.seeds}
    for seed in args.seeds:
        for arm in ARMS:
            for regime in REGIMES:
                cell = f"{arm}_{regime}"
                r = load(seed, cell)
                runs[(seed, cell)] = r
                report["per_run"][f"{seed}/{cell}"] = per_run(r, arm)

    # --- occupancy invariant across every run ------------------------------
    n = min(len(r.recycle_seq) for r in runs.values())
    digests = {f"{s}/{c}": recycle_digest(r, n) for (s, c), r in runs.items()}
    report["occupancy_invariant"] = {
        "runs": len(digests), "common_prefix_recycles": n,
        "distinct_digests": sorted(set(digests.values())),
        "identical": len(set(digests.values())) == 1,
    }

    T = {(s, c): runs[(s, c)].clock.get(4000) for (s, c) in runs}

    # --- per cell over seeds -------------------------------------------------
    cells = {}
    for arm in ARMS:
        for regime in REGIMES:
            cell = f"{arm}_{regime}"
            per = [report["per_run"][f"{s}/{cell}"] for s in args.seeds]
            ts = [p["clock_at"]["4000"] for p in per]
            ws = [p["waste"] for p in per]
            xs = [p["dispatch_share"]["overall"]["xavier"] for p in per]
            gr = [p["gi_late"]["xavier_thor"] for p in per] if arm != "baseline" else None
            cells[cell] = {
                "T4000": {"per_seed": ts, "mean": st.mean(ts), "sd": sd(ts)},
                "waste": {"per_seed": ws, "mean": st.mean(ws), "sd": sd(ws)},
                "xavier_share": {"per_seed": xs, "mean": st.mean(xs), "sd": sd(xs)},
                "gi_xavier_thor": None if gr is None else
                    {"per_seed": gr, "mean": st.mean(gr), "sd": sd(gr)},
            }
    report["cells"] = cells

    # --- claims, per seed ------------------------------------------------------
    def verdict(signs):
        return "held" if all(signs) else ("contradicted" if not any(signs) else "undecided")

    claims = {}
    for arm, key in (("gpu", "a"), ("observed", "b")):
        gaps = [pct(T[(s, f"{arm}_closed")], T[(s, f"{arm}_open")]) for s in args.seeds]
        claims[key] = {"what": f"{arm}: open faster than closed",
                       "closed_minus_open_pct": gaps, "verdict": verdict([g > 0 for g in gaps])}

    base_gaps = [pct(T[(s, "baseline_closed")], T[(s, "baseline_open")]) for s in args.seeds]
    base_open = [T[(s, "baseline_open")] for s in args.seeds]
    base_closed = [T[(s, "baseline_closed")] for s in args.seeds]
    claims["c"] = {
        "what": "baseline: no material open-closed difference",
        "closed_minus_open_pct": base_gaps,
        "baseline_open_seed_sd_pct": None if sd(base_open) is None
            else 100 * sd(base_open) / st.mean(base_open),
        "baseline_closed_seed_sd_pct": None if sd(base_closed) is None
            else 100 * sd(base_closed) / st.mean(base_closed),
        "polaris_gaps_for_scale_pct": {k: claims[k]["closed_minus_open_pct"] for k in ("a", "b")},
    }

    for regime, key, want_ahead in (("open", "d", True), ("closed", "e", False)):
        out = {}
        for arm in ("gpu", "observed"):
            gaps = [pct(T[(s, f"{arm}_{regime}")], T[(s, f"baseline_{regime}")]) for s in args.seeds]
            signs = [(g < 0) if want_ahead else (g > 0) for g in gaps]
            out[arm] = {"vs_baseline_pct": gaps, "verdict": verdict(signs)}
        claims[key] = {"what": f"{regime}: Polaris arm {'ahead of' if want_ahead else 'behind'} baseline at 4000",
                       **out}

    f = {}
    for arm in ("gpu", "observed"):
        pairs = [(report["per_run"][f"{s}/{arm}_open"]["gi_late"]["xavier_thor"],
                  report["per_run"][f"{s}/{arm}_closed"]["gi_late"]["xavier_thor"]) for s in args.seeds]
        f[arm] = {"open_closed_ratio_per_seed": pairs,
                  "verdict": verdict([c > o for o, c in pairs])}
    claims["f"] = {"what": "G_i xavier:thor larger closed than open", **f}

    claims["g"] = {"what": "round each closed Polaris arm crosses baseline",
                   **{arm: {str(s): crossing(runs[(s, f"{arm}_closed")], runs[(s, "baseline_closed")])
                            for s in args.seeds} for arm in ("gpu", "observed")}}

    c2 = {}
    for regime in REGIMES:
        gaps = [pct(T[(s, f"gpu_{regime}")], T[(s, f"observed_{regime}")]) for s in args.seeds]
        c2[regime] = {"gpu_minus_observed_pct": gaps,
                      "sign_stable": all(g < 0 for g in gaps) or all(g > 0 for g in gaps)}
    claims["C2"] = c2
    report["claims"] = claims

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))

    # --- console summary ---------------------------------------------------------
    print("soundness (all runs):")
    bad = [k for k, v in report["per_run"].items()
           if not (v["soundness"]["conservation_ok"] and v["soundness"]["plateau_ok"]
                   and v["soundness"]["predictor_ok"] and v["soundness"]["rounds"] >= 4000)]
    print("  all pass" if not bad else f"  FAIL: {bad}")
    print(f"occupancy: {report['occupancy_invariant']['runs']} runs, prefix "
          f"{n}, identical={report['occupancy_invariant']['identical']}")
    print("\nT@4000 by seed:")
    for cell in cells:
        v = cells[cell]["T4000"]
        print(f"  {cell:16s} " + "  ".join(f"{x:8.1f}" for x in v["per_seed"])
              + f"   mean {v['mean']:8.1f}  sd {v['sd'] if v['sd'] is None else round(v['sd'], 1)}")
    print("\nclaims:")
    print(json.dumps(claims, indent=1, default=lambda x: round(x, 3) if isinstance(x, float) else str(x))[:6000])
    print(f"\nwrote {args.out}")
    return 1 if bad or not report["occupancy_invariant"]["identical"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
