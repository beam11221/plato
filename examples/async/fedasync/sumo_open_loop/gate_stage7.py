"""
Stage 7 Step 2 — the pre-run gate.  Trace and allocator only, no FL training.

Seven deliverables, all from `sumo_trace/e1_open_poisson_seed4126_range400/`
plus `stage7_population.py`'s allocator, per prompts/STAGE7_open_population.md:

  1. crossing-time distribution        5. tier mix among in-coverage vehicles
  2. occupancy  -> sets M              6. slot-exhaustion rate at that M
  3. feasible-and-idle pool per round  7. warm-up / end-of-sim window handling
  4. kappa, dispatches per vehicle

Plus the realised per-tier sleep mean and CV over all 10,523 vehicles, checked
against the closed-loop fleet's 14.37 / 1.78 / 1.07 s at CV 15% (the stage's
own stop condition).

Usage:
    python3 gate_stage7.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage7_population import (  # noqa: E402
    SIGMA,
    T_TIER,
    SlotAllocator,
    vehicle_sleep_s,
)

TRACE = HERE / "sumo_trace" / "e1_open_poisson_seed4126_range400"
EVIDENCE = HERE.parent / "gpu_tiers_dataset" / "stage7_evidence"

BASE_SEED = 4126
CONCURRENCY = 4          # C
ROUNDS = 4000
START_TIME = 0.0         # remap_fleet.py's clip: training starts at t = 0
CLOSED_LOOP_CROSSING_S = 2 * 400.0 / 19.444444   # 41.142857, the vmax crossing
DATASET_SIZE = 50_000    # M x partition_size must equal this

# Closed-loop references, for the comparisons the prompt asks for.
CLOSED_RESIDENCE = {"median": 45.0, "p10": 42.0, "p90": 52.0}
CLOSED_SLEEP_MEAN = {"xavier": 14.37, "orin": 1.78, "thor": 1.07}
CLOSED_SLEEP_CV = 0.15


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def load_windows() -> list[dict]:
    with (TRACE / "coverage_windows.csv").open(newline="") as fh:
        return [
            {
                "veh_id": r["veh_id"],
                "tier": r["tier"],
                "t_enter_s": float(r["t_enter_s"]),
                "t_exit_s": float(r["t_exit_s"]),
                "residence_s": float(r["residence_s"]),
            }
            for r in csv.DictReader(fh)
        ]


def load_vehicles() -> list[dict]:
    with (TRACE / "vehicles.csv").open(newline="") as fh:
        return [
            {"veh_id": r["veh_id"], "tier": r["tier"],
             "n_passes": int(r["n_passes"]), "n_windows": int(r["n_windows"])}
            for r in csv.DictReader(fh)
        ]


def clip_windows(windows: list[dict]) -> tuple[list[dict], dict]:
    """Apply remap_fleet.py's t >= START_TIME rule and report what it removed."""
    kept, dropped, clipped = [], 0, 0
    for w in windows:
        if w["t_exit_s"] < START_TIME:
            dropped += 1
            continue
        w = dict(w)
        if w["t_enter_s"] < START_TIME:
            w["t_enter_s"] = START_TIME
            w["residence_s"] = w["t_exit_s"] - w["t_enter_s"]
            clipped += 1
        kept.append(w)
    return kept, {"dropped_entirely": dropped, "clipped_at_start": clipped}


def occupancy_series(windows: list[dict]) -> tuple[list[int], list[float]]:
    """Concurrent in-coverage count, sampled once per second over the trace."""
    events: list[tuple[float, int]] = []
    for w in windows:
        events.append((w["t_enter_s"], +1))
        events.append((w["t_exit_s"], -1))
    events.sort()

    t_end = max(w["t_exit_s"] for w in windows)
    counts: list[int] = []
    times: list[float] = []
    current = 0
    i = 0
    t = START_TIME
    while t <= t_end:
        while i < len(events) and events[i][0] <= t:
            current += events[i][1]
            i += 1
        counts.append(current)
        times.append(t)
        t += 1.0
    return counts, times


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=EVIDENCE / "stage7_gate.json")
    parser.add_argument(
        "--slots", type=int, default=None,
        help="Override M. Default is ceil(p99 occupancy); see the gate report "
             "for why the chosen M is not always that value (50,000 must divide "
             "by M).",
    )
    args = parser.parse_args()

    raw_windows = load_windows()
    vehicles = load_vehicles()
    report: dict = {}

    print(f"trace: {TRACE.name}")
    print(f"  vehicles.csv:          {len(vehicles)}")
    print(f"  coverage_windows.csv:  {len(raw_windows)}")

    # ---- 7. warm-up handling (done first: everything else uses the clip) ----
    windows, clip_stats = clip_windows(raw_windows)
    negative_enters = sum(1 for w in raw_windows if w["t_enter_s"] < 0)
    report["warmup"] = {
        "raw_windows": len(raw_windows),
        "windows_with_negative_t_enter": negative_enters,
        "dropped_entirely_before_t0": clip_stats["dropped_entirely"],
        "clipped_at_t0": clip_stats["clipped_at_start"],
        "windows_after_clip": len(windows),
        "n_windows_open_at_end_dropped_per_config": 9,
    }
    print("\n=== 7. warm-up / end-of-sim handling ===")
    print(f"  windows with t_enter < 0:        {negative_enters}")
    print(f"  dropped entirely (t_exit < 0):   {clip_stats['dropped_entirely']}")
    print(f"  clipped to start at t = 0:       {clip_stats['clipped_at_start']}")
    print(f"  usable windows after clip:       {len(windows)}")

    # ---- 1. crossing-time distribution ----
    residence = [w["residence_s"] for w in windows]
    below = sum(1 for r in residence if r < CLOSED_LOOP_CROSSING_S)
    report["crossing_time"] = {
        "median": st.median(residence),
        "p10": percentile(residence, 0.10),
        "p90": percentile(residence, 0.90),
        "mean": st.mean(residence),
        "min": min(residence), "max": max(residence),
        "fraction_below_vmax_crossing": below / len(residence),
        "vmax_crossing_s": CLOSED_LOOP_CROSSING_S,
        "closed_loop_reference": CLOSED_RESIDENCE,
    }
    print("\n=== 1. crossing-time distribution (residence_s) ===")
    print(f"  median={st.median(residence):.1f}s  p10={percentile(residence,0.10):.1f}s  "
          f"p90={percentile(residence,0.90):.1f}s  (closed loop: 45.0 / 42 / 52)")
    print(f"  below {CLOSED_LOOP_CROSSING_S:.6f}s: {below}/{len(residence)} = "
          f"{100*below/len(residence):.2f}%")

    # ---- 2. occupancy -> M ----
    counts, _times = occupancy_series(windows)
    p99 = percentile([float(c) for c in counts], 0.99)
    M_from_p99 = math.ceil(p99)
    M = args.slots if args.slots is not None else M_from_p99
    partition_size = DATASET_SIZE // M
    report["occupancy"] = {
        "median": st.median(counts), "p90": percentile([float(c) for c in counts], 0.90),
        "p99": p99, "max": max(counts), "mean": st.mean(counts),
        "samples": len(counts),
        "M_from_p99": M_from_p99,
        "M_chosen": M, "partition_size": partition_size,
        "M_times_partition_size": M * partition_size,
        "M_divides_dataset_exactly": DATASET_SIZE % M == 0,
    }
    print("\n=== 2. occupancy (concurrent in-coverage vehicles) ===")
    print(f"  median={st.median(counts):.1f}  p90={percentile([float(c) for c in counts],0.90):.1f}  "
          f"p99={p99:.1f}  max={max(counts)}  mean={st.mean(counts):.2f}")
    # New in the revised prompt: how often the RSU is at or beyond capacity.
    at_capacity = sum(1 for c in counts if c >= M) / len(counts)
    report["occupancy"]["fraction_time_at_or_above_M"] = at_capacity
    print(f"  -> M = {M} (fixed by the stage prompt), "
          f"partition_size = {partition_size}  [M*partition_size = {M*partition_size}]")
    print(f"  fraction of time occupancy >= M: {100*at_capacity:.2f}%  "
          f"(prompt expects roughly 4-5%)")

    # ---- per-vehicle sleep model: realised per-tier mean and CV ----
    sleeps_by_tier: dict[str, list[float]] = defaultdict(list)
    for v in vehicles:
        sleeps_by_tier[v["tier"]].append(
            vehicle_sleep_s(BASE_SEED, v["veh_id"], v["tier"])
        )
    sleep_stats = {}
    for tier in T_TIER:
        vals = sleeps_by_tier.get(tier, [])
        mean = st.mean(vals)
        cv = st.stdev(vals) / mean
        sleep_stats[tier] = {
            "n": len(vals), "mean": mean, "cv": cv,
            "theoretical_mean": T_TIER[tier] * math.exp(SIGMA ** 2 / 2),
            "closed_loop_mean": CLOSED_SLEEP_MEAN[tier],
        }
    report["sleep_model"] = {"per_tier": sleep_stats, "sigma": SIGMA,
                             "closed_loop_cv": CLOSED_SLEEP_CV}
    print("\n=== per-vehicle sleep model (realised over all vehicles) ===")
    for tier, s in sleep_stats.items():
        print(f"  {tier:7s} n={s['n']:5d}  mean={s['mean']:.4f}s  CV={100*s['cv']:.2f}%  "
              f"(theory {s['theoretical_mean']:.4f}s, closed-loop fleet {s['closed_loop_mean']}s)")

    # ---- seconds-per-round, from the conservation law ----
    in_cov_vehicles = {w["veh_id"] for w in windows}
    tier_of = {v["veh_id"]: v["tier"] for v in vehicles}
    mean_sleep = st.mean([
        vehicle_sleep_s(BASE_SEED, vid, tier_of[vid]) for vid in in_cov_vehicles
    ])
    t_round = mean_sleep / CONCURRENCY
    span = ROUNDS * t_round
    report["round_model"] = {
        "mean_sleep_s": mean_sleep, "t_round_s": t_round,
        "rounds": ROUNDS, "simulated_span_s": span,
        "trace_usable_s": max(w["t_exit_s"] for w in windows),
        "fits_without_wrap": span <= max(w["t_exit_s"] for w in windows),
    }
    print("\n=== round model (conservation law, no waste term) ===")
    print(f"  mean_sleep over in-coverage vehicles = {mean_sleep:.4f}s")
    print(f"  T_round = mean_sleep / C = {t_round:.4f}s  ->  4000 rounds = {span:.0f}s")
    print(f"  trace usable to {max(w['t_exit_s'] for w in windows):.0f}s  "
          f"-> fits without wrap: {span <= max(w['t_exit_s'] for w in windows)}")

    # ---- 3. feasible-and-idle pool per round ----
    # Sample occupancy at each round instant; subtract the C busy slots.
    # A vehicle with no slot is not selectable, so the pool is capped by M
    # before the C busy slots come off it.
    pool_per_round = []
    for r in range(ROUNDS):
        t = r * t_round
        idx = int(t)
        if idx >= len(counts):
            break
        pool_per_round.append(max(0, min(counts[idx], M) - CONCURRENCY))
    two_c = 2 * CONCURRENCY
    below_2c = sum(1 for p in pool_per_round if p < two_c)
    report["feasible_idle_pool"] = {
        "median": st.median(pool_per_round),
        "p10": percentile([float(p) for p in pool_per_round], 0.10),
        "p90": percentile([float(p) for p in pool_per_round], 0.90),
        "rounds_sampled": len(pool_per_round),
        "fraction_below_2C": below_2c / len(pool_per_round),
        "two_C": two_c,
        "gate_passes": st.median(pool_per_round) >= two_c,
    }
    print("\n=== 3. feasible-and-idle pool per round ===")
    print(f"  median={st.median(pool_per_round):.1f}  "
          f"p10={percentile([float(p) for p in pool_per_round],0.10):.1f}  "
          f"p90={percentile([float(p) for p in pool_per_round],0.90):.1f}")
    print(f"  rounds below 2C={two_c}: {below_2c}/{len(pool_per_round)} = "
          f"{100*below_2c/len(pool_per_round):.2f}%")
    print(f"  GATE (median >= 2C): "
          f"{'PASS' if st.median(pool_per_round) >= two_c else 'FAIL — STOP'}")

    # ---- 4. kappa: dispatches per vehicle before it exits ----
    mean_pool = st.mean(pool_per_round) + CONCURRENCY  # eligible pool incl. busy
    kappas = [w["residence_s"] / t_round / mean_pool for w in windows]
    report["kappa"] = {
        "mean": st.mean(kappas), "median": st.median(kappas),
        "p10": percentile(kappas, 0.10), "p90": percentile(kappas, 0.90),
        "max": max(kappas),
        "eligible_pool_used": mean_pool,
        "closed_loop_reference": "70-188", "k_star_reference": "30-70",
    }
    print("\n=== 4. kappa (expected dispatches per vehicle before exit) ===")
    print(f"  mean={st.mean(kappas):.2f}  median={st.median(kappas):.2f}  "
          f"p10={percentile(kappas,0.10):.2f}  p90={percentile(kappas,0.90):.2f}  "
          f"max={max(kappas):.2f}")
    print(f"  (closed-loop cells ran at 70-188; k* from Stage 3 is 30-70)")

    # ---- 5. tier mix among vehicles that appear in coverage ----
    mix = Counter(tier_of[v] for v in in_cov_vehicles)
    total_in_cov = sum(mix.values())
    report["tier_mix"] = {
        "in_coverage": {t: mix[t] / total_in_cov for t in T_TIER},
        "counts": {t: mix[t] for t in T_TIER},
        "config_realised": {"xavier": 0.4911, "orin": 0.3491, "thor": 0.1597},
        "n_vehicles_in_coverage": total_in_cov,
    }
    print("\n=== 5. tier mix among in-coverage vehicles ===")
    for t in T_TIER:
        print(f"  {t:7s} {mix[t]:5d}  {100*mix[t]/total_in_cov:.2f}%  "
              f"(config realised {100*report['tier_mix']['config_realised'][t]:.2f}%)")

    # ---- 6. slot exhaustion at the chosen M ----
    events: list[tuple[float, int, str, str]] = []
    for w in windows:
        events.append((w["t_enter_s"], 1, w["veh_id"], w["tier"]))
        events.append((w["t_exit_s"], 0, w["veh_id"], w["tier"]))
    # Departures before arrivals at the same instant, so a slot freed at t is
    # available to a vehicle arriving at t.
    events.sort(key=lambda e: (e[0], e[1]))

    allocator = SlotAllocator(slots=M, p_recur=0.0, base_seed=BASE_SEED)
    arrivals = 0
    dropped_mix: Counter = Counter()
    admitted_mix: Counter = Counter()
    # The occupancy-sequence hash the revised prompt requires: the ordered
    # (time, event, vehicle, slot) stream the allocator produces.  It is a
    # function of the TRACE alone — a slot is taken on coverage entry and
    # released on coverage exit, never on selection — so every one of the eight
    # runs must reproduce this exact digest.  An arm that differs has tied slot
    # holding to something the selector influences, which is the defect the
    # check exists to catch.
    occupancy_hasher = hashlib.blake2b(digest_size=16)
    for t, is_arrival, veh_id, tier in events:
        if is_arrival:
            arrivals += 1
            slot, _event = allocator.assign(veh_id, tier)
            if slot is None:
                dropped_mix[tier] += 1
            else:
                admitted_mix[tier] += 1
            occupancy_hasher.update(f"{t:.3f}|A|{veh_id}|{slot}\n".encode())
        else:
            slot = allocator.release(veh_id)
            occupancy_hasher.update(f"{t:.3f}|R|{veh_id}|{slot}\n".encode())
    occupancy_digest = occupancy_hasher.hexdigest()

    rate = allocator.slot_exhausted_events / arrivals
    n_dropped = sum(dropped_mix.values()) or 1
    n_admitted = sum(admitted_mix.values())
    mix_delta = {
        t: 100 * dropped_mix[t] / n_dropped - 100 * admitted_mix[t] / n_admitted
        for t in T_TIER
    }
    report["slot_exhaustion"] = {
        "M": M, "arrivals": arrivals,
        "slot_exhausted_events": allocator.slot_exhausted_events,
        "rate": rate,
        "peak_occupancy": allocator.peak_occupancy,
        "recycles": len(allocator.recycles),
        "dropped_tier_mix": {t: dropped_mix[t] / n_dropped for t in T_TIER},
        "admitted_tier_mix": {t: admitted_mix[t] / n_admitted for t in T_TIER},
        "dropped_minus_admitted_pt": mix_delta,
        "occupancy_sequence_digest": occupancy_digest,
        "gate_passes_rate": rate <= 0.15,
        "gate_passes_mix": max(abs(v) for v in mix_delta.values()) <= 5.0,
    }
    print(f"\n=== 6. slot exhaustion at M = {M} (whole-trace replay) ===")
    print(f"  arrivals={arrivals}  turned away={allocator.slot_exhausted_events}  "
          f"rate={100*rate:.4f}%   (prompt expects ~4-5%; stop condition is >15%)")
    print(f"  peak occupancy={allocator.peak_occupancy}  recycles={len(allocator.recycles)}")
    print("  tier mix of turned-away vs admitted:")
    for t in T_TIER:
        print(f"    {t:7s} dropped={100*dropped_mix[t]/n_dropped:5.2f}%  "
              f"admitted={100*admitted_mix[t]/n_admitted:5.2f}%  "
              f"delta={mix_delta[t]:+.2f}pt")
    print(f"  occupancy-sequence digest: {occupancy_digest}")
    print(f"  GATE (rate <= 15%): {'PASS' if rate <= 0.15 else 'FAIL — STOP'}")
    print(f"  GATE (turned-away mix matches admitted): "
          f"{'PASS' if max(abs(v) for v in mix_delta.values()) <= 5.0 else 'FAIL — STOP'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
