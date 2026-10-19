"""
Stage 6 analysis (prompts/STAGE6_seeds.md).

Self-contained for the same reason analyze_stage5b.py is. Reads Stage 5b's
and Task 4's JSON for the two reused reference rows (not re-run). Handles
both Batch A (seed 4126, one run per cell) and Batch B (seeds 4127/4128,
paired per cell) -- `main()` discovers whichever run directories exist under
`runs/sumo_close_loop/stage6_*` and reports on those.

Usage:
    python3 analyze_stage6.py              # every stage6_* run dir found
    python3 analyze_stage6.py --validate    # parse + sanity checks only
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import statistics as st
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEDASYNC = HERE.parent
RUNS = FEDASYNC / "runs" / "sumo_close_loop"
GPU_TIERS_DATASET = FEDASYNC / "gpu_tiers_dataset"
EVIDENCE = GPU_TIERS_DATASET / "stage6_evidence"
TASK4_STEPD = GPU_TIERS_DATASET / "stage4_evidence" / "task4_stepD.json"
STAGE5_ANALYSIS = GPU_TIERS_DATASET / "stage5_evidence" / "stage5_analysis.json"
STAGE5B_ANALYSIS = GPU_TIERS_DATASET / "stage5b_evidence" / "stage5b_analysis.json"

sys.path.insert(0, str(HERE))
from sumo_mobility import CoverageOracle  # noqa: E402

CELL = "n20r400"
CONCURRENCY = 4
SLEEP_CSV = HERE / "output" / f"client_sleep_times_{CELL}.csv"
COVERAGE_CSV = HERE / "input" / f"coverage_{CELL}.csv"
CLIENTS_CSV = HERE / "input" / f"clients_{CELL}.csv"

TIER_TRAIN = {"xavier": 15.0, "orin": 1.923077, "thor": 1.0}
WINDOWS = ((1, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000))
CHECKPOINTS = (100, 250, 500, 1000, 2000, 4000)
THRESHOLDS = (70, 75, 78, 80)
PLATEAU_FRACTION = 0.10
SMOOTH_WINDOW = 11
NOISE_WINDOW = 101
EVAL_INTERVAL = 10

ROUND_LINE = re.compile(r"Starting round (\d+)/(\d+)")
SELECT_LINE = re.compile(r"Selecting client #(\d+) for training")
MERGE_LINE = re.compile(
    r"FedAsync: step \d+ - client #(\d+), effective_round=(\d+), "
    r"staleness=(\d+), mixing=([\d.]+)"
)
REPORT_LINE = re.compile(
    r"\[SumoRsu\] report round=(\d+) client=(\d+) tier=(\S+) "
    r"dispatch_t=([\d.eE+-]+) finish_t=([\d.eE+-]+) uploaded=([01]) "
    r"raw_in_range=([01]) policy=(\S+)"
)
SUMO_RANGE_LINE = re.compile(
    r"\[SumoRsu\] round=(-?\d+) t=([\d.]+) s: (\d+) of (\d+) client\(s\) in RSU range"
)
STAGE0_ROUND_LINE = re.compile(
    r"\[Stage0\] round=(-?\d+) t_hat_source=(\S+) r_i_mode=(\S+) "
    r"eligible_after_range_gate=(\d+) eligible_after_stage0=(\d+) "
    r"stage0_rejected_ids=(\[[^\]]*\]) stage0_fallback=([01])"
)


@dataclass
class Dispatch:
    index: int
    round: int
    client_id: int
    tier: str
    start: float = 0.0
    finish: float = 0.0
    reported: bool = False
    uploaded: bool = False
    raw_in_range: bool = False
    merge_round: int | None = None


@dataclass
class RoundStats:
    round: int
    t: float | None = None
    eligible_after_busy: int | None = None
    eligible_after_range_gate: int | None = None
    eligible_after_stage0: int | None = None
    stage0_rejected_ids: list[int] = field(default_factory=list)
    stage0_fallback: bool = False


@dataclass
class Run:
    name: str
    dispatches: list[Dispatch] = field(default_factory=list)
    curve: list[dict] = field(default_factory=list)
    clock: dict[int, float] = field(default_factory=dict)
    round_stats: dict[int, RoundStats] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    n_report_lines: int = 0


def load_sleeps() -> dict[int, float]:
    with SLEEP_CSV.open(newline="") as fh:
        return {int(r["client_id"]): float(r["sleep_time"]) for r in csv.DictReader(fh)}


def load_tiers() -> dict[int, str]:
    with CLIENTS_CSV.open(newline="") as fh:
        return {int(r["client_id"]): r["tier_id"] for r in csv.DictReader(fh)}


def discover_run_dirs() -> list[str]:
    return sorted(
        p.name[len("stage6_"):] for p in RUNS.glob("stage6_*")
        if p.is_dir() and (p / "results").exists()
    )


def server_log_path(directory: Path) -> Path:
    logs = sorted(directory.glob("server_*.log"))
    if len(logs) != 1:
        raise SystemExit(f"{len(logs)} server logs under {directory}; cannot pick one")
    return logs[0]


def results_csv_path(directory: Path) -> Path:
    found = sorted(directory.glob("results/*.csv"))
    if len(found) != 1:
        raise SystemExit(f"{len(found)} results CSVs under {directory}")
    return found[0]


def parse_and_replay(name: str, sleeps: dict[int, float], tiers: dict[int, str]) -> Run:
    run = Run(name=name)
    directory = RUNS / f"stage6_{name}"
    slots = [0.0] * CONCURRENCY
    pending_report: dict[int, list[Dispatch]] = defaultdict(list)
    pending_merge: dict[int, list[Dispatch]] = defaultdict(list)
    current_round = 0

    with server_log_path(directory).open(errors="replace") as fh:
        for line in fh:
            match = ROUND_LINE.search(line)
            if match:
                current_round = int(match.group(1))
                continue

            match = SUMO_RANGE_LINE.search(line)
            if match:
                rnd = int(match.group(1))
                rs = run.round_stats.setdefault(rnd, RoundStats(round=rnd))
                rs.t = float(match.group(2))
                rs.eligible_after_range_gate = int(match.group(3))
                rs.eligible_after_busy = int(match.group(4))
                continue

            match = STAGE0_ROUND_LINE.search(line)
            if match:
                rnd = int(match.group(1))
                rs = run.round_stats.setdefault(rnd, RoundStats(round=rnd))
                rs.eligible_after_stage0 = int(match.group(5))
                rs.stage0_rejected_ids = ast.literal_eval(match.group(6))
                rs.stage0_fallback = match.group(7) == "1"
                continue

            match = SELECT_LINE.search(line)
            if match:
                cid = int(match.group(1))
                slot = min(range(CONCURRENCY), key=lambda i: slots[i])
                start = slots[slot]
                finish = start + sleeps[cid]
                slots[slot] = finish
                item = Dispatch(
                    index=len(run.dispatches), round=current_round, client_id=cid,
                    tier=tiers[cid], start=start, finish=finish,
                )
                run.dispatches.append(item)
                pending_report[cid].append(item)
                continue

            match = REPORT_LINE.search(line)
            if match:
                run.n_report_lines += 1
                cid = int(match.group(2))
                queue = pending_report[cid]
                if not queue:
                    run.problems.append(f"report for client {cid} with no unconsumed dispatch")
                    continue
                item = queue.pop(0)
                item.reported = True
                item.uploaded = match.group(6) == "1"
                item.raw_in_range = match.group(7) == "1"
                if item.uploaded:
                    pending_merge[cid].append(item)
                continue

            match = MERGE_LINE.search(line)
            if match:
                cid = int(match.group(1))
                effective_round = int(match.group(2))
                queue = pending_merge[cid]
                if not queue:
                    run.problems.append(f"round {effective_round} aggregated client {cid} "
                                         f"with no uploaded, unmerged dispatch")
                    continue
                item = queue.pop(0)
                item.merge_round = effective_round
                continue

    clock: dict[int, float] = {}
    for item in run.dispatches:
        if item.merge_round is not None:
            prev = clock.get(item.merge_round)
            clock[item.merge_round] = max(prev, item.finish) if prev else item.finish
    run.clock = clock

    with results_csv_path(directory).open(newline="") as fh:
        for row in csv.DictReader(fh):
            rnd = int(row["round"])
            if rnd not in clock:
                continue
            run.curve.append({
                "round": rnd, "accuracy": float(row["accuracy"]),
                "sim_time_reconstructed": clock[rnd],
            })
    return run


def validate(run: Run) -> list[str]:
    notes = [
        f"dispatches={len(run.dispatches)}  report lines={run.n_report_lines}",
        f"round_stats captured={len(run.round_stats)}",
    ]
    if run.curve:
        notes.append(f"round {run.curve[-1]['round']}: "
                      f"reconstructed={run.curve[-1]['sim_time_reconstructed']:.1f}s")
    return [f"  {n}" for n in notes] + [f"  ** {p}" for p in run.problems]


def in_window(rnd: int, lo: int, hi: int) -> bool:
    return (lo <= rnd <= hi) if lo == 1 else (lo < rnd <= hi)


def waste_table(run: Run) -> dict:
    reported = [d for d in run.dispatches if d.reported]

    def tally(items):
        n = len(items)
        wasted = sum(1 for d in items if not d.uploaded)
        return {"dispatches": n, "wasted": wasted, "wasted_frac": wasted / n if n else 0.0}

    out = {"overall": tally(reported), "per_tier": {}}
    for tier in TIER_TRAIN:
        out["per_tier"][tier] = tally([d for d in reported if d.tier == tier])
    return out


def dispatch_share(run: Run) -> dict:
    reported = [d for d in run.dispatches if d.reported]
    counts = defaultdict(int)
    for d in reported:
        counts[d.tier] += 1
    return {t: counts[t] / len(reported) for t in TIER_TRAIN} if reported else {}


def clock_at(run: Run, rnd: int) -> float | None:
    if rnd in run.clock:
        return run.clock[rnd]
    earlier = [r for r in run.clock if r <= rnd]
    return run.clock[max(earlier)] if earlier else None


def conservation_check(run: Run, tier_sleep: dict[str, float]) -> dict:
    reported = [d for d in run.dispatches if d.reported]
    if not run.clock:
        return {"rounds": None, "measured": None, "predicted": None, "rel_error": None}
    rounds = max(run.clock)
    measured = run.clock[rounds]
    share = {t: sum(1 for d in reported if d.tier == t) / len(reported) for t in TIER_TRAIN}
    mean_sleep = sum(share[t] * tier_sleep[t] for t in TIER_TRAIN)
    waste = sum(1 for d in reported if not d.uploaded) / len(reported)
    predicted = rounds * mean_sleep / (CONCURRENCY * (1 - waste))
    return {
        "rounds": rounds, "measured": measured, "predicted": predicted,
        "rel_error": (predicted - measured) / measured,
    }


def rolling_median(values: list[float], window: int) -> list[float]:
    half = window // 2
    return [st.median(values[max(0, i - half): min(len(values), i + half + 1)])
            for i in range(len(values))]


def curve_stats(run: Run) -> dict:
    rows = [r for r in run.curve if r["round"] % EVAL_INTERVAL == 0]
    if not rows:
        return {}
    acc = [r["accuracy"] for r in rows]
    tail = acc[-max(1, int(len(acc) * PLATEAU_FRACTION)):]
    smoothed = rolling_median(acc, SMOOTH_WINDOW)
    baseline = rolling_median(acc, NOISE_WINDOW)
    residuals = [a - b for a, b in zip(acc, baseline)]
    stats = {
        "plateau": st.median(tail), "best": max(acc),
        "noise_sd": st.stdev(residuals) if len(residuals) > 1 else None,
    }
    for target in THRESHOLDS:
        hit = next((r for r, v in zip(rows, smoothed) if v >= target / 100.0), None)
        stats[f"rounds_to_{target}"] = hit["round"] if hit else None
        stats[f"tta_h_{target}"] = hit["sim_time_reconstructed"] / 3600.0 if hit else None
    return stats


def stage0_fallback_accounting(run: Run) -> dict:
    total_rounds = len(run.round_stats)
    total_fallback = sum(1 for rs in run.round_stats.values() if rs.stage0_fallback)
    return {
        "rounds": total_rounds, "fallback": total_fallback,
        "fallback_frac": total_fallback / total_rounds if total_rounds else None,
    }


def stage0_pool_sizes_min_median(run: Run) -> float | None:
    mins = []
    for lo, hi in WINDOWS:
        vals = [rs.eligible_after_stage0 for r, rs in run.round_stats.items()
                if in_window(r, lo, hi) and rs.eligible_after_stage0 is not None]
        if vals:
            mins.append(st.median(vals))
    return min(mins) if mins else None


def stage0_confusion(run: Run, sleeps: dict[int, float], tiers: dict[int, str],
                     oracle: CoverageOracle) -> dict:
    per_tier: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rs in run.round_stats.values():
        if rs.t is None:
            continue
        for cid in rs.stage0_rejected_ids:
            tier = tiers.get(cid, "unknown")
            end = oracle.window_end(cid, rs.t)
            would_upload = end is not None and end >= rs.t + sleeps[cid]
            per_tier[tier]["killed_viable" if would_upload else "killed_waste"] += 1
    for d in run.dispatches:
        if not d.reported:
            continue
        per_tier[d.tier]["passed_useful" if d.raw_in_range else "missed_waste"] += 1
    return {t: dict(v) for t, v in per_tier.items()}


def decompose(rounds: int, control_waste: float, control_mean_sleep: float,
             control_T: float, run_waste: float, run_share: dict,
             tier_sleep: dict[str, float]) -> dict:
    run_mean_sleep = sum(run_share.get(t, 0.0) * tier_sleep[t] for t in TIER_TRAIN)
    t_waste_only = rounds * control_mean_sleep / (CONCURRENCY * (1 - run_waste))
    t_mix_only = rounds * run_mean_sleep / (CONCURRENCY * (1 - control_waste))
    return {
        "waste_channel_pct": 100 * (t_waste_only - control_T) / control_T,
        "mix_channel_pct": 100 * (t_mix_only - control_T) / control_T,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--out", type=Path, default=EVIDENCE / "stage6_analysis.json")
    args = parser.parse_args()

    sleeps = load_sleeps()
    tiers = load_tiers()
    tier_sleep = {
        t: st.mean([s for c, s in sleeps.items() if tiers[c] == t]) for t in TIER_TRAIN
    }
    oracle = CoverageOracle.from_csv(COVERAGE_CSV)

    names = discover_run_dirs()
    runs: dict[str, Run] = {}
    for name in names:
        run = parse_and_replay(name, sleeps, tiers)
        runs[name] = run
        print(run.name)
        for line in validate(run):
            print(line)
        print()

    if args.validate:
        return 0

    task4 = json.loads(TASK4_STEPD.read_text())
    stage5b = json.loads(STAGE5B_ANALYSIS.read_text())
    control_none = task4["baseline|coverage_gated"]
    control_vmax_distance = stage5b["baseline_tier_spec_vmax_distance"]

    control_T = control_none["conservation"]["measured"]
    control_waste = control_none["waste"]["overall"]["wasted_frac"]
    control_rounds = control_none["conservation"]["rounds"]
    control_mean_sleep = control_T * CONCURRENCY * (1 - control_waste) / control_rounds

    payload = {
        "_reference": {
            "baseline_none": {
                "T": control_T, "waste": control_waste,
                "xavier_share": control_none["dispatch_share"]["overall"]["xavier"],
            },
            "baseline_tier_spec_vmax_distance": {
                "T": control_vmax_distance["conservation"]["measured"],
                "waste": control_vmax_distance["waste"]["overall"]["wasted_frac"],
                "xavier_share": control_vmax_distance["dispatch_share"]["xavier"],
            },
        },
    }

    for name, run in runs.items():
        waste = waste_table(run)
        share = dispatch_share(run)
        payload[name] = {
            "conservation": conservation_check(run, tier_sleep),
            "waste": waste,
            "dispatch_share": share,
            "curve": curve_stats(run),
            "clock_at": {str(c): clock_at(run, c) for c in CHECKPOINTS},
            "stage0_fallback": stage0_fallback_accounting(run),
            "stage0_pool_sizes_min_median": stage0_pool_sizes_min_median(run),
            "stage0_confusion": stage0_confusion(run, sleeps, tiers, oracle),
            "decomposition_vs_baseline_none": decompose(
                control_rounds, control_waste, control_mean_sleep, control_T,
                waste["overall"]["wasted_frac"], share, tier_sleep,
            ),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
