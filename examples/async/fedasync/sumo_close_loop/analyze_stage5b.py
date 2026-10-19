"""
Stage 5b -- analysis of the two R_i-fix runs (prompts/STAGE5B_fix_Ri.md,
Step 3).

Self-contained for the same reason analyze_stage5.py is (see its own module
docstring): each stage stays reproducible without risking a previous stage's
finished output. The two reference rows -- Task 4's `baseline/none` and
Stage 5's `baseline_tier_spec` (the `vmax_elapsed` control) -- are read
directly from `task4_stepD.json` / `stage5_analysis.json` rather than
re-derived, since Step 2 forbids re-running either.

Usage:
    python3 analyze_stage5b.py            # both runs
    python3 analyze_stage5b.py --validate # parse + sanity checks only
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
EVIDENCE = GPU_TIERS_DATASET / "stage5b_evidence"
TASK4_STEPD = GPU_TIERS_DATASET / "stage4_evidence" / "task4_stepD.json"
STAGE5_ANALYSIS = GPU_TIERS_DATASET / "stage5_evidence" / "stage5_analysis.json"

sys.path.insert(0, str(HERE))
from sumo_mobility import CoverageOracle  # noqa: E402
from sumo_position import PositionOracle  # noqa: E402

CELL = "n20r400"
CONCURRENCY = 4
SLEEP_CSV = HERE / "output" / f"client_sleep_times_{CELL}.csv"
COVERAGE_CSV = HERE / "input" / f"coverage_{CELL}.csv"
CLIENTS_CSV = HERE / "input" / f"clients_{CELL}.csv"
POSITIONS_CSV = HERE / "input" / f"positions_{CELL}.csv"

TIER_TRAIN = {"xavier": 15.0, "orin": 1.923077, "thor": 1.0}
RSU_X_M = 600.0
RADIUS_M = 400.0
V_MAX_MPS = 19.444444
CROSSING_S = 2.0 * RADIUS_M / V_MAX_MPS

WINDOWS = ((1, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000))
CHECKPOINTS = (100, 250, 500, 1000, 2000, 4000)
THRESHOLDS = (70, 75, 78, 80)
PLATEAU_FRACTION = 0.10
SMOOTH_WINDOW = 11
NOISE_WINDOW = 101
EVAL_INTERVAL = 10

RUN_NAMES = ("baseline_tier_spec_vmax_distance", "baseline_tier_spec_oracle_exit")
R_I_MODE_OF = {
    "baseline_tier_spec_vmax_distance": "vmax_distance",
    "baseline_tier_spec_oracle_exit": "oracle_exit",
}

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
STAGE0_SELECT_LINE = re.compile(
    r"\[Stage0\] round=(-?\d+) selected_id=(\d+) selected_tier=(\S+) "
    r"t_hat_used=(\S+) R_i_selected=(\S+)"
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
    log_dispatch_t: float = 0.0
    log_finish_t: float = 0.0
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
    r_i_mode: str
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


def run_dir(name: str) -> Path:
    path = RUNS / f"stage5b_{name}"
    if not path.exists():
        raise SystemExit(f"missing run directory {path}")
    return path


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
    run = Run(name=name, r_i_mode=R_I_MODE_OF[name])
    directory = run_dir(name)
    slots = [0.0] * CONCURRENCY
    pending_report: dict[int, list[Dispatch]] = defaultdict(list)
    pending_merge: dict[int, list[Dispatch]] = defaultdict(list)
    current_round = 0
    stage0_select: dict[tuple[int, int], tuple[float | None, float | None]] = {}

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

            match = STAGE0_SELECT_LINE.search(line)
            if match:
                rnd, cid = int(match.group(1)), int(match.group(2))
                t_hat = None if match.group(4) == "nan" else float(match.group(4))
                r_i = None if match.group(5) == "nan" else float(match.group(5))
                stage0_select[(rnd, cid)] = (t_hat, r_i)
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
                item.log_dispatch_t = float(match.group(4))
                item.log_finish_t = float(match.group(5))
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

    for item in run.dispatches:
        key = (item.round, item.client_id)
        if key in stage0_select:
            pass  # t_hat/R_i are logged but not needed for Stage 5b's tables

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

    out = {"overall": tally(reported), "per_tier": {}, "per_window": {}}
    for tier in TIER_TRAIN:
        out["per_tier"][tier] = tally([d for d in reported if d.tier == tier])
    for lo, hi in WINDOWS:
        out["per_window"][f"{lo}-{hi}"] = tally(
            [d for d in reported if in_window(d.round, lo, hi)]
        )
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


def useful_work_rate_last_window(run: Run) -> dict | None:
    lo, hi = WINDOWS[-1]
    reported = [d for d in run.dispatches if d.reported]
    items = [d for d in reported if in_window(d.round, lo, hi)]
    useful = sum(1 for d in items if d.uploaded)
    t_lo, t_hi = clock_at(run, lo), clock_at(run, hi)
    if t_lo is None or t_hi is None or t_hi <= t_lo:
        return None
    seconds = t_hi - t_lo
    return {"useful": useful, "seconds": seconds, "rate": useful / seconds}


def conservation_check(run: Run, tier_sleep: dict[str, float]) -> dict:
    """T = rounds * mean_sleep(mix) / (C * (1 - waste)).

    mean_sleep MUST use the fleet's actual (jittered) per-tier mean training
    time -- tier_sleep, from sleep_times.csv -- not TIER_TRAIN's noise-free
    anchor values. TIER_TRAIN is what the PREDICTOR (T_hat, the Stage 0
    belief) uses; this law is about actual wall-clock seconds consumed, which
    is the SIMULATOR side and includes jitter. Conflating the two understates
    mean_sleep and inflates the predicted T (caught during Stage 5b's own
    review: using TIER_TRAIN here gave a ~5% conservation error against the
    ~1% Stage 5 measured, immediately suspicious against Stage 5's own
    precedent of using the empirical mean in analyze_stage5.py).
    """
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


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    return values[f] + (values[c] - values[f]) * (k - f)


def stage0_pool_sizes(run: Run) -> dict:
    out = {}
    for lo, hi in WINDOWS:
        rs_list = [rs for r, rs in run.round_stats.items() if in_window(r, lo, hi)]

        def stats(attr):
            vals = [getattr(rs, attr) for rs in rs_list if getattr(rs, attr) is not None]
            return {"median": st.median(vals) if vals else None,
                    "p10": _percentile(vals, 0.10), "p90": _percentile(vals, 0.90)}

        out[f"{lo}-{hi}"] = {
            "eligible_after_busy": stats("eligible_after_busy"),
            "eligible_after_range_gate": stats("eligible_after_range_gate"),
            "eligible_after_stage0": stats("eligible_after_stage0"),
        }
    return out


def stage0_fallback_accounting(run: Run) -> dict:
    total_rounds = len(run.round_stats)
    total_fallback = sum(1 for rs in run.round_stats.values() if rs.stage0_fallback)
    per_window = {}
    for lo, hi in WINDOWS:
        rs_list = [rs for r, rs in run.round_stats.items() if in_window(r, lo, hi)]
        n = len(rs_list)
        fb = sum(1 for rs in rs_list if rs.stage0_fallback)
        per_window[f"{lo}-{hi}"] = {"rounds": n, "fallback_frac": fb / n if n else None}
    return {
        "rounds": total_rounds, "fallback": total_fallback,
        "fallback_frac": total_fallback / total_rounds if total_rounds else None,
        "per_window": per_window,
    }


def stage0_confusion(run: Run, sleeps: dict[int, float], tiers: dict[int, str],
                     oracle: CoverageOracle) -> dict:
    """Same method as analyze_stage5.stage0_confusion -- see its docstring."""
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


def fairness(run: Run, tiers: dict[int, str]) -> dict:
    """Table 5: dispatches per client per tier, thor:xavier ratio, Jain's
    fairness index over the 20 clients."""
    reported = [d for d in run.dispatches if d.reported]
    counts = defaultdict(int)
    for d in reported:
        counts[d.client_id] += 1
    for cid in tiers:
        counts.setdefault(cid, 0)

    by_tier = defaultdict(list)
    for cid, tier in tiers.items():
        by_tier[tier].append(counts[cid])

    xavier_mean = st.mean(by_tier["xavier"]) if by_tier["xavier"] else 0.0
    thor_mean = st.mean(by_tier["thor"]) if by_tier["thor"] else 0.0

    n = list(counts.values())
    jain = (sum(n) ** 2) / (len(n) * sum(x * x for x in n)) if any(n) else None

    return {
        "per_client": dict(counts),
        "mean_per_tier": {t: st.mean(v) if v else 0.0 for t, v in by_tier.items()},
        "thor_to_xavier_ratio": thor_mean / xavier_mean if xavier_mean else None,
        "jain_index": jain,
    }


# ---------------------------------------------------------------------------
# Table 4 -- eligibility window (offline, formula-only; independent of a
# specific run's dispatch history -- only depends on the fixed trace, the
# R_i formula, and stage0_t_hat's tier_spec belief, which is the same in
# both Stage 5b runs).
# ---------------------------------------------------------------------------

def eligibility_window(sleeps_ignored, oracle_windows, positions: PositionOracle) -> dict:
    """Mean fraction of each coverage window a vehicle of that tier passes
    T_hat_i <= R_i, for each of the three R_i formulas."""
    sums: dict[str, dict[str, float]] = {
        mode: defaultdict(float) for mode in ("vmax_elapsed", "vmax_distance", "oracle_exit")
    }
    counts: dict[str, dict[str, int]] = {
        mode: defaultdict(int) for mode in sums
    }
    tiers = load_tiers()

    for client_id, enter, exit_ in oracle_windows:
        tier = tiers[client_id]
        t_hat = TIER_TRAIN[tier]
        duration = exit_ - enter
        if duration <= 0:
            continue
        eligible_seconds = {mode: 0.0 for mode in sums}
        t = enter
        while t <= exit_ + 1e-9:
            r_elapsed = enter + CROSSING_S - t
            r_oracle = exit_ - t
            x = positions.x_at(client_id, t)
            direction = positions.direction_at(client_id, t)
            r_distance = None
            if x is not None and direction is not None:
                x_exit = RSU_X_M + RADIUS_M if direction > 0 else RSU_X_M - RADIUS_M
                r_distance = abs(x_exit - x) / V_MAX_MPS

            if r_elapsed >= t_hat:
                eligible_seconds["vmax_elapsed"] += 1.0
            if r_oracle >= t_hat:
                eligible_seconds["oracle_exit"] += 1.0
            if r_distance is not None and r_distance >= t_hat:
                eligible_seconds["vmax_distance"] += 1.0
            t += 1.0

        for mode in sums:
            sums[mode][tier] += eligible_seconds[mode] / duration
            counts[mode][tier] += 1

    return {
        mode: {tier: sums[mode][tier] / counts[mode][tier] for tier in TIER_TRAIN}
        for mode in sums
    }


# ---------------------------------------------------------------------------
# Table 2 -- decomposition (waste channel / mix channel), via the
# conservation law T = rounds * mean_sleep(mix) / (C * (1 - waste)).
# ---------------------------------------------------------------------------

def decompose(rounds: int, control_waste: float, control_mean_sleep: float,
             control_T: float, run_waste: float, run_share: dict,
             tier_sleep: dict[str, float]) -> dict:
    run_mean_sleep = sum(run_share.get(t, 0.0) * tier_sleep[t] for t in TIER_TRAIN)
    t_waste_only = rounds * control_mean_sleep / (CONCURRENCY * (1 - run_waste))
    t_mix_only = rounds * run_mean_sleep / (CONCURRENCY * (1 - control_waste))
    return {
        "waste_channel_pct": 100 * (t_waste_only - control_T) / control_T,
        "mix_channel_pct": 100 * (t_mix_only - control_T) / control_T,
        "run_mean_sleep": run_mean_sleep,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--out", type=Path, default=EVIDENCE / "stage5b_analysis.json")
    args = parser.parse_args()

    sleeps = load_sleeps()
    tiers = load_tiers()
    tier_sleep = {
        t: st.mean([s for c, s in sleeps.items() if tiers[c] == t]) for t in TIER_TRAIN
    }
    oracle = CoverageOracle.from_csv(COVERAGE_CSV)
    stage5 = json.loads(STAGE5_ANALYSIS.read_text())
    task4 = json.loads(TASK4_STEPD.read_text())

    runs: dict[str, Run] = {}
    for name in RUN_NAMES:
        run = parse_and_replay(name, sleeps, tiers)
        runs[name] = run
        print(run.name)
        for line in validate(run):
            print(line)
        print()

    if args.validate:
        return 0

    positions = PositionOracle.from_csv(POSITIONS_CSV)
    with COVERAGE_CSV.open(newline="") as f:
        oracle_windows = [
            (int(r["client_id"]), float(r["t_enter_s"]), float(r["t_exit_s"]))
            for r in csv.DictReader(f)
        ]

    control_none = task4["baseline|coverage_gated"]
    control_vmax_elapsed = stage5["baseline_tier_spec"]

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
            "baseline_tier_spec_vmax_elapsed": {
                "T": control_vmax_elapsed["conservation"]["measured"],
                "waste": control_vmax_elapsed["waste"]["overall"]["wasted_frac"],
                "xavier_share": control_vmax_elapsed["dispatch_share"]["overall"]["xavier"],
            },
        },
        "_eligibility_window": eligibility_window(sleeps, oracle_windows, positions),
    }

    for name, run in runs.items():
        waste = waste_table(run)
        share = dispatch_share(run)
        conservation = conservation_check(run, tier_sleep)
        payload[name] = {
            "r_i_mode": run.r_i_mode,
            "conservation": conservation,
            "waste": waste,
            "dispatch_share": share,
            "useful_work_rate_last_window": useful_work_rate_last_window(run),
            "curve": curve_stats(run),
            "clock_at": {str(c): clock_at(run, c) for c in CHECKPOINTS},
            "stage0_pool_sizes": stage0_pool_sizes(run),
            "stage0_fallback": stage0_fallback_accounting(run),
            "stage0_confusion": stage0_confusion(run, sleeps, tiers, oracle),
            "fairness": fairness(run, tiers),
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
