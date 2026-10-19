"""
Stage 5 -- analysis of the six Stage 0 filter runs
(prompts/STAGE5_stage0_filter.md, Steps 3-5).

A separate analyzer from analyze_stage4_task4.py on purpose, for the same
reason that file gives for being separate from make_sumo_configs.py: each
stage stays reproducible on its own, without risking the finished,
already-documented Task 4 analysis. Small constants (TIER_TRAIN, WINDOWS,
CHECKPOINTS, ...) are duplicated rather than imported for the same reason.
Task 4's own control numbers (baseline/gpu/cpu/observed, coverage_gated) are
read directly from its output, `task4_stepD.json`, rather than re-derived --
they are already validated and Step 2 forbids re-running those controls.

Ground truth for the filter's confusion matrix (Table F) is reconstructed the
same way Task 4's Step D.4 counterfactual was: from CoverageOracle plus each
client's TRUE (already jittered) sleep time in sleep_times.csv -- NOT from
what actually happened in this run, because a candidate Stage 0 rejects is
never dispatched, so its live "would it have uploaded" is not directly
observable. Given the round's own dispatch time (logged) and the client's
fixed true sleep time (same file every arm's sleep_simulation reads from),
the counterfactual finish time and hence the verdict is fully determined --
this is exactly Task 4's own method, applied to Stage 5's live rejection
events instead of Task 4's full unfiltered dispatch log.

Usage:
    python3 analyze_stage5.py            # all six runs
    python3 analyze_stage5.py --validate # parse + sanity checks only
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
EVIDENCE = GPU_TIERS_DATASET / "stage5_evidence"
TASK4_STEPD = GPU_TIERS_DATASET / "stage4_evidence" / "task4_stepD.json"

sys.path.insert(0, str(HERE))
from sumo_mobility import CoverageOracle  # noqa: E402

CELL = "n20r400"
CONCURRENCY = 4
TOTAL_ROUNDS = 4000
SLEEP_CSV = HERE / "output" / f"client_sleep_times_{CELL}.csv"
COVERAGE_CSV = HERE / "input" / f"coverage_{CELL}.csv"
CLIENTS_CSV = HERE / "input" / f"clients_{CELL}.csv"

# Same table analyze_stage4_task4.py uses -- the tier-mean prediction at
# anchor_sleep=15, noise-free.
TIER_TRAIN = {"xavier": 15.0, "orin": 1.923077, "thor": 1.0}
FLEET_MEAN_T_HAT_S = 8.323077

WINDOWS = ((1, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000))
CHECKPOINTS = (100, 250, 500, 1000, 2000, 4000)
THRESHOLDS = (70, 75, 78, 80)
PLATEAU_FRACTION = 0.10
SMOOTH_WINDOW = 11
NOISE_WINDOW = 101
EVAL_INTERVAL = 10

# run_name -> (selector arm, stage0_t_hat, Task 4 control key in task4_stepD.json)
RUN_META: dict[str, tuple[str, str, str]] = {
    "baseline_fleet_mean": ("baseline", "fleet_mean", "baseline|coverage_gated"),
    "baseline_tier_spec": ("baseline", "tier_spec", "baseline|coverage_gated"),
    "baseline_oracle": ("baseline", "oracle", "baseline|coverage_gated"),
    "gpu_tier_spec": ("gpu", "tier_spec", "gpu|coverage_gated"),
    "cpu_tier_spec": ("cpu", "tier_spec", "cpu|coverage_gated"),
    "observed_fleet_mean": ("observed", "fleet_mean", "observed|coverage_gated"),
}
RUN_NAMES = tuple(RUN_META)

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
IDLE_LINE = re.compile(
    r"\[SumoRsu\] No vehicle in range at t=([\d.]+) s; idling ([\d.]+) s"
)
EMPTY_ROUND_LINE = re.compile(r"\[SumoRsu\] Round (\d+) produced no aggregation")
SUMO_RANGE_LINE = re.compile(
    r"\[SumoRsu\] round=(-?\d+) t=([\d.]+) s: (\d+) of (\d+) client\(s\) in RSU range"
)
STAGE0_ROUND_LINE = re.compile(
    r"\[Stage0\] round=(-?\d+) t_hat_source=(\S+) eligible_after_range_gate=(\d+) "
    r"eligible_after_stage0=(\d+) stage0_rejected_ids=(\[[^\]]*\]) stage0_fallback=([01])"
)
STAGE0_SELECT_LINE = re.compile(
    r"\[Stage0\] round=(-?\d+) selected_id=(\d+) selected_tier=(\S+) "
    r"t_hat_used=(\S+) R_i_selected=(\S+)"
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

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
    server_staleness: int | None = None
    t_hat_used: float | None = None
    r_i_selected: float | None = None


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
    arm: str
    t_hat_source: str
    dispatches: list[Dispatch] = field(default_factory=list)
    merges: list[tuple[int, int]] = field(default_factory=list)
    idles: list[tuple[float, float]] = field(default_factory=list)
    empty_rounds: int = 0
    n_report_lines: int = 0
    curve: list[dict] = field(default_factory=list)
    clock: dict[int, float] = field(default_factory=dict)
    round_stats: dict[int, RoundStats] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_sleeps() -> dict[int, float]:
    with SLEEP_CSV.open(newline="") as fh:
        return {int(r["client_id"]): float(r["sleep_time"]) for r in csv.DictReader(fh)}


def load_tiers() -> dict[int, str]:
    with CLIENTS_CSV.open(newline="") as fh:
        return {int(r["client_id"]): r["tier_id"] for r in csv.DictReader(fh)}


def run_dir(name: str) -> Path:
    path = RUNS / f"stage5_{name}"
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


def load_task4_controls() -> dict:
    return json.loads(TASK4_STEPD.read_text())


# ---------------------------------------------------------------------------
# Parsing and replay
# ---------------------------------------------------------------------------

def parse_and_replay(name: str, sleeps: dict[int, float], tiers: dict[int, str]) -> Run:
    """One pass over the server log, mirroring analyze_stage4_task4.parse_and_
    replay's dispatch/merge/clock reconstruction, plus Stage 0's own per-round
    and per-selection lines (parsed in a second, cheap pass since a selection
    line can appear between a round's two Stage0 lines -- see module notes).
    """
    arm, t_hat_source, _ = RUN_META[name]
    run = Run(name=name, arm=arm, t_hat_source=t_hat_source)
    directory = run_dir(name)
    log_path = server_log_path(directory)

    # Pass 1: everything Task 4's parser reads, plus round-level pool sizes.
    slots = [0.0] * CONCURRENCY
    pending_report: dict[int, list[Dispatch]] = defaultdict(list)
    pending_merge: dict[int, list[Dispatch]] = defaultdict(list)
    current_round = 0
    stage0_select: dict[tuple[int, int], tuple[float | None, float | None]] = {}

    with log_path.open(errors="replace") as fh:
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
                rs.eligible_after_stage0 = int(match.group(4))
                rs.stage0_rejected_ids = ast.literal_eval(match.group(5))
                rs.stage0_fallback = match.group(6) == "1"
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
                if cid not in sleeps:
                    raise SystemExit(f"{name}: client {cid} not in the sleep table")
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
                    run.problems.append(
                        f"report for client {cid} with no unconsumed dispatch "
                        f"(report #{run.n_report_lines})"
                    )
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
                run.merges.append((effective_round, cid))
                queue = pending_merge[cid]
                if not queue:
                    run.problems.append(
                        f"round {effective_round} aggregated client {cid} with "
                        f"no uploaded, unmerged dispatch"
                    )
                    continue
                item = queue.pop(0)
                item.merge_round = effective_round
                item.server_staleness = int(match.group(3))
                continue

            match = IDLE_LINE.search(line)
            if match:
                run.idles.append((float(match.group(1)), float(match.group(2))))
                continue

            if EMPTY_ROUND_LINE.search(line):
                run.empty_rounds += 1

    for item in run.dispatches:
        key = (item.round, item.client_id)
        if key in stage0_select:
            item.t_hat_used, item.r_i_selected = stage0_select[key]

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
                "round": rnd,
                "accuracy": float(row["accuracy"]),
                "sim_time_plato": float(row["elapsed_time"]),
                "sim_time_reconstructed": clock[rnd],
            })
    return run


def validate(run: Run) -> list[str]:
    notes = []
    problems = list(run.problems)
    n_disp = len(run.dispatches)
    unreported = [d for d in run.dispatches if not d.reported]
    notes.append(f"dispatches={n_disp}  report lines={run.n_report_lines}  "
                 f"unreported at exit={len(unreported)}")
    with_t_hat = sum(1 for d in run.dispatches if d.t_hat_used is not None)
    notes.append(f"dispatches with a Stage0 t_hat_used match: {with_t_hat}/{n_disp}")
    if run.curve:
        last = run.curve[-1]
        gap = last["sim_time_plato"] - last["sim_time_reconstructed"]
        notes.append(
            f"round {last['round']}: plato={last['sim_time_plato']:.1f}s  "
            f"reconstructed={last['sim_time_reconstructed']:.1f}s  gap={gap:+.1f}s"
        )
    notes.append(f"idle events={len(run.idles)}  empty rounds={run.empty_rounds}  "
                 f"round_stats captured={len(run.round_stats)}")
    return [f"  {n}" for n in notes] + [f"  ** {p}" for p in problems]


# ---------------------------------------------------------------------------
# Metrics -- waste/share/rate/conservation/curve mirror analyze_stage4_task4.py
# exactly (same Dispatch field names, same formulas); stage0_* are new.
# ---------------------------------------------------------------------------

def in_window(rnd: int, lo: int, hi: int) -> bool:
    return (lo <= rnd <= hi) if lo == 1 else (lo < rnd <= hi)


def waste_table(run: Run) -> dict:
    reported = [d for d in run.dispatches if d.reported]
    out = {"overall": {}, "per_tier": {}, "per_window": {}, "per_tier_window": {}}

    def tally(items):
        n = len(items)
        wasted = sum(1 for d in items if not d.uploaded)
        return {
            "dispatches": n, "wasted": wasted, "useful": n - wasted,
            "wasted_frac": wasted / n if n else 0.0,
        }

    out["overall"] = tally(reported)
    for tier in TIER_TRAIN:
        out["per_tier"][tier] = tally([d for d in reported if d.tier == tier])
    for lo, hi in WINDOWS:
        items = [d for d in reported if in_window(d.round, lo, hi)]
        out["per_window"][f"{lo}-{hi}"] = tally(items)
        for tier in TIER_TRAIN:
            out["per_tier_window"][f"{tier}|{lo}-{hi}"] = tally(
                [d for d in items if d.tier == tier]
            )
    return out


def dispatch_share(run: Run) -> dict:
    reported = [d for d in run.dispatches if d.reported]
    out = {}
    overall_counts = defaultdict(int)
    for d in reported:
        overall_counts[d.tier] += 1
    out["overall"] = (
        {t: overall_counts[t] / len(reported) for t in TIER_TRAIN} if reported else {}
    )
    for lo, hi in WINDOWS:
        items = [d for d in reported if in_window(d.round, lo, hi)]
        counts = defaultdict(int)
        for d in items:
            counts[d.tier] += 1
        n = len(items)
        out[f"{lo}-{hi}"] = {t: counts[t] / n for t in TIER_TRAIN} if n else {}
    return out


def clock_at(run: Run, rnd: int) -> float | None:
    if rnd in run.clock:
        return run.clock[rnd]
    earlier = [r for r in run.clock if r <= rnd]
    return run.clock[max(earlier)] if earlier else None


def useful_work_rate(run: Run) -> dict:
    reported = [d for d in run.dispatches if d.reported]
    out = {}
    for lo, hi in WINDOWS:
        items = [d for d in reported if in_window(d.round, lo, hi)]
        useful = sum(1 for d in items if d.uploaded)
        t_lo = clock_at(run, lo if lo > 1 else 1)
        t_hi = clock_at(run, hi)
        if t_lo is None or t_hi is None or t_hi <= t_lo:
            out[f"{lo}-{hi}"] = None
            continue
        seconds = t_hi - t_lo
        out[f"{lo}-{hi}"] = {"useful": useful, "seconds": seconds, "rate": useful / seconds}
    return out


def conservation_check(run: Run, tier_sleep: dict[str, float]) -> dict:
    """T = rounds * mean_sleep(mix) / (C * (1 - waste)) -- Task 4's law,
    checked against these six runs' own measured mix and waste."""
    reported = [d for d in run.dispatches if d.reported]
    if not run.clock:
        return {"rounds": None, "measured": None, "predicted": None, "rel_error": None}
    rounds = max(run.clock)
    measured = run.clock[rounds]
    share = {t: sum(1 for d in reported if d.tier == t) / len(reported)
             for t in TIER_TRAIN}
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
        "evaluations": len(acc), "plateau": st.median(tail), "best": max(acc),
        "noise_sd": st.stdev(residuals) if len(residuals) > 1 else None,
    }
    for target in THRESHOLDS:
        hit = next((r for r, v in zip(rows, smoothed) if v >= target / 100.0), None)
        stats[f"rounds_to_{target}"] = hit["round"] if hit else None
        stats[f"tta_h_{target}"] = hit["sim_time_reconstructed"] / 3600.0 if hit else None
    return stats


# -- Stage 0 specific ---------------------------------------------------

def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    return values[f] + (values[c] - values[f]) * (k - f)


def stage0_pool_sizes(run: Run) -> dict:
    """median/p10/p90 of pool size after each gate, per window (Step 3)."""
    out = {}
    for lo, hi in WINDOWS:
        rs_list = [rs for r, rs in run.round_stats.items() if in_window(r, lo, hi)]

        def stats(attr):
            vals = [getattr(rs, attr) for rs in rs_list if getattr(rs, attr) is not None]
            return {
                "median": st.median(vals) if vals else None,
                "p10": _percentile(vals, 0.10), "p90": _percentile(vals, 0.90),
                "n": len(vals),
            }

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
        per_window[f"{lo}-{hi}"] = {
            "rounds": n, "fallback": fb, "fallback_frac": fb / n if n else None,
        }
    return {
        "rounds": total_rounds, "fallback": total_fallback,
        "fallback_frac": total_fallback / total_rounds if total_rounds else None,
        "per_window": per_window,
    }


def redispatch_gap(run: Run) -> dict:
    """Cycles until the same client is dispatched again, after a wasted
    dispatch versus after a useful one (Task 4 Step D's feedback-loop check,
    STATUS.md Sec.5.4 -- a copy of analyze_stage4_task4.redispatch_gap, which
    operates only on fields this Run shares with Task 4's)."""
    reported = [d for d in run.dispatches if d.reported]
    positions: dict[int, list[int]] = defaultdict(list)
    for i, d in enumerate(reported):
        positions[d.client_id].append(i)
    after_waste, after_useful = [], []
    for idxs in positions.values():
        for a, b in zip(idxs, idxs[1:]):
            (after_waste if not reported[a].raw_in_range else after_useful).append(b - a)
    return {
        "after_waste": st.mean(after_waste) if after_waste else None,
        "after_useful": st.mean(after_useful) if after_useful else None,
        "ratio": (st.mean(after_waste) / st.mean(after_useful)
                  if after_waste and after_useful else None),
        "n_waste": len(after_waste),
    }


def crossover_round(run: Run, baseline: Run, hold: int = 200) -> int | None:
    """First round after which `run` is slower than `baseline` and stays
    slower for `hold` rounds (copy of analyze_stage4_task4.crossover_round).
    Used here on (gpu_tier_spec, baseline_tier_spec) etc. -- pairs that share
    no control in task4_stepD.json, since both are Stage 5's own runs."""
    shared = sorted(r for r in run.clock if r in baseline.clock)
    for i, rnd in enumerate(shared):
        if rnd < 50 or run.clock[rnd] <= baseline.clock[rnd]:
            continue
        if all(run.clock[q] > baseline.clock[q]
               for q in shared[i:] if q <= rnd + hold):
            return rnd
    return None


def stage0_confusion_by_window(run: Run, sleeps: dict[int, float], tiers: dict[int, str],
                               oracle: CoverageOracle) -> dict:
    """stage0_confusion(), sliced by round window (Table F)."""
    out = {}
    for lo, hi in WINDOWS:
        per_tier: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for rs in run.round_stats.values():
            if rs.t is None or not in_window(rs.round, lo, hi):
                continue
            for cid in rs.stage0_rejected_ids:
                tier = tiers.get(cid, "unknown")
                end = oracle.window_end(cid, rs.t)
                would_upload = end is not None and end >= rs.t + sleeps[cid]
                per_tier[tier]["killed_viable" if would_upload else "killed_waste"] += 1
        for d in run.dispatches:
            if not d.reported or not in_window(d.round, lo, hi):
                continue
            per_tier[d.tier]["passed_useful" if d.raw_in_range else "missed_waste"] += 1
        out[f"{lo}-{hi}"] = {t: dict(v) for t, v in per_tier.items()}
    return out


def stage0_confusion(run: Run, sleeps: dict[int, float], tiers: dict[int, str],
                     oracle: CoverageOracle) -> dict:
    """Filter quality against ground truth, per tier (Step 3).

    killed_waste / killed_viable: reconstructed for every REJECTED candidate
    from a counterfactual dispatch at the round's own t, using the client's
    TRUE sleep time -- see the module docstring for why this is the only
    honest way to score a live (not counterfactual-replayed) run.
    missed_waste / passed_useful: read directly from the dispatches that WERE
    admitted (selected), where ground truth is directly observed.
    """
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--out", type=Path, default=EVIDENCE / "stage5_analysis.json")
    args = parser.parse_args()

    sleeps = load_sleeps()
    tiers = load_tiers()
    oracle = CoverageOracle.from_csv(COVERAGE_CSV)
    controls = load_task4_controls()

    runs: dict[str, Run] = {}
    print(f"trace coverage spans t in [{oracle.t_start:.1f}, {oracle.t_end:.1f}] s\n")
    for name in RUN_NAMES:
        run = parse_and_replay(name, sleeps, tiers)
        runs[name] = run
        print(run.name)
        for line in validate(run):
            print(line)
        print()

    if args.validate:
        return 0

    tier_sleep = {
        t: st.mean([s for c, s in sleeps.items() if tiers[c] == t])
        for t in TIER_TRAIN
    }

    payload = {}
    for name, run in runs.items():
        _, t_hat_source, control_key = RUN_META[name]
        payload[name] = {
            "arm": run.arm,
            "t_hat_source": t_hat_source,
            "control_key": control_key,
            "control": controls[control_key],
            "conservation": conservation_check(run, tier_sleep),
            "waste": waste_table(run),
            "dispatch_share": dispatch_share(run),
            "useful_work_rate": useful_work_rate(run),
            "curve": curve_stats(run),
            "clock_at": {str(c): clock_at(run, c) for c in CHECKPOINTS},
            "redispatch_gap": redispatch_gap(run),
            "stage0_pool_sizes": stage0_pool_sizes(run),
            "stage0_fallback": stage0_fallback_accounting(run),
            "stage0_confusion": stage0_confusion(run, sleeps, tiers, oracle),
            "stage0_confusion_by_window": stage0_confusion_by_window(run, sleeps, tiers, oracle),
            "dispatch_cycles": sum(1 for d in run.dispatches if d.reported),
            "idle_events": len(run.idles),
            "empty_rounds": run.empty_rounds,
            "rounds_done": run.curve[-1]["round"] if run.curve else None,
        }

    # Crossover rounds between pairs of Stage 5's OWN runs -- neither has a
    # shared Task 4 control, so this is computed here rather than pulled from
    # task4_stepD.json. hold=200 matches Task 4 Step D's own choice.
    crossover_pairs = [
        ("gpu_tier_spec", "baseline_tier_spec"),
        ("cpu_tier_spec", "baseline_tier_spec"),
        ("cpu_tier_spec", "gpu_tier_spec"),
        ("observed_fleet_mean", "baseline_fleet_mean"),
    ]
    payload["_crossover"] = {
        f"{a}_vs_{b}": crossover_round(runs[a], runs[b]) for a, b in crossover_pairs
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
