"""
Stage 4 Task 4, Step D — analysis of the eight coverage-gating runs.

Reads each run's own server log and results CSV and produces the four things
Step D asks for, plus the Stage 3 summary fields and the Step E predictions.

Why the clock is rebuilt rather than read
-----------------------------------------
PLATO's simulated clock carries `report.processing_time`, a `perf_counter`
delta measured on the host (`stage2_evidence/clock_reconstruction.md`). These
eight runs shared one GPU, and the inflation differs per run (739-853 s on the
two gpu arms alone), so PLATO's `elapsed_time` is not comparable across arms.
Step D asks for `sim_time_reconstructed`, and this rebuilds it with the same
C-slot queue replay Stage 2 established: the dispatch ORDER comes from the log,
the durations come from the fleet's sleep table, and nothing host-measured
enters the time axis.

The SUMO addition to that model is the waiting clock, which idles the server
forward when no dispatchable vehicle is in range. On the radius-400 trace it
fires twice across all eight runs (3.2 s and 3.7 s), so it is reported and
otherwise ignored -- see `validate()`.

Usage:
    python3 analyze_stage4_task4.py            # all eight runs
    python3 analyze_stage4_task4.py --validate # checks only, no report
"""

from __future__ import annotations

import argparse
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
LOGS = HERE / "logs"
EVIDENCE = FEDASYNC / "gpu_tiers_dataset" / "stage4_evidence"

sys.path.insert(0, str(HERE))
from sumo_mobility import CoverageOracle  # noqa: E402

ARMS = ("baseline", "gpu", "cpu", "observed")
POLICIES = ("coverage_gated", "always_accept")
CELL = "n20r400"

CONCURRENCY = 4
TOTAL_ROUNDS = 4000
SLEEP_CSV = HERE / "output" / f"client_sleep_times_{CELL}.csv"
COVERAGE_CSV = HERE / "input" / f"coverage_{CELL}.csv"
CLIENTS_CSV = HERE / "input" / f"clients_{CELL}.csv"

# Tier-mean training time at anchor_sleep = 15, the same table Step A's gate
# check used. The Stage 0 filter is a PREDICTOR, so it gets the tier belief,
# not the client's own jittered draw; the jittered variant is reported beside
# it as a sensitivity check.
TIER_TRAIN = {"xavier": 15.0, "orin": 1.923077, "thor": 1.0}

WINDOWS = ((1, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000))
CHECKPOINTS = (100, 250, 500, 1000, 2000, 4000)
THRESHOLDS = (70, 75, 78, 80)
PLATEAU_FRACTION = 0.10
SMOOTH_WINDOW = 11
NOISE_WINDOW = 101
USABLE_MARGIN = 0.05
EVAL_INTERVAL = 10

ROUND_LINE = re.compile(r"Starting round (\d+)/(\d+)")
SELECT_LINE = re.compile(r"Selecting client #(\d+) for training")
# The aggregation event. NOT "Adding client #N to the list of clients for
# aggregation": that is PLATO's generic pre-filter bookkeeping and it fires for
# updates the UploadGate goes on to discard (4,606 of them against 4,000 actual
# aggregations in the baseline coverage_gated run). This line is written by the
# FedAsync strategy at the point a weight update is really applied, and it
# carries the server's own staleness number.
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


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_sleeps() -> dict[int, float]:
    with SLEEP_CSV.open(newline="") as fh:
        return {int(r["client_id"]): float(r["sleep_time"]) for r in csv.DictReader(fh)}


def load_tiers() -> dict[int, str]:
    with CLIENTS_CSV.open(newline="") as fh:
        return {int(r["client_id"]): r["tier_id"] for r in csv.DictReader(fh)}


def run_dir(arm: str, policy: str) -> Path:
    path = RUNS / f"{arm}_{CELL}_{policy}"
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


# ---------------------------------------------------------------------------
# Parsing and replay
# ---------------------------------------------------------------------------

@dataclass
class Dispatch:
    index: int
    round: int          # the round this client was SELECTED at
    client_id: int
    tier: str
    start: float = 0.0  # reconstructed
    finish: float = 0.0  # reconstructed
    # filled from the matching report
    reported: bool = False
    uploaded: bool = False
    raw_in_range: bool = False
    log_dispatch_t: float = 0.0
    log_finish_t: float = 0.0
    merge_round: int | None = None
    server_staleness: int | None = None


@dataclass
class Run:
    arm: str
    policy: str
    dispatches: list[Dispatch] = field(default_factory=list)
    merges: list[tuple[int, int]] = field(default_factory=list)
    idles: list[tuple[float, float]] = field(default_factory=list)
    empty_rounds: int = 0
    n_report_lines: int = 0
    curve: list[dict] = field(default_factory=list)
    clock: dict[int, float] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.arm}_{CELL}_{self.policy}"


def parse_and_replay(arm: str, policy: str, sleeps: dict[int, float],
                     tiers: dict[int, str]) -> Run:
    """One pass over the server log: dispatch order, reports, merges.

    Reports and merges are matched to dispatches FIFO per client. That is
    unambiguous here because PLATO withholds a client from selection while it
    is training and while its report is waiting to be merged
    (`servers/base.py:567-585`), so a client can never have two dispatches in
    flight at once.
    """
    run = Run(arm=arm, policy=policy)
    directory = run_dir(arm, policy)

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

            match = SELECT_LINE.search(line)
            if match:
                cid = int(match.group(1))
                if cid not in sleeps:
                    raise SystemExit(f"{run.name}: client {cid} not in the sleep table")
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
                if match.group(8) != policy:
                    run.problems.append(
                        f"log says policy={match.group(8)}, expected {policy}"
                    )
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

    # Round -> reconstructed clock. A staleness flush can merge several clients
    # under one round number; the round's clock is the latest of them, which is
    # when the server actually got there.
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(run: Run, sleeps: dict[int, float], oracle: CoverageOracle) -> list[str]:
    notes = []
    problems = list(run.problems)

    n_disp = len(run.dispatches)
    unreported = [d for d in run.dispatches if not d.reported]
    notes.append(f"dispatches={n_disp}  report lines={run.n_report_lines}  "
                 f"unreported at exit={len(unreported)}")
    if run.n_report_lines + len(unreported) != n_disp:
        problems.append(
            f"{run.n_report_lines} reports + {len(unreported)} unreported "
            f"!= {n_disp} dispatches"
        )

    uploaded = [d for d in run.dispatches if d.reported and d.uploaded]
    merged = [d for d in run.dispatches if d.merge_round is not None]
    notes.append(f"uploaded={len(uploaded)}  aggregated={len(merged)}  "
                 f"aggregation lines={len(run.merges)}  "
                 f"rounds with a clock={len(run.clock)}")
    if len(merged) > len(uploaded):
        problems.append("more aggregations than uploads")
    stale = staleness(run)
    notes.append(f"staleness: mean={stale['mean']:.3f} max={stale['max']} "
                 f"(rebuilt mean={stale['rebuilt_mean']:.3f}, "
                 f"{stale['rebuilt_disagreements']} disagreements)")

    # The policy must behave as declared.
    if run.policy == "always_accept":
        bad = [d for d in run.dispatches if d.reported and not d.uploaded]
        if bad:
            problems.append(f"always_accept rejected {len(bad)} uploads")
    else:
        bad = [d for d in run.dispatches
               if d.reported and d.uploaded != d.raw_in_range]
        if bad:
            problems.append(
                f"coverage_gated: uploaded != raw_in_range on {len(bad)} reports"
            )

    # The coverage table must reproduce the run's own ground truth. This is the
    # check that the Stage 0 counterfactual below is reading the same mobility
    # the run read.
    mismatch = sum(
        1 for d in run.dispatches if d.reported
        and oracle.in_coverage(d.client_id, d.log_finish_t) != d.raw_in_range
    )
    notes.append(f"raw_in_range reproduced from coverage CSV: "
                 f"{run.n_report_lines - mismatch}/{run.n_report_lines}")
    if mismatch:
        problems.append(f"coverage CSV disagrees with the log on {mismatch} reports")

    # Merge order must be ascending in reconstructed time, or the queue model is
    # not the model the server ran (Stage 2's verify_merge_order).
    finishes = [d.finish for d in sorted(merged, key=lambda d: d.index)
                if d.merge_round is not None]
    ordered = sorted(merged, key=lambda d: (d.merge_round, d.index))
    out_of_order = sum(
        1 for a, b in zip(ordered, ordered[1:]) if b.finish < a.finish - 1e-9
    )
    notes.append(f"merges out of ascending reconstructed order: {out_of_order}")

    # PLATO's clock minus the rebuilt one is the accumulated host overhead. It
    # must be positive and must match the per-report inflation measured from the
    # log's own durations.
    if run.curve:
        last = run.curve[-1]
        gap = last["sim_time_plato"] - last["sim_time_reconstructed"]
        inflation = sum(
            (d.log_finish_t - d.log_dispatch_t) - sleeps[d.client_id]
            for d in run.dispatches if d.reported
        )
        notes.append(
            f"round {last['round']}: plato={last['sim_time_plato']:.1f}s  "
            f"reconstructed={last['sim_time_reconstructed']:.1f}s  "
            f"gap={gap:+.1f}s ({100 * gap / last['sim_time_reconstructed']:+.1f}%)  "
            f"log-measured host inflation={inflation:.1f}s"
        )
        if gap < 0:
            problems.append("reconstructed clock runs ahead of PLATO's")

    idle_total = sum(d for _, d in run.idles)
    notes.append(f"idle events={len(run.idles)} totalling {idle_total:.1f}s; "
                 f"empty rounds logged={run.empty_rounds}")

    wrapped = [d for d in run.dispatches if d.log_finish_t > oracle.t_end]
    if wrapped:
        problems.append(f"{len(wrapped)} reports past the end of the trace (wrap)")

    return [f"  {n}" for n in notes] + [f"  ** {p}" for p in problems]


# ---------------------------------------------------------------------------
# Step D metrics
# ---------------------------------------------------------------------------

def in_window(rnd: int, lo: int, hi: int) -> bool:
    return (lo <= rnd <= hi) if lo == 1 else (lo < rnd <= hi)


def waste_table(run: Run) -> dict:
    """Dispatch/waste counts overall, per tier and per window.

    Two waste columns, because under `always_accept` nothing is ever rejected:

      policy waste   -- the upload the server actually refused (uploaded == 0).
                        Zero by construction under always_accept.
      physical waste -- the vehicle was out of range when it finished
                        (raw_in_range == 0), measured identically under both
                        policies. Under always_accept this is the work that
                        WOULD have been lost, and it is the only column that
                        compares the two policies like for like.
    """
    reported = [d for d in run.dispatches if d.reported]
    out = {"overall": {}, "per_tier": {}, "per_window": {}, "per_tier_window": {}}

    def tally(items):
        n = len(items)
        policy_wasted = sum(1 for d in items if not d.uploaded)
        physical_wasted = sum(1 for d in items if not d.raw_in_range)
        return {
            "dispatches": n,
            "wasted": policy_wasted,
            "useful": n - policy_wasted,
            "wasted_frac": policy_wasted / n if n else 0.0,
            "physical_wasted": physical_wasted,
            "physical_wasted_frac": physical_wasted / n if n else 0.0,
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
    counts = defaultdict(int)
    for d in reported:
        counts[d.tier] += 1
    total = len(reported)
    out["overall"] = {t: counts[t] / total for t in TIER_TRAIN} if total else {}
    for lo, hi in WINDOWS:
        items = [d for d in reported if in_window(d.round, lo, hi)]
        counts = defaultdict(int)
        for d in items:
            counts[d.tier] += 1
        n = len(items)
        out[f"{lo}-{hi}"] = {t: counts[t] / n for t in TIER_TRAIN} if n else {}
    return out


def clock_at(run: Run, rnd: int) -> float | None:
    """Reconstructed seconds at a round, or the nearest earlier round's."""
    if rnd in run.clock:
        return run.clock[rnd]
    earlier = [r for r in run.clock if r <= rnd]
    return run.clock[max(earlier)] if earlier else None


def useful_work_rate(run: Run) -> dict:
    """Useful dispatches per reconstructed second, per window."""
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
        out[f"{lo}-{hi}"] = {
            "useful": useful,
            "seconds": seconds,
            "rate": useful / seconds,
            "dispatch_rate": len(items) / seconds,
        }
    return out


def stage0_counterfactual(run: Run, oracle: CoverageOracle,
                          sleeps: dict[int, float]) -> dict:
    """Replay the Stage 0 feasibility filter over this run's own dispatches.

    At dispatch time the filter asks whether the vehicle can still finish
    before it leaves range:

        R_i = window_end(i, t_dispatch) - t_dispatch  >=  T_hat_i

    `T_comm` is zero here -- `sumo_server.configure()` raises unless
    `comm_simulation = true`, so the harness guarantees it.

    Ground truth is the run's own `raw_in_range` at finish. The four cells:

        killed_waste   filter rejected it AND it would have been wasted   (good)
        killed_viable  filter rejected it BUT it would have uploaded      (cost)
        missed_waste   filter passed it AND it was wasted                 (leak)
        passed_useful  filter passed it AND it uploaded                   (good)

    The timestamps are the LOG's, not the rebuilt clock's: the run's own
    coverage verdicts were taken on that clock, so a counterfactual asked on a
    different one would be answering about a different trajectory.
    """
    reported = [d for d in run.dispatches if d.reported]
    out = {}
    for label, predict in (
        ("tier_mean", lambda d: TIER_TRAIN[d.tier]),
        ("oracle_sleep", lambda d: sleeps[d.client_id]),
    ):
        killed_waste = killed_viable = missed_waste = passed_useful = 0
        not_in_range_at_dispatch = 0
        for d in reported:
            end = oracle.window_end(d.client_id, d.log_dispatch_t)
            if end is None:
                # Dispatched while out of range: the selection gate should make
                # this impossible. Counted and reported rather than assumed away.
                not_in_range_at_dispatch += 1
                feasible = False
            else:
                feasible = (end - d.log_dispatch_t) >= predict(d)
            if feasible:
                if d.raw_in_range:
                    passed_useful += 1
                else:
                    missed_waste += 1
            else:
                if d.raw_in_range:
                    killed_viable += 1
                else:
                    killed_waste += 1
        total_waste = killed_waste + missed_waste
        total_viable = killed_viable + passed_useful
        out[label] = {
            "killed_waste": killed_waste,
            "killed_viable": killed_viable,
            "missed_waste": missed_waste,
            "passed_useful": passed_useful,
            "waste_caught_frac": killed_waste / total_waste if total_waste else None,
            "viable_killed_frac": killed_viable / total_viable if total_viable else None,
            "waste_remaining_frac": missed_waste / len(reported) if reported else None,
            "dispatched_out_of_range": not_in_range_at_dispatch,
        }
    return out


def staleness(run: Run) -> dict:
    """The server's own staleness per aggregation, cross-checked against
    `merge_round - dispatch_round` rebuilt from the dispatch sequence."""
    reported = [d for d in run.dispatches if d.merge_round is not None]
    server = [d.server_staleness for d in reported if d.server_staleness is not None]
    rebuilt = [d.merge_round - d.round for d in reported]
    disagree = sum(
        1 for d in reported
        if d.server_staleness is not None
        and d.server_staleness != d.merge_round - d.round
    )
    return {
        "mean": st.mean(server) if server else None,
        "max": max(server) if server else None,
        "n": len(server),
        "rebuilt_mean": st.mean(rebuilt) if rebuilt else None,
        "rebuilt_disagreements": disagree,
    }


def mix_decomposition(runs: dict[str, Run], policy: str) -> dict:
    """Is an arm's waste anything more than the tiers it chose?

    Pools the per-tier waste rate across the four arms of one policy, predicts
    each arm's overall waste from its own dispatch mix against that pooled
    rate, and reports the residual against its binomial standard error. A
    selector that could see feasibility would show a negative residual well
    outside the error; one that only reorders tiers shows nothing.
    """
    arms = [runs[f"{a}|{policy}"] for a in ARMS]
    pooled = {}
    for tier in TIER_TRAIN:
        items = [d for r in arms for d in r.dispatches if d.reported and d.tier == tier]
        pooled[tier] = sum(1 for d in items if not d.raw_in_range) / len(items)

    out = {"pooled_tier_rate": pooled, "arms": {}}
    for run in arms:
        reported = [d for d in run.dispatches if d.reported]
        n = len(reported)
        share = {t: sum(1 for d in reported if d.tier == t) / n for t in TIER_TRAIN}
        predicted = sum(share[t] * pooled[t] for t in TIER_TRAIN)
        measured = sum(1 for d in reported if not d.raw_in_range) / n
        se = (predicted * (1 - predicted) / n) ** 0.5
        out["arms"][run.arm] = {
            "measured": measured,
            "predicted_from_mix": predicted,
            "residual": measured - predicted,
            "se": se,
            "z": (measured - predicted) / se,
        }
    return out


def crossover_round(run: Run, baseline: Run, hold: int = 200) -> int | None:
    """First round after which this arm is slower than its policy's baseline
    and stays slower for `hold` rounds. The Stage 3 analysis put this at ~2,400
    for C=4 with no mobility."""
    shared = sorted(r for r in run.clock if r in baseline.clock)
    for i, rnd in enumerate(shared):
        if rnd < 50 or run.clock[rnd] <= baseline.clock[rnd]:
            continue
        if all(run.clock[q] > baseline.clock[q]
               for q in shared[i:] if q <= rnd + hold):
            return rnd
    return None


def conservation_check(run: Run, sleeps: dict[int, float],
                       tier_sleep: dict[str, float]) -> dict:
    """`T = rounds * mean_sleep(mix) / (C * (1 - waste))`.

    Stage 3's conservation law said the dispatch tier mix alone sets the total
    simulated time. Under gating there is a second term -- the cycles waste
    forces the pipeline to repeat -- so this checks the law in the form that
    includes it. `packing` is the sanity check on the queue model: the fraction
    of C-slot-seconds that actually held a training client.
    """
    reported = [d for d in run.dispatches if d.reported]
    rounds = max(run.clock)
    measured = run.clock[rounds]
    share = {t: sum(1 for d in reported if d.tier == t) / len(reported)
             for t in TIER_TRAIN}
    mean_sleep = sum(share[t] * tier_sleep[t] for t in TIER_TRAIN)
    waste = sum(1 for d in reported if not d.uploaded) / len(reported)
    predicted = rounds * mean_sleep / (CONCURRENCY * (1 - waste))
    packing = sum(sleeps[d.client_id] for d in reported) / (CONCURRENCY * measured)
    return {
        "rounds": rounds,
        "measured": measured,
        "predicted": predicted,
        "rel_error": (predicted - measured) / measured,
        "packing": packing,
    }


def redispatch_gap(run: Run) -> dict:
    """Cycles until the same client is dispatched again, after a wasted
    dispatch versus after a useful one.

    `STATUS.md` §5.4 flagged a suspected feedback loop: a vehicle whose uploads
    keep failing is never marked explored, keeps its 10x `G_i` proxy and so
    keeps being picked. If that loop bites, the gap after waste is SHORTER, and
    shorter in the scoring arms than in `baseline`.
    """
    reported = [d for d in run.dispatches if d.reported]
    positions: dict[int, list[int]] = defaultdict(list)
    for i, d in enumerate(reported):
        positions[d.client_id].append(i)
    after_waste, after_useful = [], []
    for idxs in positions.values():
        for a, b in zip(idxs, idxs[1:]):
            # Keyed on physical outcome so the two policies are comparable.
            (after_waste if not reported[a].raw_in_range else after_useful).append(b - a)
    return {
        "after_waste": st.mean(after_waste) if after_waste else None,
        "after_useful": st.mean(after_useful) if after_useful else None,
        "ratio": (st.mean(after_waste) / st.mean(after_useful)
                  if after_waste and after_useful else None),
        "n_waste": len(after_waste),
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
        "evaluations": len(acc),
        "plateau": st.median(tail),
        "best": max(acc),
        "noise_sd": st.stdev(residuals) if len(residuals) > 1 else None,
    }
    for target in THRESHOLDS:
        hit = next((r for r, v in zip(rows, smoothed) if v >= target / 100.0), None)
        stats[f"rounds_to_{target}"] = hit["round"] if hit else None
        stats[f"tta_h_{target}"] = (
            hit["sim_time_reconstructed"] / 3600.0 if hit else None
        )
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--out", type=Path, default=EVIDENCE / "task4_stepD.json")
    args = parser.parse_args()

    sleeps = load_sleeps()
    tiers = load_tiers()
    oracle = CoverageOracle.from_csv(COVERAGE_CSV)

    runs: dict[str, Run] = {}
    print(f"trace coverage spans t in [{oracle.t_start:.1f}, {oracle.t_end:.1f}] s\n")
    for policy in POLICIES:
        for arm in ARMS:
            run = parse_and_replay(arm, policy, sleeps, tiers)
            runs[f"{arm}|{policy}"] = run
            print(f"{run.name}")
            for line in validate(run, sleeps, oracle):
                print(line)
            print()

    if args.validate:
        return 0

    tier_sleep = {
        t: st.mean([s for c, s in sleeps.items() if tiers[c] == t])
        for t in TIER_TRAIN
    }

    payload = {"_mix_decomposition": {
        p: mix_decomposition(runs, p) for p in POLICIES
    }}
    for key, run in runs.items():
        baseline = runs[f"baseline|{run.policy}"]
        payload[key] = {
            "arm": run.arm,
            "policy": run.policy,
            "crossover_round": (None if run.arm == "baseline"
                                else crossover_round(run, baseline)),
            "conservation": conservation_check(run, sleeps, tier_sleep),
            "redispatch_gap": redispatch_gap(run),
            "waste": waste_table(run),
            "dispatch_share": dispatch_share(run),
            "useful_work_rate": useful_work_rate(run),
            "stage0": stage0_counterfactual(run, oracle, sleeps),
            "staleness": staleness(run),
            "curve": curve_stats(run),
            "clock_at": {str(c): clock_at(run, c) for c in CHECKPOINTS},
            "plato_clock_final": run.curve[-1]["sim_time_plato"] if run.curve else None,
            "rounds_done": run.curve[-1]["round"] if run.curve else None,
            "dispatch_cycles": sum(1 for d in run.dispatches if d.reported),
            "idle_events": len(run.idles),
            "idle_seconds": sum(d for _, d in run.idles),
            "empty_rounds": run.empty_rounds,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")

    csv_path = args.out.with_name("task4_waste_windowed.csv")
    fields = ["arm", "policy", "scope", "window", "tier", "dispatches",
              "useful", "policy_wasted", "physical_wasted",
              "policy_wasted_frac", "physical_wasted_frac"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key, run in runs.items():
            table = payload[key]["waste"]
            rows = [("overall", "all", "all", table["overall"])]
            rows += [("tier", "all", t, v) for t, v in table["per_tier"].items()]
            rows += [("window", w, "all", v) for w, v in table["per_window"].items()]
            rows += [("tier_window", k.split("|")[1], k.split("|")[0], v)
                     for k, v in table["per_tier_window"].items()]
            for scope, window, tier, cell in rows:
                writer.writerow({
                    "arm": run.arm, "policy": run.policy, "scope": scope,
                    "window": window, "tier": tier,
                    "dispatches": cell["dispatches"], "useful": cell["useful"],
                    "policy_wasted": cell["wasted"],
                    "physical_wasted": cell["physical_wasted"],
                    "policy_wasted_frac": round(cell["wasted_frac"], 6),
                    "physical_wasted_frac": round(cell["physical_wasted_frac"], 6),
                })
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
