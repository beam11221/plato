"""
Check one sweep run against the preconditions in HANDOFF § 4b.

A run that fails these is void rather than interesting — the scheduler was not
doing what the experiment assumes — so this is meant to be run as each run
finishes, not at the end of a 30-hour sweep.

    python3 check_run.py gpu_xa20_gap40          # one run
    python3 check_run.py --all                   # every run with a log
    python3 check_run.py --all --quiet           # one line each, for a monitor

Checks, from § 4b:

1. in-flight concurrency holds near ``max_concurrency`` (the flush-accounting bug
   drained it to 1; see § 4.9)
2. flushes stay occasional and staleness stays under ``staleness_bound``
3. mean staleness ~ C/M — this is an identity, not a target: the mean of
   T_i/T_mean is 1 whatever the selector picks, so a run that disagrees means the
   model in § 4.9 is wrong, and every comparison built on it is suspect

Exit status is 1 if any checked run fails, so it can gate a rerun.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

from plot_time_to_accuracy import concurrency_trace  # noqa: E402
from recompute_wall_time import dispatch_queues, parse_schedule, read_config  # noqa: E402

DEFAULT_LOG_DIR = _MODULE_DIR / "logs"
DEFAULT_CONFIG_DIR = _MODULE_DIR / "configs"
DEFAULT_RUNS_DIR = _MODULE_DIR.parent / "runs" / "gap_sweep"

# Concurrency may dip on a flush and refill next round, so the bar is the median
# of the steady state, not the minimum.
MIN_CONCURRENCY_FRACTION = 0.8
MAX_FLUSH_FRACTION = 0.10
MAX_STALENESS_MODEL_ERROR = 0.25


def _config_value(config_path: Path, key: str, default: float) -> float:
    if not config_path.exists():
        return default
    match = re.search(rf"^{key}\s*=\s*([\d.]+)", config_path.read_text(), re.MULTILINE)
    return float(match.group(1)) if match else default


def check(run: str, log_dir: Path, config_dir: Path, runs_dir: Path) -> tuple[bool, str, dict]:
    log_path = log_dir / f"{run}.log"
    if not log_path.exists():
        return False, f"{run}: no log", {}

    config_path = config_dir / f"{run}.toml"
    bound = _config_value(config_path, "staleness_bound", 25)
    max_conc = _config_value(config_path, "max_concurrency", 10)
    cfg = read_config(config_path)

    trace, extra_merged = concurrency_trace(log_path)
    if not trace:
        return False, f"{run}: no aggregation events", {}

    dispatch, merges, _ = parse_schedule(log_path)
    queues = dispatch_queues(dispatch)
    staleness, durations, flushes = [], [], 0
    for rnd, entries in merges:
        if len(entries) > 1:
            flushes += 1
        for client_id, _ in entries:
            queue = queues.get(client_id)
            if queue:
                start_round, sleep = queue.popleft()
                staleness.append(rnd - start_round)
                durations.append(cfg["avg_training_time"] + sleep)
    if not staleness:
        return False, f"{run}: no matched dispatches", {}

    tail = trace[len(trace) // 5:]
    concurrency = statistics.median(tail)
    merges_per_round = sum(len(e) for _, e in merges) / len(merges)
    flush_fraction = flushes / len(merges)
    staleness.sort()
    p90 = staleness[int(len(staleness) * 0.9)]
    mean_staleness = statistics.mean(staleness)
    predicted = concurrency / merges_per_round
    model_error = abs(predicted - mean_staleness) / mean_staleness

    failures = []
    if concurrency < max_conc * MIN_CONCURRENCY_FRACTION:
        failures.append(f"concurrency {concurrency:.0f} < {max_conc * MIN_CONCURRENCY_FRACTION:.0f}")
    if flush_fraction > MAX_FLUSH_FRACTION:
        failures.append(f"flushes {flush_fraction:.1%} > {MAX_FLUSH_FRACTION:.0%}")
    if p90 > bound:
        failures.append(f"staleness p90 {p90} > bound {bound:.0f}")
    if model_error > MAX_STALENESS_MODEL_ERROR:
        failures.append(f"mean staleness {mean_staleness:.1f} vs C/M {predicted:.1f}")

    accuracy, rounds_run = None, len(merges)
    results = runs_dir / run / "results"
    csvs = sorted(results.glob("*.csv"), key=lambda p: p.stat().st_mtime) if results.is_dir() else []
    if csvs:
        with csvs[-1].open(newline="") as f:
            values = [float(row["accuracy"]) for row in csv.DictReader(f) if float(row["accuracy"]) > 0]
        accuracy = max(values) if values else None

    stats = {
        "run": run,
        "rounds": rounds_run,
        "concurrency": concurrency,
        "merges_per_round": merges_per_round,
        "flush_fraction": flush_fraction,
        "mean_staleness": mean_staleness,
        "predicted_staleness": predicted,
        "staleness_p90": p90,
        "extra_merged": extra_merged,
        "best_accuracy": accuracy,
        "finished": (log_dir / f"{run}.done").exists(),
    }
    verdict = "PASS" if not failures else "FAIL"
    summary = (
        f"{verdict}  {run:<22} r={rounds_run:<6} conc={concurrency:.0f}  "
        f"flush={flush_fraction:.1%}  stale mean={mean_staleness:.1f} (C/M {predicted:.1f}) "
        f"p90={p90}/{bound:.0f}  acc={accuracy if accuracy is None else f'{accuracy:.4f}'}"
    )
    if failures:
        summary += "  <- " + "; ".join(failures)
    return not failures, summary, stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("runs", nargs="*", help="run names (default: --all)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="one line per run, no header")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runs = args.runs
    if args.all or not runs:
        runs = sorted(
            p.stem for p in args.log_dir.glob("*.log")
            if ".cleaner" not in p.name and not p.stem.startswith(("verify", "runner"))
        )
    if not runs:
        print("no runs to check")
        return 0

    if not args.quiet:
        print(f"Preconditions from HANDOFF § 4b — {len(runs)} run(s)\n")
    ok = True
    for run in runs:
        passed, summary, _ = check(run, args.log_dir, args.config_dir, args.runs_dir)
        print(summary)
        ok = ok and passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
