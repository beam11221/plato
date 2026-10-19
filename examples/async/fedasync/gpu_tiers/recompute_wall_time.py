"""
Rebuild each run's simulated clock from its schedule, free of host effects.

Why the reported clock cannot be compared across arms
-----------------------------------------------------
PLATO advances its simulated clock to the finish time of whichever report it
aggregates (``servers/base.py:1159``), and that finish time is

    finish_time = training_time + processing_time + comm_time + start_time
                  └ simulated ┘   └──── measured on the host that ran it ────┘

``training_time`` is the honest part: with ``sleep_simulation = true`` the client
reports ``(avg_training_time + tier_sleep) * epochs`` and never really sleeps.
But ``processing_time`` and ``comm_time`` are `time.time()` deltas for serialising,
saving and shipping a 42.69 MB payload — real work on real hardware.  They enter
the *simulated* clock as an additive floor c per report.

That floor is not a constant of the experiment; it is a property of the machine
and its load at that moment.  The sweep's arms did not share a machine — the
`cpu` arm ran on a second pod while `gpu` and `baseline` ran here, and arms that
share a pod contend for it.  So the reported ``elapsed_time`` of two arms differs
partly because their selectors differed and partly because their hosts did, and
nothing in the CSV separates the two.

What this script does instead
-----------------------------
The logs record the whole schedule: which client was dispatched in which round
(``[GpuTierSleep] round=N client_id=C … sleep_time=S``) and which clients were
merged into the global model in which round, with their staleness
(``FedAsync: K client(s) in aggregation pool: #C(staleness=S)``).  Staleness is
``current_round - starting_round``, so each merge names its own dispatch round.

That is enough to replay the recorded schedule and recompute the clock with a
*uniform* cost model, reproducing PLATO's own rule:

    clock[r] = max over clients merged in round r of
                   clock[dispatch_round(c)] + (avg_training_time + sleep_c) * epochs + floor

held non-decreasing.  Every arm then pays the same floor for the same payload, so
a difference between two arms is a difference between two selectors.

This replays the schedule that actually happened; it does not re-run selection.
The recorded choices were themselves made against the contaminated clock (the
selector reads ``wall_now``), and no post-hoc pass can undo that — a rerun with
``comm_simulation = true`` is the only way to remove it at the source.  What this
does remove is the contamination in the *measurement*, which is where it does the
most damage: it is the axis every comparison is read off.

Outputs, into ``--out-dir`` (default ``figures/clean_time``):

    <run>.csv        round, clean_elapsed, reported_elapsed, and the per-report
                     floor implied by the reported clock
    floor_report.csv one row per run: the implied floor's median/IQR, i.e. what
                     that host was really charging, and how far the reported
                     clock drifts from the clean one

Usage:
    python3 recompute_wall_time.py                       # every run, floor 0
    python3 recompute_wall_time.py --floor 2.0           # charge a uniform 2 s
    python3 recompute_wall_time.py --runs gpu_xa20_gap40 --verbose
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import deque
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
DEFAULT_LOG_DIR = _MODULE_DIR / "logs"
DEFAULT_CONFIG_DIR = _MODULE_DIR / "configs"
DEFAULT_RUNS_DIR = _MODULE_DIR.parent / "runs" / "gap_sweep"
DEFAULT_OUT_DIR = _MODULE_DIR / "figures" / "clean_time"

DISPATCH_RE = re.compile(
    r"\[GpuTierSleep\] round=(\d+) client_id=(\d+) tier_id=(\w+) .*?sleep_time=([\d.]+)"
)
POOL_RE = re.compile(r"FedAsync: \d+ client\(s\) in aggregation pool: (.+)")
POOL_ENTRY_RE = re.compile(r"#(\d+)\(staleness=(\d+)\)")
ROUND_MARKER_RE = re.compile(r"\[(?:FedAsync|PolarisVi)\] Round (\d+)")


def read_config(config_path: Path) -> dict:
    """avg_training_time / epochs, the two numbers the cost model needs."""
    values = {"avg_training_time": 0.0, "epochs": 1}
    if not config_path.exists():
        return values
    for line in config_path.read_text().splitlines():
        for key in values:
            match = re.match(rf"\s*{key}\s*=\s*([\d.]+)", line)
            if match:
                values[key] = float(match.group(1))
    values["epochs"] = int(values["epochs"])
    return values


def parse_schedule(log_path: Path) -> tuple[dict, list, int]:
    """(dispatch, merges, mismatches) from one run's log.

    ``dispatch``  client_id -> [(round, sleep_time), …] in log order.
    ``merges``    per round, in order: [(client_id, staleness), …].

    Merge rounds are numbered sequentially — PLATO aggregates once per round —
    and cross-checked against the explicit ``Round N`` markers.  The marker is
    absent on evaluation rounds (that line only prints when evaluation is
    *skipped*), so a stale marker is not evidence of anything and only a marker
    seen since the previous merge is allowed to disagree.
    """
    dispatch: dict[int, list[tuple[int, float]]] = {}
    merges: list[tuple[int, list[tuple[int, int]]]] = []
    marker: int | None = None
    mismatches = 0

    with log_path.open(errors="replace") as handle:
        for line in handle:
            if "GpuTierSleep] round=" in line:
                match = DISPATCH_RE.search(line)
                if match:
                    dispatch.setdefault(int(match[2]), []).append((int(match[1]), float(match[4])))
                continue

            found = ROUND_MARKER_RE.search(line)
            if found:
                marker = int(found[1])
                continue

            if "in aggregation pool" in line:
                match = POOL_RE.search(line)
                if not match:
                    continue
                entries = [
                    (int(cid), int(staleness))
                    for cid, staleness in POOL_ENTRY_RE.findall(match[1])
                ]
                rnd = len(merges) + 1
                if marker is not None and marker != rnd:
                    mismatches += 1
                marker = None
                merges.append((rnd, entries))

    return dispatch, merges, mismatches


def dispatch_queues(dispatch: dict) -> dict[int, deque]:
    """Per client, its dispatches in round order, to be consumed FIFO.

    A merge is matched to the client's earliest unconsumed dispatch rather than
    to ``round - staleness``.  A client trains one job at a time — the server
    drops it from ``training_clients`` on report — so FIFO is exact, whereas the
    staleness arithmetic is not: measured against the recorded dispatch rounds it
    disagrees by hundreds of rounds on re-selected clients, and using it puts the
    implied per-report floor at a median of -2.5 s, which is not a thing that can
    happen.  With FIFO the same measurement lands at +0.38 s with an inter-decile
    range of 0.07 s.
    """
    return {cid: deque(sorted(rounds)) for cid, rounds in dispatch.items()}


def replay(dispatch: dict, merges: list, avg_training_time: float, epochs: int, floor: float):
    """Recompute the clock from the recorded schedule.  Returns rows per round.

    ``start_time`` is the clock at the *start* of the dispatch round — PLATO sets
    ``round_start_wall_time = wall_time`` before dispatching (``base.py:534``) —
    which is ``clock[s - 1]``, the value left by the previous round's
    aggregation.  Using ``clock[s]`` instead charges the dispatch round's own
    aggregation to the client and makes the reconstruction diverge.
    """
    queues = dispatch_queues(dispatch)
    clock = {0: 0.0}
    rows = []
    now = 0.0
    unresolved = 0

    for rnd, entries in merges:
        finish_times = []
        for client_id, _ in entries:
            queue = queues.get(client_id)
            if not queue:
                unresolved += 1
                continue
            start_round, sleep = queue.popleft()
            started = clock.get(start_round - 1, now)
            finish_times.append(started + (avg_training_time + sleep) * epochs + floor)
        if finish_times:
            now = max(now, max(finish_times))
        clock[rnd] = now
        rows.append({"round": rnd, "clean_elapsed": now})

    return rows, unresolved


def implied_floor(reported: dict, dispatch: dict, merges: list,
                  avg_training_time: float, epochs: int) -> list[float]:
    """Per-report ``processing_time + comm_time`` the host actually charged.

    Only single-client rounds are usable: there the reported clock lands exactly
    on that one client's finish time, so the residual after subtracting the
    simulated training time is what the machine spent serialising and shipping
    42.69 MB — the quantity that has no business being in a simulated clock.
    """
    queues = dispatch_queues(dispatch)
    floors = []
    for rnd, entries in merges:
        client_id = entries[0][0]
        queue = queues.get(client_id)
        if not queue:
            continue
        start_round, sleep = queue.popleft()
        if len(entries) != 1 or rnd not in reported:
            continue
        started = 0.0 if start_round <= 1 else reported.get(start_round - 1)
        if started is None:
            continue
        floors.append(reported[rnd] - started - (avg_training_time + sleep) * epochs)
    return floors


def reported_elapsed(run_dir: Path) -> dict[int, float]:
    results = sorted((run_dir / "results").glob("*.csv"), key=lambda p: p.stat().st_mtime)
    results = [p for p in results if not p.stem.endswith(("_binned", "_accuracy_bins"))]
    if not results:
        return {}
    with results[-1].open(newline="") as f:
        return {int(row["round"]): float(row["elapsed_time"]) for row in csv.DictReader(f)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--runs", nargs="+", default=None, help="run names (default: all)")
    parser.add_argument(
        "--floor",
        type=float,
        default=0.0,
        help="uniform per-report cost in seconds, charged to every arm alike "
        "(default: 0 — tier sleep only; try 2.0 to keep the clock on the old scale)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    names = args.runs or sorted(p.stem for p in args.log_dir.glob("*.log") if ".cleaner" not in p.name)
    report_rows = []

    print(f"Uniform floor c = {args.floor:.2f} s per report\n")
    print(f"{'run':<24}{'rounds':>8}{'clean_h':>10}{'reported_h':>12}{'ratio':>8}"
          f"{'floor_med':>11}{'floor_iqr':>11}{'refit_err':>10}")

    for name in names:
        log_path = args.log_dir / f"{name}.log"
        run_dir = args.runs_dir / name
        if not log_path.exists() or not run_dir.exists():
            continue

        config = read_config(args.config_dir / f"{name}.toml")
        dispatch, merges, mismatches = parse_schedule(log_path)
        if not merges:
            print(f"{name:<24}  no aggregation events in log — skipped")
            continue

        rows, unresolved = replay(
            dispatch, merges, config["avg_training_time"], config["epochs"], args.floor
        )
        reported = reported_elapsed(run_dir)
        floors = implied_floor(
            reported, dispatch, merges, config["avg_training_time"], config["epochs"]
        )

        for row in rows:
            row["reported_elapsed"] = reported.get(row["round"])
        out_path = args.out_dir / f"{name}.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "clean_elapsed", "reported_elapsed"])
            writer.writeheader()
            writer.writerows(rows)

        last_round = rows[-1]["round"]
        clean_h = rows[-1]["clean_elapsed"] / 3600.0
        reported_h = reported.get(last_round, float("nan")) / 3600.0
        floor_med = statistics.median(floors) if floors else float("nan")
        check_rows, _ = replay(
            dispatch, merges, config["avg_training_time"], config["epochs"],
            floor_med if floors else 0.0,
        )
        residuals = [
            abs(row["clean_elapsed"] - reported[row["round"]]) / max(reported[row["round"]], 1.0)
            for row in check_rows
            if row["round"] in reported and reported[row["round"]] > 0
        ]
        fidelity = statistics.median(residuals) if residuals else float("nan")
        floor_iqr = (
            statistics.quantiles(floors, n=4)[2] - statistics.quantiles(floors, n=4)[0]
            if len(floors) > 8 else float("nan")
        )
        print(f"{name:<24}{last_round:>8}{clean_h:>10.2f}{reported_h:>12.2f}"
              f"{reported_h / clean_h if clean_h else float('nan'):>8.2f}"
              f"{floor_med:>11.3f}{floor_iqr:>11.3f}{fidelity:>10.2%}")

        if args.verbose and (mismatches or unresolved):
            print(f"    round-marker disagreements: {mismatches}, unresolved dispatches: {unresolved}")

        report_rows.append({
            "run": name,
            "rounds": last_round,
            "clean_hours": round(clean_h, 4),
            "reported_hours": round(reported_h, 4),
            "reported_over_clean": round(reported_h / clean_h, 4) if clean_h else None,
            "implied_floor_median": round(floor_med, 3),
            "implied_floor_iqr": round(floor_iqr, 3),
            "implied_floor_n": len(floors),
            "refit_median_error": round(fidelity, 5),
            "round_marker_mismatches": mismatches,
            "unresolved_dispatches": unresolved,
        })

    if report_rows:
        report_path = args.out_dir / "floor_report.csv"
        with report_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"\nWrote {report_path} and {len(report_rows)} per-run CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
