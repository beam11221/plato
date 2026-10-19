"""
Rebuild a Wen-environment run's clean clock from its logs (Prompt B, Step 3).

Replaces, for this environment, `gpu_tiers_dataset/clock_reconstruction.py`,
which assumes one static sleep per client and an immediate refill of every
freed slot -- neither holds once upload time depends on position and S2 can
leave a slot empty.

Inputs (all written by wen_server.py into the run directory):
    dispatch_log.csv   per dispatch: cycle, client_id, duration (T_train + T_up)
    merge_log.csv      per popped update, in pop order, tagged with the cycle
                       whose selection followed it
    cycle_log.csv      per dispatch cycle: the idle advance applied before it

Replay -- uses only durations, idle advances and the event ORDER, never the
logged times:

    now = 0
    for each cycle c:
        for each update merged before c (pop order): now = its finish
        now += idle_advance(c)
        for each dispatch in c: start = now; finish = start + duration

and compares every start / finish with the online values the server logged.

Usage:
    python wen_env/wen_clock_replay.py runs/wen_env/<run>    [--tol 1e-6]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def replay(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    dispatches = _rows(run_dir / "dispatch_log.csv")
    merges = _rows(run_dir / "merge_log.csv")
    cycles = _rows(run_dir / "cycle_log.csv")

    by_cycle_dispatch = defaultdict(list)
    for d in dispatches:
        by_cycle_dispatch[int(d["cycle"])].append(d)
    by_cycle_merge = defaultdict(list)
    for m in merges:
        by_cycle_merge[int(m["cycle"])].append(m)
    idle = {int(c["cycle"]): float(c["idle_advance"]) for c in cycles}

    now = 0.0
    in_flight: dict[int, float] = {}
    worst_start = worst_finish = worst_merge = 0.0
    n = 0
    last_cycle = max([*idle, *by_cycle_merge] or [0])
    for c in range(1, last_cycle + 1):
        for m in by_cycle_merge.get(c, []):
            now = in_flight.pop(int(m["client_id"]))
            worst_merge = max(worst_merge, abs(now - float(m["t_finish_clean"])))
        now = now + idle.get(c, 0.0)
        for d in by_cycle_dispatch.get(c, []):
            start = now
            finish = start + float(d["duration"])
            in_flight[int(d["client_id"])] = finish
            worst_start = max(worst_start, abs(start - float(d["t_dispatch"])))
            worst_finish = max(worst_finish, abs(finish - float(d["t_finish_clean"])))
            n += 1
    return {
        "run": run_dir.name, "dispatches": n, "merges": len(merges),
        "max_abs_start_diff_s": worst_start, "max_abs_finish_diff_s": worst_finish,
        "max_abs_merge_diff_s": worst_merge,
        "max_abs_diff_s": max(worst_start, worst_finish, worst_merge),
        "order_residuals": sum(int(m["order_violation"]) for m in merges),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()
    ok = True
    for run in args.runs:
        result = replay(run)
        result["pass"] = result["max_abs_diff_s"] < args.tol
        ok &= result["pass"]
        print(json.dumps(result))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
