"""
Turn a server log's `[SumoRsu] report` lines into a CSV, and print the tally.

One row per client report:

    round, client_id, tier, dispatch_t, finish_t, duration, uploaded

`uploaded = 0` is a wasted round: the vehicle trained and then had nothing to
upload to, so the global model never saw the work. That column is the headline
metric of this experiment -- the next step adds location prediction, and the
number it has to move is the share of zeros.

Usage:
    python3 parse_waste_log.py <server.log> [output.csv]
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

REPORT_RE = re.compile(
    r"\[SumoRsu\] report round=(\d+) client=(\d+) tier=(\S+) "
    r"dispatch_t=([\d.-]+) finish_t=([\d.-]+) uploaded=([01])"
)


def parse(log_path: Path) -> list[dict]:
    rows = []
    with log_path.open(errors="replace") as f:
        for line in f:
            match = REPORT_RE.search(line)
            if not match:
                continue
            round_no, client_id, tier, dispatch_t, finish_t, uploaded = match.groups()
            rows.append(
                {
                    "round": int(round_no),
                    "client_id": int(client_id),
                    "tier": tier,
                    "dispatch_t": float(dispatch_t),
                    "finish_t": float(finish_t),
                    "duration": float(finish_t) - float(dispatch_t),
                    "uploaded": int(uploaded),
                }
            )
    return rows


def summarize(rows: list[dict]) -> None:
    dispatched: dict[str, int] = defaultdict(int)
    wasted: dict[str, int] = defaultdict(int)
    for row in rows:
        dispatched[row["tier"]] += 1
        if not row["uploaded"]:
            wasted[row["tier"]] += 1

    print(f"\n{'tier':<10}{'dispatched':>12}{'wasted':>10}{'waste %':>10}")
    for tier in sorted(dispatched):
        share = 100 * wasted[tier] / dispatched[tier]
        print(f"{tier:<10}{dispatched[tier]:>12}{wasted[tier]:>10}{share:>9.1f}%")

    total = sum(dispatched.values())
    total_wasted = sum(wasted.values())
    print(
        f"{'ALL':<10}{total:>12}{total_wasted:>10}"
        f"{100 * total_wasted / max(total, 1):>9.1f}%"
    )


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    log_path = Path(sys.argv[1])
    out_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else log_path.with_suffix(".waste.csv")
    )

    rows = parse(log_path)
    if not rows:
        sys.exit(f"No '[SumoRsu] report' lines found in {log_path}.")

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} reports to {out_path}.")
    summarize(rows)


if __name__ == "__main__":
    main()
