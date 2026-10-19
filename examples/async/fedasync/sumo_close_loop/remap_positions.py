"""
Stage 5b (prompts/STAGE5B_fix_Ri.md, Step 1): remap the raw per-second SUMO
trace onto client_id, the way remap_fleet.py already does for coverage
windows -- and for the same reason: `trace.csv`'s own `veh_id` labelling must
not leak downstream (mapping is tier-preserving, not positional; see
remap_fleet.py's docstring).

Unlike coverage windows, this table is filtered to `in_coverage = 1` rows
only: Stage 0 only ever asks for a client's position while it is inside
coverage (the in-range-now gate already filters to that set before Stage 0
runs), so rows outside coverage would never be queried and just bloat the
file for no reason -- 712,230 raw rows become ~470,985.

Output
------
`positions_<cell>.csv`  client_id, t, x_m, direction
    One row per second the client is in RSU range, from `start_time` onward
    (same t=0 clipping convention as write_coverage() in remap_fleet.py, and
    for the same reason: training starts at t=0, so nothing before it is
    reachable).

Usage
-----
    python3 remap_positions.py \
        --trace-dir sumo_trace/e1_closed_seed4126_range400 \
        --mapping input/mapping_n20r400.csv \
        --out input/positions_n20r400.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_mapping(mapping_csv: Path) -> dict[str, int]:
    """Return {veh_id: client_id} from remap_fleet.py's mapping_<cell>.csv."""
    with mapping_csv.open(newline="") as f:
        return {row["veh_id"]: int(row["client_id"]) for row in csv.DictReader(f)}


def write_positions(
    trace_dir: Path,
    mapping: dict[str, int],
    out_path: Path,
    start_time: float = 0.0,
) -> int:
    dropped = 0
    rows = []
    with (trace_dir / "trace.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["in_coverage"] != "1":
                continue
            t = float(row["t"])
            if t < start_time:
                dropped += 1
                continue
            client_id = mapping.get(row["veh_id"])
            if client_id is None:
                raise SystemExit(f"{row['veh_id']} appears in trace.csv but not in mapping")
            rows.append((client_id, t, float(row["x_m"]), int(row["direction"])))

    if dropped:
        print(f"  dropped {dropped} row(s) before t = {start_time}")

    rows.sort(key=lambda r: (r[0], r[1]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["client_id", "t", "x_m", "direction"])
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-time", type=float, default=0.0)
    args = parser.parse_args()

    mapping = read_mapping(args.mapping)
    n = write_positions(args.trace_dir, mapping, args.out, args.start_time)
    print(f"wrote {args.out} ({n} rows, {len(mapping)} clients)")


if __name__ == "__main__":
    main()
