"""
Map SUMO vehicles onto PLATO client IDs and emit a coverage-window table.

Why a remap is needed
---------------------
The SUMO trace labels each vehicle with a hardware tier of its own
(`vehicles.csv`), but this project's single source of truth for hardware is
`gpu_tiers/input/tiers.csv` plus a `clients.csv` that lays tiers out in
contiguous ID blocks (1-10 xavier, 11-17 orin, 18-20 thor for a 20-vehicle
fleet).  Two reasons that layout must win:

  * PLATO derives a client's data shard from `random_seed` and the client ID,
    so contiguous blocks keep a shard attached to a fixed tier across runs.
  * Every existing tool in this repo -- the sleep-time generator, the predictor,
    the dispatch-share analysis -- reads the tier from `clients.csv`.

So we keep each vehicle's trajectory and only renumber it.  The mapping is
tier-preserving: a vehicle SUMO called `thor` becomes a client ID that
`clients.csv` also calls `thor`.  That is possible because the two files agree
on the composition exactly (10/7/3 for n20, 50/35/15 for n100); the script
refuses to run if they ever disagree.

Do NOT map `veh_00 -> 1, veh_01 -> 2, ...` instead.  SUMO assigns tiers in no
particular order, so a positional map would silently put, say, an orin vehicle
on a client ID the rest of the pipeline believes is xavier.

Output
------
`coverage_<cell>.csv`  client_id, window_no, t_enter_s, t_exit_s, residence_s
    One row per continuous stay inside RSU range.  SUMO's own `tier` column is
    dropped on purpose so nothing downstream can read it by accident.
`mapping_<cell>.csv`   client_id, veh_id, tier_id
    The audit trail, so any result can be traced back to a vehicle.

Usage
-----
    python3 remap_fleet.py \
        --trace-dir sumo_trace/e1_closed_n20_seed1 \
        --clients-csv input/clients_n20.csv \
        --out-coverage input/coverage_n20.csv \
        --out-mapping  input/mapping_n20.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_vehicle_tiers(trace_dir: Path) -> dict[str, str]:
    """Return {veh_id: tier_id} from the trace's vehicles.csv."""
    with (trace_dir / "vehicles.csv").open(newline="") as f:
        return {row["veh_id"]: row["tier"] for row in csv.DictReader(f)}


def read_client_tiers(clients_csv: Path) -> dict[int, str]:
    """Return {client_id: tier_id} from this project's clients.csv."""
    with clients_csv.open(newline="") as f:
        return {int(row["client_id"]): row["tier_id"] for row in csv.DictReader(f)}


def build_mapping(
    vehicle_tiers: dict[str, str],
    client_tiers: dict[int, str],
) -> dict[str, int]:
    """Return {veh_id: client_id}, pairing vehicles and clients within each tier.

    Both sides are sorted before pairing, so the mapping is deterministic and
    reproducible from the two input files alone.
    """
    by_tier_vehicles: dict[str, list[str]] = defaultdict(list)
    for veh_id, tier in vehicle_tiers.items():
        by_tier_vehicles[tier].append(veh_id)

    by_tier_clients: dict[str, list[int]] = defaultdict(list)
    for client_id, tier in client_tiers.items():
        by_tier_clients[tier].append(client_id)

    if set(by_tier_vehicles) != set(by_tier_clients):
        raise SystemExit(
            f"tier names differ: SUMO has {sorted(by_tier_vehicles)}, "
            f"clients.csv has {sorted(by_tier_clients)}"
        )

    mapping: dict[str, int] = {}
    for tier in sorted(by_tier_vehicles):
        vehicles = sorted(by_tier_vehicles[tier])
        clients = sorted(by_tier_clients[tier])
        if len(vehicles) != len(clients):
            raise SystemExit(
                f"tier '{tier}': SUMO has {len(vehicles)} vehicles but "
                f"clients.csv has {len(clients)} clients. A tier-preserving map "
                f"needs the two compositions to match exactly."
            )
        mapping.update(zip(vehicles, clients))
    return mapping


def write_coverage(
    trace_dir: Path,
    mapping: dict[str, int],
    out_path: Path,
    start_time: float = 0.0,
) -> int:
    """Rewrite coverage_windows.csv keyed by client_id. Returns the row count.

    Windows before *start_time* are dropped and a window straddling it is
    clipped to begin there, so the file contains exactly the windows the
    experiment can reach. The trace's `t` starts at -150 s: that is SUMO's
    warm-up, during which vehicles are still spreading out from their initial
    positions and the mean speed has not settled (17.76 m/s at t = -150 against
    a steady 16.6-17.3 by t = -75). Training starts at t = 0, so the FL clock
    never queries a negative time -- but leaving those windows in the file makes
    `t_start` read as -149 and invites someone to use them.
    """
    dropped = 0
    rows = []
    with (trace_dir / "coverage_windows.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            client_id = mapping.get(row["veh_id"])
            if client_id is None:
                raise SystemExit(
                    f"{row['veh_id']} appears in coverage_windows.csv but not in "
                    "vehicles.csv"
                )

            enter, exit_ = float(row["t_enter_s"]), float(row["t_exit_s"])
            if exit_ < start_time:
                dropped += 1
                continue
            # A vehicle already inside range when training starts stays inside;
            # only the part of the window before start_time is unreachable.
            enter = max(enter, start_time)

            rows.append(
                (
                    client_id,
                    int(row["window_no"]),
                    enter,
                    exit_,
                    exit_ - enter,
                )
            )

    if dropped:
        print(f"  dropped {dropped} window(s) entirely before t = {start_time}")

    rows.sort(key=lambda r: (r[0], r[2]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["client_id", "window_no", "t_enter_s", "t_exit_s", "residence_s"]
        )
        writer.writerows(rows)
    return len(rows)


def write_mapping(
    mapping: dict[str, int],
    vehicle_tiers: dict[str, str],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["client_id", "veh_id", "tier_id"])
        for veh_id, client_id in sorted(mapping.items(), key=lambda kv: kv[1]):
            writer.writerow([client_id, veh_id, vehicle_tiers[veh_id]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--clients-csv", type=Path, required=True)
    parser.add_argument("--out-coverage", type=Path, required=True)
    parser.add_argument("--out-mapping", type=Path, required=True)
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Drop coverage before this trace timestamp (default 0.0, the end "
        "of SUMO's warm-up). Training starts here, so earlier windows are "
        "unreachable.",
    )
    args = parser.parse_args()

    vehicle_tiers = read_vehicle_tiers(args.trace_dir)
    client_tiers = read_client_tiers(args.clients_csv)
    mapping = build_mapping(vehicle_tiers, client_tiers)

    n_rows = write_coverage(
        args.trace_dir, mapping, args.out_coverage, args.start_time
    )
    write_mapping(mapping, vehicle_tiers, args.out_mapping)

    print(f"Mapped {len(mapping)} vehicles onto client IDs 1-{len(mapping)}.")
    print(f"  {args.out_coverage}  ({n_rows} coverage windows)")
    print(f"  {args.out_mapping}")


if __name__ == "__main__":
    main()
