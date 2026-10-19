"""
Regenerates every number in POPULATION_MODEL.md.

Read-only: replays the open trace through the real `SlotAllocator` and
`ReturningPopulation` (no PLATO, no training) and summarises the SUMO traces.

    python3 population_facts.py            # all sections, horizon 4,600 s
    python3 population_facts.py --horizon 7200
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from occupant_identity import ReturningPopulation, load_fleet
from stage7_population import SlotAllocator, partition_index

HERE = Path(__file__).resolve().parent
OPEN = HERE / "sumo_trace" / "e1_open_poisson_seed4126_range400"
CLOSED = HERE.parent / "sumo_close_loop" / "sumo_trace" / "e1_closed_seed4126_range400"
FLEET = (
    HERE.parent / "sumo_close_loop" / "input" / "clients_n20r400.csv",
    HERE.parent / "sumo_close_loop" / "output" / "client_sleep_times_n20r400.csv",
)
SLOTS, PHYSICS_SEED = 20, 4126
P_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


def trace_summary(d: Path) -> None:
    t = pd.read_csv(d / "trace.csv")
    w = pd.read_csv(d / "coverage_windows.csv")
    tp = t[t.t >= 0]
    n = tp.groupby("t").size()
    c = tp[tp.in_coverage == 1].groupby("t").size().reindex(n.index, fill_value=0)
    cov = tp[tp.in_coverage == 1].x_m
    print(f"\n## {d.name}")
    print(f"unique vehicles {t.veh_id.nunique()} | windows {len(w)}")
    print(f"on road     {n.mean():.2f} +- {n.std():.2f} (min {n.min()}, max {n.max()})")
    print(f"in coverage {c.mean():.2f} +- {c.std():.2f} (min {c.min()}, max {c.max()})")
    print(f"covered x   {cov.min():.1f} .. {cov.max():.1f} m")
    print(f"residence   {w.residence_s.mean():.1f} +- {w.residence_s.std():.1f} s | speed {tp.speed_mps.mean():.2f} m/s")

    t = t.sort_values(["veh_id", "t"])
    t["pos"] = np.where(t.direction == 1, t.x_m, 1200 - t.x_m)  # along own direction
    g = t.groupby("veh_id")
    dt = g.t.diff()
    print(f"first pos along direction: {g.pos.first().min():.1f} .. {g.pos.first().max():.1f} m")
    print(f"pos before any disappearance (gap > 1 s): min {g.pos.shift()[dt > 1].min():.1f} m "
          f"({int((dt > 1).sum())} disappearances)")
    last = g.agg(t1=("t", "last"), pos=("pos", "last"))
    print(f"last pos along direction (vehicles gone before the trace ends): "
          f"min {last[last.t1 < t.t.max()].pos.min():.1f} m")
    flips = (g.direction.diff().fillna(0) != 0).sum()
    print(f"direction flips {flips} of {int((dt > 1).sum())} re-insertions")


def open_events():
    events = []
    with (OPEN / "coverage_windows.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if float(r["t_exit_s"]) < 0:
                continue
            events.append((max(0.0, float(r["t_enter_s"])), 1, r["veh_id"], r["tier"]))
            events.append((float(r["t_exit_s"]), 0, r["veh_id"], r["tier"]))
    events.sort(key=lambda e: (e[0], e[1]))  # releases before assigns at equal t
    return events


def replay(p: float, events, horizon: float, fleet):
    """Run the Stage 11 allocator to `horizon`; return per-allocation records + time averages."""
    alloc = SlotAllocator(SLOTS, 0.0, base_seed=PHYSICS_SEED)
    rp = ReturningPopulation(p, fleet, PHYSICS_SEED)
    alloc.on_assign = lambda s, v, tier: (rp.assign(s, v, tier, PHYSICS_SEED, SLOTS), None)[1]
    alloc.on_release = lambda s, v: rp.release(s)
    recs, last = [], 0.0
    acc = collections.Counter()  # time integrals
    for t, kind, veh, tier in events:
        if t > horizon:
            break
        parts = [rp.slot_identity[s].partition for s in rp.slot_identity]
        span = t - last
        acc["occ"] += len(parts) * span
        acc["distinct"] += len(set(parts)) * span
        acc["commuter"] += list(rp.slot_kind.values()).count("commuter") * span
        acc["dup_time"] += span * (len(parts) > len(set(parts)))
        last = t
        if kind == 0:
            alloc.release(veh)
            continue
        in_use = {rp.slot_identity[s].partition: rp.slot_kind[s] for s in rp.slot_identity}
        slot, _ = alloc.assign(veh, tier)
        if slot is None:
            continue
        ident, k = rp.slot_identity[slot], rp.slot_kind[slot]
        recs.append(dict(t=t, veh=veh, kind=k, identity=ident.identity_id, partition=ident.partition,
                         veh_hash=partition_index(veh, SLOTS),
                         clash=in_use.get(ident.partition)))
    return pd.DataFrame(recs), {k: v / last for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=float, default=4600.0)
    args = ap.parse_args()
    H = args.horizon

    print("# Traces")
    trace_summary(OPEN)
    trace_summary(CLOSED)

    fleet = load_fleet(*FLEET)
    events = open_events()

    print("\n# Unique identities by p_return (commuters + transients)")
    for p in P_VALUES:
        cells = []
        for h in (4000, 4600, 7200, 36000):
            df, _ = replay(p, events, h, fleet)
            nc = df[df.kind == "commuter"].identity.nunique()
            cells.append(f"t<={h}: {df.identity.nunique()} ({nc}+{df.identity.nunique() - nc})")
        print(f"p={p}: " + " | ".join(cells) + f" | trace vehicles admitted to {36000}: {df.veh.nunique()}")

    print(f"\n# Commuter share, partitions and clashes (t <= {H:.0f} s)")
    for p in P_VALUES:
        df, avg = replay(p, events, H, fleet)
        clash = df[df.clash.notna()]
        pairs = collections.Counter(tuple(sorted((a, b))) for a, b in zip(clash.kind, clash.clash))
        heads = df.identity.nunique()
        print(f"p={p}: visits {(df.kind == 'commuter').mean():.1%} | in coverage "
              f"{avg['commuter']:.1f}/{avg['occ']:.1f} = {avg['commuter'] / avg['occ']:.1%} | "
              f"headcount {df[df.kind == 'commuter'].identity.nunique()}/{heads} | "
              f"distinct partitions in coverage {avg['distinct']:.1f} | arrivals clashing "
              f"{len(clash)}/{len(df)} = {len(clash) / len(df):.1%} | time with a clash "
              f"{avg['dup_time']:.1%} | pairs {dict(pairs)}")

    df, _ = replay(0.25, events, H, fleet)
    c7 = df[df.identity == "C7"]
    print(f"\n# C7 at p=0.25 (t <= {H:.0f} s): {len(c7)} visits, partition used "
          f"{sorted(c7.partition.unique())}, distinct hash(veh_id) of the lending vehicles "
          f"{c7.veh_hash.nunique()}")
    print(c7[["t", "veh", "veh_hash", "partition"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
