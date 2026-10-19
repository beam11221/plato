"""
Recompute the simulated clock outside PLATO, from the dispatch order and the
per-client sleep times.

Why
---
`stage1_evidence/determinism.txt` section 3: PLATO builds every report's finish
time as

    training_time + processing_time + comm_time + start_time
                                              plato/servers/base.py:1025-1030

and that finish time becomes the simulated clock. `processing_time` is a
`time.perf_counter()` delta measured on the host
(`plato/clients/strategies/defaults.py:252`), so the "simulated" clock is
inflated by whatever else the machine was doing. Two runs on a loaded box are
therefore not comparable, parallel runs contaminate each other, and the x0.15
anchor rescale failed earlier for exactly this reason -- host overhead does not
scale with the anchor.

PLATO is not being fixed. The clock is recomputed afterwards from quantities that
are properties of the experiment rather than of the machine.

The model
---------
C slots, each holding one client at a time; a freed slot is refilled by the next
dispatch in the log's order.

    slots = [0.0] * C
    for client in dispatch_order:
        i        = argmin(slots)
        t_start  = slots[i]
        t_finish = t_start + sleep[client] + comm_time
        slots[i] = t_finish

Round R's clock is then the finish time of the client PLATO merged at round R.
Merges are matched to dispatches by walking the log's merge sequence and taking
each client's earliest not-yet-merged dispatch -- rather than assuming the merge
order is the ascending finish order. It should be, and `verify_merge_order()`
checks that it is; assuming it would hide a staleness-bound flush
(`plato/servers/base.py` step 3), which merges several clients at once and jumps
the clock.

Accuracy needs no adjustment: it is indexed by round, and rounds do not move.
Only the time axis is recomputed.

Usage:
    python3 clock_reconstruction.py --runs det1 det2
    python3 clock_reconstruction.py --runs det1 --out-dir stage2_evidence
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import tomllib
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEDASYNC = HERE.parent
RUNS_DIR = FEDASYNC / "runs" / "gpu_tiers_dataset"

# The three log lines the replay needs. See `read_events()` for why the FILE
# ORDER of these lines is the load-bearing part, not the round numbers.
ROUND_LINE = re.compile(r"Starting round (\d+)/(\d+)")
SELECT_LINE = re.compile(r"Selecting client #(\d+) for training")
MERGE_LINE = re.compile(
    r"Adding client #(\d+) to the list of clients for aggregation"
)

# Written into each CLIENT log by GpuTierLogCallback. It is the run's own record
# of the fleet it slept on, independent of any config file -- see
# verify_fleet_matches_run().
SLEEP_LINE = re.compile(
    r"\[GpuTierSleep\] round=-?\d+ client_id=(\d+) tier_id=\S+ "
    r"dataset_size=\d+ sleep_time=([0-9.]+) s"
)


@dataclass(frozen=True)
class Dispatch:
    index: int          # position in the dispatch sequence, 0-based
    round: int          # the round whose "Starting round" line precedes it
    client_id: int
    slot: int
    start: float
    finish: float


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def run_dir(name: str) -> Path:
    for candidate in (RUNS_DIR / name,
                      *(RUNS_DIR.glob(f"*_{name}"))):
        if candidate.exists():
            return candidate
    raise SystemExit(f"no run directory for {name!r} under {RUNS_DIR}")


def config_for(directory: Path) -> dict:
    """The TOML this run was ACTUALLY launched with.

    Prefers `<run_dir>/config_used.toml`, the copy `fedasync_seeded.py` writes at
    startup, and falls back to `configs/<name>.toml` only for runs made before
    that existed.

    The fallback is dangerous and is why the copy exists: `configs/` is
    regenerated whenever a generator constant changes, so a run made at one
    sigma would be reconstructed against a later sigma's fleet -- different sleep
    times, silently, with the reconstruction still producing plausible numbers.
    That happened once during Stage 2 and `verify_fleet_matches_run()` below is
    what caught it.
    """
    saved = directory / "config_used.toml"
    if saved.exists():
        return tomllib.loads(saved.read_text())
    path = HERE / "configs" / f"{directory.name}.toml"
    if not path.exists():
        raise SystemExit(
            f"no config at {directory / 'config_used.toml'} nor at {path}. The "
            f"reconstruction needs it for C, for the sleep CSV and for "
            f"comm_time; none of those may be guessed."
        )
    return tomllib.loads(path.read_text())


def server_log(directory: Path) -> str:
    logs = sorted(directory.glob("server_*.log"))
    if not logs:
        raise SystemExit(f"no server log under {directory}")
    if len(logs) > 1:
        raise SystemExit(
            f"{len(logs)} server logs under {directory}; the run directory was "
            f"reused and the dispatch order cannot be read unambiguously."
        )
    return logs[0].read_text(errors="replace")


def sleep_times(config: dict) -> dict[int, float]:
    path = Path(config["clients"]["simulation_distribution"]["csv_path"])
    with path.open(newline="") as fh:
        return {int(r["client_id"]): float(r["sleep_time"])
                for r in csv.DictReader(fh)}


def comm_time_seconds(config: dict) -> float:
    """The per-report communication time PLATO charges, from the CONFIG.

    Not hard-coded to zero. It IS zero today -- `comm_simulation = true` routes
    the payload through a file and sets `comm_time = 0`
    (`plato/servers/base.py`) -- but a config that turned that off would make a
    zero here silently wrong, and the error would look like a growing
    reconstruction gap rather than like a missing term.

    With `comm_simulation = false` PLATO MEASURES comm_time on the host, exactly
    like `processing_time`. There is then no configured value to reconstruct
    from, so this refuses rather than inventing one.

    Note this is separate from any comm latency baked into the sleep times
    themselves: `build_fleet.py --comm-bandwidth-mbps` adds C = D/B to every
    client's sleep, and that term is already inside `sleep_times.csv`. Adding it
    again here would double-count it.
    """
    clients = config.get("clients", {})
    if "comm_simulation" not in clients:
        raise SystemExit(
            "[clients] comm_simulation is absent from the config, so the "
            "communication term cannot be determined. Refusing to assume zero."
        )
    if clients["comm_simulation"] is True:
        return 0.0
    raise SystemExit(
        "[clients] comm_simulation = false: PLATO measures comm_time on the "
        "host, so there is no configured value to reconstruct from. This run's "
        "clock cannot be rebuilt by this script."
    )


# ---------------------------------------------------------------------------
# C3 -- reading the dispatch ORDER out of the log
# ---------------------------------------------------------------------------

def verify_fleet_matches_run(directory: Path, sleeps: dict[int, float]) -> int:
    """Cross-check the sleep CSV against what the run's clients actually slept.

    `GpuTierLogCallback` writes the scheduled sleep into each client log on every
    epoch:

        [GpuTierSleep] round=7 client_id=12 tier_id=orin dataset_size=2500
                       sleep_time=1.679216 s

    That is the run's own record of the fleet it used, independent of any config
    file, so it is the one thing that can prove the sleep times being replayed
    are the sleep times that were simulated. Without it a reconstruction against
    the wrong fleet produces entirely plausible numbers and says nothing.

    Returns the number of clients checked. Raises on any mismatch.
    """
    observed: dict[int, float] = {}
    for path in sorted(directory.glob("client_*.log")):
        for cid, sleep in SLEEP_LINE.findall(path.read_text(errors="replace")):
            observed[int(cid)] = float(sleep)

    if not observed:
        raise SystemExit(
            f"{directory} has no [GpuTierSleep] lines in its client logs, so the "
            f"fleet it ran on cannot be confirmed. Refusing to reconstruct "
            f"against an unverified sleep table."
        )

    wrong = {
        cid: (value, sleeps.get(cid))
        for cid, value in observed.items()
        if cid not in sleeps or abs(sleeps[cid] - value) > 1e-5
    }
    if wrong:
        sample = ", ".join(
            f"client {cid}: ran {ran:.6f} s, table says {table}"
            for cid, (ran, table) in list(wrong.items())[:3]
        )
        raise SystemExit(
            f"{directory.name} was run on a DIFFERENT fleet than the sleep table "
            f"being replayed ({len(wrong)} of {len(observed)} clients disagree). "
            f"{sample}. The reconstruction would be arithmetic on the wrong "
            f"numbers."
        )
    return len(observed)


def read_events(text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    """(dispatches, merges, rounds) as (round, client_id) in FILE ORDER.

    The sequence is what matters, and it comes from the order the lines appear
    in the single server log -- not from the round number, which cannot carry it:
    round 1 dispatches C clients (it fills every slot), so several dispatches
    share one round number and the round alone would leave their order ambiguous.

    `Starting round N/M` lines are used only to attach a round number to each
    event, for reporting.
    """
    dispatches: list[tuple[int, int]] = []
    merges: list[tuple[int, int]] = []
    current_round = 0
    total_rounds = 0

    for line in text.splitlines():
        match = ROUND_LINE.search(line)
        if match:
            current_round = int(match.group(1))
            total_rounds = int(match.group(2))
            continue
        match = SELECT_LINE.search(line)
        if match:
            dispatches.append((current_round, int(match.group(1))))
            continue
        match = MERGE_LINE.search(line)
        if match:
            merges.append((current_round, int(match.group(1))))

    return dispatches, merges, total_rounds


# ---------------------------------------------------------------------------
# The queue replay
# ---------------------------------------------------------------------------

def replay(
    dispatches: list[tuple[int, int]],
    sleeps: dict[int, float],
    concurrency: int,
    comm_time: float = 0.0,
) -> list[Dispatch]:
    """C slots; each dispatch takes the slot that frees earliest."""
    slots = [0.0] * concurrency
    out: list[Dispatch] = []
    for index, (rnd, client_id) in enumerate(dispatches):
        if client_id not in sleeps:
            raise SystemExit(
                f"client {client_id} was dispatched but is not in the sleep CSV; "
                f"the run and the fleet do not match."
            )
        slot = min(range(concurrency), key=lambda i: slots[i])
        start = slots[slot]
        finish = start + sleeps[client_id] + comm_time
        slots[slot] = finish
        out.append(Dispatch(index, rnd, client_id, slot, start, finish))
    return out


def matched_dispatches(
    schedule: list[Dispatch],
    merges: list[tuple[int, int]],
) -> list[Dispatch]:
    """The Dispatch each merge actually consumed, IN MERGE ORDER.

    Each merge takes that client's earliest dispatch not already consumed. A
    client can be dispatched many times over a run, so matching by client alone
    would be ambiguous; matching in order is not. `clock_per_round`,
    `staleness_per_merge` and `verify_merge_order` all read the same match, so
    there is exactly one place this pairing can drift out of sync with itself.
    """
    pending: dict[int, list[Dispatch]] = {}
    for item in schedule:
        pending.setdefault(item.client_id, []).append(item)
    for items in pending.values():
        items.reverse()          # pop() then takes the earliest

    out: list[Dispatch] = []
    for rnd, client_id in merges:
        queue = pending.get(client_id)
        if not queue:
            raise SystemExit(
                f"round {rnd} merged client {client_id}, which has no unconsumed "
                f"dispatch. The log's dispatch and merge sequences disagree."
            )
        out.append(queue.pop())
    return out


def clock_per_round(
    schedule: list[Dispatch],
    merges: list[tuple[int, int]],
) -> dict[int, float]:
    """Round -> reconstructed clock, from each merge's matched dispatch."""
    matched = matched_dispatches(schedule, merges)
    return {rnd: item.finish for (rnd, _client_id), item in zip(merges, matched)}


def staleness_per_merge(
    schedule: list[Dispatch],
    merges: list[tuple[int, int]],
) -> list[int]:
    """Report staleness of every merge, in merge order: merge_round - dispatch_round.

    This is exactly PLATO's own `client_staleness = self.current_round -
    starting_round` (`plato/servers/base.py:1168`, `:1219`, `:1246`) — the round
    a client was DISPATCHED at is the `Dispatch.round` this replay already
    carries, and the round it was MERGED at is the round half of the merge tuple
    -- both read from the same "Starting round N/M" bookkeeping PLATO itself
    uses to set `self.current_round`, so no new quantity is invented here.

    Chosen over parsing `[PolarisVi] ... actual=%.4f rounds` (the only place
    staleness previously appeared as text) because that line is written only by
    `PolarisViClientSelectionStrategy.on_reports_received`
    (`fedasync_polaris_vi_selection.py`), which the `baseline` arm never
    instantiates -- a log-text approach would have no answer for that arm.
    Cross-checked against it anyway, for the arms that DO log it: 0 mismatches
    over 200 merges of a finished run (`tests/test_clock_reconstruction.py`).

    Returns one value per entry in `merges`, so a caller can compute mean/max
    over EVERY report rather than one value per round -- a staleness-bound flush
    can merge several clients under one round number, and collapsing them into a
    dict keyed by round would silently drop all but the last.
    """
    matched = matched_dispatches(schedule, merges)
    return [rnd - item.round for (rnd, _client_id), item in zip(merges, matched)]


def verify_merge_order(
    schedule: list[Dispatch],
    merges: list[tuple[int, int]],
) -> tuple[bool, list[str]]:
    """Did PLATO merge in ascending reconstructed finish time?

    It should: the server pops the smallest (finish_time, client_id) off a heap
    every round (`plato/servers/base.py:1036-1047`). A violation means either the
    queue model is wrong, or a staleness-bound flush merged several clients at
    once, or `processing_time` reordered a near-tie. All three are worth seeing
    rather than smoothing over.
    """
    matched = matched_dispatches(schedule, merges)
    ordered = [item.finish for item in matched]
    problems = []
    for i in range(1, len(ordered)):
        if ordered[i] < ordered[i - 1] - 1e-9:
            problems.append(
                f"merge {i} (round {merges[i][0]}, client {merges[i][1]}) "
                f"finishes at {ordered[i]:.6f} s, before merge {i - 1} at "
                f"{ordered[i - 1]:.6f} s"
            )
    return not problems, problems


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def results_csv(directory: Path) -> Path:
    matches = sorted(directory.rglob("results/*.csv"))
    if not matches:
        raise SystemExit(f"no results CSV under {directory}")
    return matches[0]


def plato_curve(directory: Path) -> list[dict]:
    with results_csv(directory).open(newline="") as fh:
        return list(csv.DictReader(fh))


def run_metadata(directory: Path, config: dict) -> dict:
    """(experiment, arm, seed) -- "experiment" identifies the CELL, shared
    across every arm run on it; "arm" is the one thing that cell varies.

    Uses `make_configs.arm_of` / `cell_of` rather than re-deriving the split
    here: those already handle `gpu_oracle` (which contains an underscore) and
    every alpha value, because a run directory's name IS
    `make_configs.run_name()`'s output by construction. A second, ad hoc parse
    of the same string is exactly the kind of thing that quietly breaks the day
    alpha stops being 0.8 everywhere -- `"_a0p"` is a literal fragment of THAT
    one alpha's own tag and matches no other.
    """
    import make_configs  # noqa: PLC0415 -- same-folder module, no cycle

    stem = directory.name
    fake_path = Path(f"{stem}.toml")   # both helpers only read the stem
    return {
        "experiment": make_configs.cell_of(fake_path),
        "arm": make_configs.arm_of(fake_path),
        "seed": config["clients"]["random_seed"],
    }


def reconstruct(directory: Path) -> dict:
    config = config_for(directory)
    concurrency = int(config["clients"]["per_round"])
    sleeps = sleep_times(config)
    comm = comm_time_seconds(config)

    verified = verify_fleet_matches_run(directory, sleeps)

    text = server_log(directory)
    dispatches, merges, total_rounds = read_events(text)
    schedule = replay(dispatches, sleeps, concurrency, comm)
    clocks = clock_per_round(schedule, merges)
    staleness = staleness_per_merge(schedule, merges)
    ordered_ok, order_problems = verify_merge_order(schedule, merges)

    rows = []
    meta = run_metadata(directory, config)
    for row in plato_curve(directory):
        rnd = int(row["round"])
        if rnd not in clocks:
            continue
        rows.append({
            **meta,
            "round": rnd,
            "sim_time_plato": float(row["elapsed_time"]),
            "sim_time_reconstructed": clocks[rnd],
            "accuracy": float(row["accuracy"]),
        })

    return {
        "directory": directory,
        "config": config,
        "concurrency": concurrency,
        "comm_time": comm,
        "total_rounds": total_rounds,
        "dispatches": dispatches,
        "merges": merges,
        "schedule": schedule,
        "rows": rows,
        "staleness_per_merge": staleness,
        "clients_verified": verified,
        "merge_order_ok": ordered_ok,
        "merge_order_problems": order_problems,
    }


def write_curves(results: list[dict], out: Path) -> Path:
    fieldnames = ["experiment", "arm", "seed", "round",
                  "sim_time_plato", "sim_time_reconstructed", "accuracy"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for row in result["rows"]:
                writer.writerow({k: row[k] for k in fieldnames})
    return out


def gap_series(result: dict) -> list[tuple[int, float]]:
    """(round, plato - reconstructed). This is the accumulated processing_time."""
    return [(r["round"], r["sim_time_plato"] - r["sim_time_reconstructed"])
            for r in result["rows"]]


def summarise(result: dict) -> str:
    lines = []
    add = lines.append
    gaps = gap_series(result)
    final_round, final_gap = gaps[-1]
    final = result["rows"][-1]

    add(f"{result['directory'].name}")
    add(f"  C = {result['concurrency']},  comm_time = {result['comm_time']:g} s "
        f"(from [clients] comm_simulation)")
    add(f"  dispatches {len(result['dispatches'])}, merges {len(result['merges'])}, "
        f"rounds {result['total_rounds']}")
    add(f"  round {final_round}:  plato {final['sim_time_plato']:.6f} s   "
        f"reconstructed {final['sim_time_reconstructed']:.6f} s")
    add(f"  gap {final_gap:+.6f} s  "
        f"({100 * final_gap / final['sim_time_reconstructed']:+.4f}% of the "
        f"reconstructed clock)")
    add(f"  per-report overhead implied: "
        f"{final_gap / len(result['merges']) * result['concurrency']:.6f} s")

    values = [g for _, g in gaps]
    monotone = all(b >= a - 1e-9 for a, b in zip(values, values[1:]))
    negative = [(r, g) for r, g in gaps if g < 0]
    add(f"  positive throughout: {'YES' if not negative else 'NO'}")
    add(f"  monotonically increasing: {'YES' if monotone else 'NO'}")
    if not monotone:
        drops = [(gaps[i][0], values[i] - values[i - 1])
                 for i in range(1, len(values)) if values[i] < values[i - 1] - 1e-9]
        add(f"    {len(drops)} decrease(s); largest {min(d for _, d in drops):+.6f} s "
            f"at round {min(drops, key=lambda d: d[1])[0]}")
    increments = [b - a for a, b in zip(values, values[1:])]
    if increments:
        add(f"  per-round increment: median {statistics.median(increments):+.6f} s, "
            f"min {min(increments):+.6f}, max {max(increments):+.6f}")
    add(f"  merge order ascending in reconstructed time: "
        f"{'YES' if result['merge_order_ok'] else 'NO'}")
    for problem in result["merge_order_problems"][:3]:
        add(f"    {problem}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write accuracy_curves_reconstructed.csv "
                             "(default: beside the first run's results).")
    args = parser.parse_args()

    results = [reconstruct(run_dir(name)) for name in args.runs]
    out = args.out or (HERE / "results" / "accuracy_curves_reconstructed.csv")
    write_curves(results, out)

    for result in results:
        print(summarise(result))
        print()
    print(f"wrote {out}")

    if len(results) == 2:
        a, b = (r["rows"][-1] for r in results)
        for label, key in (("plato", "sim_time_plato"),
                           ("reconstructed", "sim_time_reconstructed")):
            spread = abs(a[key] - b[key]) / min(a[key], b[key])
            print(f"{label:14s} spread between the two runs: {100 * spread:.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
