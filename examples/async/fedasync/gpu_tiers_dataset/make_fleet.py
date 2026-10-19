"""
Build the client -> tier map (the fleet's COMPOSITION) for a given fleet size.

Moved here from `../surgical/make_fleet.py` when `gpu_tiers_dataset/` became the
single source of truth for tier specs and fleets.  The layout it produces is
unchanged, and `--total 20 --mix 50 35 15` still reproduces
`../separability/input/clients_n20.csv` byte for byte (asserted by
`tests/test_make_fleet.py`).

Layout: contiguous ID blocks, slowest tier first.

    50/35/15 over 20 clients  ->  1-10 xavier, 11-17 orin, 18-20 thor

Two properties of that layout are load-bearing (see gpu_tiers/HANDOFF.md § 4d):

  * PLATO derives a client's data shard from `random_seed` and the client ID, so
    contiguous blocks mean client i keeps its data when the mix changes -- the mix
    then changes which SPEED is attached to a fixed dataset, instead of
    confounding composition with which shards landed on the fast tier.
  * Blocks make fleets nested: every thor client at 15 % is still thor at 50 %.

Fleets that do not divide evenly
--------------------------------
The version in `../surgical/` REFUSED a mix that did not divide the fleet
exactly, and the reason it gave is still correct:

    silently turning 50/35/15 into 10/7/3 when the user asked for something that
    cannot be built would put a different fleet on disk than the one named in
    every filename, log line and figure downstream.

`--total 30 --mix 50 35 15` is exactly that case: 15 / 10.5 / 4.5.  Refusing it
blocks the planned fleet-size sweep, so this version builds it instead -- but it
removes the objection rather than ignoring it.  The remainder rule is:

    every tier but the fastest takes floor(total * pct / 100);
    the FASTEST tier takes whatever is left over.

    30 clients at 50/35/15  ->  15 xavier, 10 orin, 5 thor  (= 50 / 33.3 / 16.7)

and `fleet_tag()` names the fleet by its REALISED COUNTS (`n30_x15-o10-t5`), not
by the requested percentages.  Nothing downstream can then be mislabelled: the
directory, the manifest and every figure built from them carry the fleet that is
actually on disk.

The remainder goes to the fastest tier because that is the direction chosen for
the fleet-size sweep; it is a deliberate asymmetry and it is NOT free.  At N = 30
thor holds 16.7 % of the fleet against 15.0 % at N = 20 and N = 40, and thor is
the fastest tier, so E[T] at N = 30 is slightly lower for a reason that has
nothing to do with fleet size.  A sweep along N therefore moves two things at
once in that cell, and the realised mix is recorded in the fleet manifest so the
analysis can see it.

Usage:
    python3 make_fleet.py --total 20 --mix 50 35 15 -o fleets/.../clients.csv
    python3 make_fleet.py --total 30 --mix 50 35 15 --check   # print, write nothing
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

TIERS = ("xavier", "orin", "thor")   # slowest -> fastest, the block order
SCHEMA_VERSION = 1


def counts_for(total: int, mix: tuple[int, int, int]) -> tuple[int, int, int]:
    """Client counts per tier, as exact integers.

    The mix is given in percent and must sum to 100.  Tiers are filled slowest
    first with `floor(total * pct / 100)`, and the FASTEST tier absorbs the
    remainder, so every fleet size is buildable and the realised mix is whatever
    the counts say it is.  See the module docstring for why that is safe here and
    what it costs.

    Raises SystemExit if the mix does not sum to 100, or if the remainder rule
    would give a tier a negative count (which can only happen for a mix that
    sums to 100 but has a tier at more than 100 %, i.e. a negative entry).
    """
    if sum(mix) != 100:
        raise SystemExit(f"--mix must sum to 100, got {mix} = {sum(mix)}")
    if any(pct < 0 for pct in mix):
        raise SystemExit(f"--mix entries must be non-negative, got {mix}")
    if total <= 0:
        raise SystemExit(f"--total must be positive, got {total}")

    counts = [total * pct // 100 for pct in mix[:-1]]
    counts.append(total - sum(counts))
    if counts[-1] < 0:
        raise SystemExit(
            f"--mix {mix} over {total} clients leaves {counts[-1]} for "
            f"{TIERS[-1]}; the slower tiers already claim more than the fleet."
        )
    return tuple(counts)  # type: ignore[return-value]


def exact_counts(total: int, mix: tuple[int, int, int]) -> tuple[float, float, float]:
    """What the percentages ask for, before the remainder rule rounds them."""
    return tuple(total * pct / 100 for pct in mix)  # type: ignore[return-value]


def divides_evenly(total: int, mix: tuple[int, int, int]) -> bool:
    """True when the requested mix needs no remainder rule at all."""
    return all(float(v).is_integer() for v in exact_counts(total, mix))


def fleet_tag(counts: tuple[int, int, int]) -> str:
    """`n30_x15-o10-t5` -- the fleet named by what it actually contains.

    Counts, not percentages: 15/10/5 over 30 rounds to 50/33/17, and a reader
    cannot recover the counts from that.  The first letter of each tier is
    enough while TIERS stays xavier/orin/thor, and this function is the single
    place that would change if a tier were ever added.
    """
    total = sum(counts)
    body = "-".join(f"{tier[0]}{n}" for tier, n in zip(TIERS, counts))
    return f"n{total}_{body}"


def rows_for(counts: tuple[int, int, int]) -> list[dict]:
    rows, cid = [], 1
    for tier, n in zip(TIERS, counts):
        for _ in range(n):
            rows.append(
                {"client_id": cid, "tier_id": tier, "schema_version": SCHEMA_VERSION}
            )
            cid += 1
    return rows


def write_fleet_csv(output: Path, rows: list[dict]) -> Path:
    """Write a `client_id,tier_id,schema_version` CSV, creating parent dirs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["client_id", "tier_id", "schema_version"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return output


def describe(total: int, counts: tuple[int, int, int]) -> str:
    bounds = []
    lo = 1
    for n in counts:
        bounds.append((lo, lo + n - 1))
        lo += n
    return "fleet of {}: ".format(total) + ", ".join(
        f"{tier} {n} ({100 * n / total:.1f}%) ids {lo}-{hi}"
        for tier, n, (lo, hi) in zip(TIERS, counts, bounds)
        if n
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--total", type=int, required=True, help="Fleet size.")
    parser.add_argument(
        "--mix", type=int, nargs=3, metavar=("XAVIER", "ORIN", "THOR"),
        default=[50, 35, 15], help="Percent per tier, slowest first (default: 50 35 15).",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("input/clients.csv"))
    parser.add_argument("--check", action="store_true", help="Print only; write nothing.")
    args = parser.parse_args()

    mix = tuple(args.mix)
    counts = counts_for(args.total, mix)
    print(describe(args.total, counts))
    print(f"tag: {fleet_tag(counts)}")

    if not divides_evenly(args.total, mix):
        exact = exact_counts(args.total, mix)
        print(
            "note: {} does not divide {} evenly ({}); the remainder went to {}. "
            "The realised mix is {} -- name and manifest carry the counts, not "
            "the requested percentages.".format(
                "/".join(str(p) for p in mix),
                args.total,
                "/".join(f"{v:g}" for v in exact),
                TIERS[-1],
                "/".join(f"{100 * n / args.total:.1f}" for n in counts),
            )
        )

    if args.check:
        print("--check given; nothing written.")
        return 0

    out = write_fleet_csv(args.output, rows_for(counts))
    print(f"wrote {out} ({args.total} clients)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
