"""
Generate clients.csv variants that change the FLEET COMPOSITION, not the speeds.

The gap sweep holds the mix at 50 xavier / 35 orin / 15 thor and moves the tier
speeds.  This is the other axis: the tier speeds stay pinned at one gap G, and
what moves is how many clients sit in each tier.

    input/clients.csv                 50 / 35 / 15   (the historical fleet)
    input/clients_m70_20_10.csv       70 / 20 / 10   straggler-heavy
    input/clients_m34_33_33.csv       34 / 33 / 33   balanced
    input/clients_m15_35_50.csv       15 / 35 / 50   fast-heavy (mirror of base)

Only the client -> tier map changes.  tiers_geo_gap<G>.csv is untouched, so a
xavier client sleeps the same 20 s in every mix and `generate_sleep_times.py`
re-run against a new clients.csv writes the same three sleep values to a
different set of client IDs.

Construction: contiguous blocks in client-ID order
--------------------------------------------------
Clients 1..n_xavier are xavier, the next n_orin are orin, the rest thor — the
same layout `input/clients.csv` already has (1-50 / 51-85 / 86-100).  Two
properties matter and both come from keeping it contiguous:

* **The data partition never moves.**  PLATO derives client i's shard from
  `random_seed` and the client ID alone, so client 7 trains on the identical 500
  images in every mix.  Changing the mix changes which *speed* is attached to a
  fixed dataset, which is the only way the comparison is about composition
  rather than about data.
* **The mixes are nested.**  Promoting a client is monotone in ID: every thor
  client of a 15 %-thor fleet is still thor at 50 %, and the clients that change
  tier between two mixes are exactly the ones between their cut points.  A
  random re-draw per mix would move a different arbitrary subset each time, and
  any difference between two mixes would then confound composition with which
  particular shards happened to land on the fast tier.

The cost is that tier membership correlates with client ID, hence with shard
index — but that correlation is already in `input/clients.csv` and every run of
the gap sweep, so keeping it is what makes the mix sweep comparable to those.

Usage:
    python3 make_mixes.py                          # the four sweep mixes
    python3 make_mixes.py --mixes 50/35/15 85/10/5
    python3 make_mixes.py --list                   # names only, write nothing
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
DEFAULT_BASE_CSV = _MODULE_DIR / "input" / "clients.csv"
DEFAULT_OUTDIR = _MODULE_DIR / "input"

TIER_ORDER = ("xavier", "orin", "thor")  # slowest -> fastest
SCHEMA_VERSION = "1"

# The four cells of the mix sweep.  50/35/15 is the historical fleet and is
# written out under its own explicit name rather than reusing clients.csv: the
# sweep's runs all carry a mix label, and a cell that silently pointed at a
# differently-named file would be the one cell nobody could check by reading the
# run directory.
DEFAULT_MIXES = [(50, 35, 15), (70, 20, 10), (34, 33, 33), (15, 35, 50)]


def mix_label(mix: tuple[int, int, int]) -> str:
    """`(70, 20, 10)` -> `m70_20_10` — the suffix used by every generated file."""
    return "m" + "_".join(str(n) for n in mix)


def parse_mix(text: str) -> tuple[int, int, int]:
    """Accept `70/20/10`, `70,20,10` or `m70_20_10`."""
    cleaned = text.removeprefix("m").replace("/", " ").replace(",", " ").replace("_", " ")
    parts = cleaned.split()
    if len(parts) != len(TIER_ORDER):
        raise argparse.ArgumentTypeError(
            f"'{text}' does not name {len(TIER_ORDER)} tiers ({'/'.join(TIER_ORDER)})."
        )
    try:
        counts = tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' has a non-integer count.") from None
    if any(n < 0 for n in counts):
        raise argparse.ArgumentTypeError(f"'{text}' has a negative count.")
    return counts  # type: ignore[return-value]


def client_tiers(mix: tuple[int, int, int]) -> list[str]:
    """Tier of client 1, 2, … in contiguous blocks (see the module docstring)."""
    tiers: list[str] = []
    for tier, count in zip(TIER_ORDER, mix):
        tiers.extend([tier] * count)
    return tiers


def write_clients_csv(mix: tuple[int, int, int], total: int, outdir: Path) -> Path:
    if sum(mix) != total:
        raise SystemExit(
            f"mix {mix} sums to {sum(mix)}, but the fleet has {total} clients — "
            f"a short mix would leave clients with no tier and PLATO would raise "
            f"on the first dispatch to one of them."
        )
    path = outdir / f"clients_{mix_label(mix)}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["client_id", "tier_id", "schema_version"])
        for client_id, tier in enumerate(client_tiers(mix), start=1):
            writer.writerow([client_id, tier, SCHEMA_VERSION])
    return path


def base_mix(base_csv: Path) -> tuple[tuple[int, int, int], int]:
    """The mix and client count of an existing clients.csv."""
    with base_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    counts = {t: 0 for t in TIER_ORDER}
    for row in rows:
        counts[row["tier_id"]] += 1
    return tuple(counts[t] for t in TIER_ORDER), len(rows)  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--mixes",
        nargs="+",
        type=parse_mix,
        default=DEFAULT_MIXES,
        help="xavier/orin/thor counts, e.g. 70/20/10 (default: the four sweep mixes).",
    )
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--list", action="store_true", help="Print the labels; write nothing.")
    args = parser.parse_args()

    existing, total = base_mix(args.base_csv)

    if args.list:
        for mix in args.mixes:
            print(mix_label(mix))
        return 0

    print(f"{args.base_csv.name}: {'/'.join(str(n) for n in existing)} over {total} clients\n")
    print(f"{'file':<28}{'xavier':>8}{'orin':>8}{'thor':>8}   fast share")
    for mix in args.mixes:
        path = write_clients_csv(mix, total, args.outdir)
        fast = (mix[1] + mix[2]) / total
        marker = "   <- same as clients.csv" if mix == existing else ""
        print(
            f"{path.name:<28}{mix[0]:>8}{mix[1]:>8}{mix[2]:>8}   "
            f"{fast:.0%} non-straggler{marker}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
