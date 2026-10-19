"""
Generate tiers.csv variants that sweep the hardware heterogeneity gap.

The gap G is the TFLOPS ratio between the fastest and slowest tier
(the real Jetson lineup 10 / 78 / 150 has G = 15).

Construction, for each requested G:
    xavier = 10.0                    (anchored to the real Xavier spec)
    thor   = 10.0 * G
    orin   = midpoint(xavier, thor)  (see --mid-mode)

Anchoring the slowest tier keeps the straggler's sleep time identical across
every setting, so any change in observed staleness comes from the gap alone.
(Anchoring in TFLOPS only survives into seconds if generate_sleep_times.py is
also anchored on xavier — see the sweep recipe in README_gpu_tier.md.)

--mid-mode picks where Orin sits:
    arithmetic  orin = (xavier + thor) / 2   — matches the real lineup, where
                Orin (78) sits almost exactly at the mean of Xavier and Thor
                (80).  Careful: in the TIME domain this pins Orin at
                t_orin = t_thor * 2G/(1+G), i.e. ~2x Thor for every G >= 10,
                so the mid tier collapses onto the fast tier and the sweep
                degenerates into a two-tier experiment.
    geometric   orin = sqrt(xavier * thor)   — the three tiers are then evenly
                log-spaced in BOTH TFLOPS and time (t ratios 1 : 1/sqrt(G) :
                1/G), so Orin stays a genuine middle tier at every gap.

Every other column (year, arch, CPU spec, eta_nominal, noise_sigma) is copied
unchanged from the baseline tiers.csv.

Usage:
    python make_tier_gaps.py
    python make_tier_gaps.py --gaps 1 10 100 --outdir input
    python make_tier_gaps.py --mid-mode geometric --prefix tiers_geo
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
DEFAULT_BASE_CSV = _MODULE_DIR / "input" / "tiers.csv"
DEFAULT_OUTDIR = _MODULE_DIR / "input"
DEFAULT_GAPS = [1, 5, 10, 15, 20, 25, 30, 35, 40]

SLOW_TIER = "xavier"
MID_TIER = "orin"
FAST_TIER = "thor"
SLOW_TFLOPS = 10.0


def tflops_for_gap(gap: float, mid_mode: str = "arithmetic") -> dict[str, float]:
    """TFLOPS per tier for a given fastest/slowest ratio."""
    fast = SLOW_TFLOPS * gap
    if mid_mode == "geometric":
        mid = math.sqrt(SLOW_TFLOPS * fast)
    elif mid_mode == "arithmetic":
        mid = (SLOW_TFLOPS + fast) / 2.0
    else:
        raise ValueError(f"Unknown mid_mode '{mid_mode}'.")
    return {
        SLOW_TIER: SLOW_TFLOPS,
        MID_TIER: mid,
        FAST_TIER: fast,
    }


def write_tiers_csv(
    base_rows: list[dict[str, str]],
    fieldnames: list[str],
    gap: float,
    output_path: Path,
    mid_mode: str = "arithmetic",
) -> Path:
    """Copy the baseline rows, overriding only gpu_tflops_fp16."""
    tflops = tflops_for_gap(gap, mid_mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in base_rows:
            new_row = dict(row)
            new_row["gpu_tflops_fp16"] = f"{tflops[row['tier_id']]:g}"
            writer.writerow(new_row)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--gaps", type=float, nargs="+", default=DEFAULT_GAPS)
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--mid-mode",
        choices=("arithmetic", "geometric"),
        default="arithmetic",
        help="Where the mid tier sits between slowest and fastest (see module docstring).",
    )
    parser.add_argument(
        "--prefix",
        default="tiers",
        help="Output basename prefix; files are <prefix>_gapNN.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    with args.base_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        base_rows = list(reader)

    missing = {SLOW_TIER, MID_TIER, FAST_TIER} - {row["tier_id"] for row in base_rows}
    if missing:
        raise ValueError(f"{args.base_csv} is missing tier(s): {sorted(missing)}")

    for gap in args.gaps:
        out = args.outdir / f"{args.prefix}_gap{int(gap):02d}.csv"
        write_tiers_csv(base_rows, fieldnames, gap, out, args.mid_mode)
        tflops = tflops_for_gap(gap, args.mid_mode)
        print(
            f"gap={int(gap):>3}x  "
            + "  ".join(f"{t}={tflops[t]:g}" for t in (SLOW_TIER, MID_TIER, FAST_TIER))
            + f"  -> {out.name}"
        )


if __name__ == "__main__":
    main()
