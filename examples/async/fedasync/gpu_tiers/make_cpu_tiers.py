"""
Generate CPU-based tiers.csv variants (low-heterogeneity counterpart to the GPU sweep).

Following the convention already used in this example (see test_t_round.py,
"Test 6"), CPU throughput is written into the gpu_tflops_fp16 column, so the
existing sleep-time formula works unchanged.  The column name is a misnomer for
these files, but keeping the schema identical means the same loader, the same
formula, and the same anchoring logic apply to both.

--metric picks what "CPU throughput" means:

    freq          cpu_freq_ghz / 1000            (default)
                  Clock speed only, matching the paper this arm reproduces.
                  Xavier and Orin both clock at 2.00 GHz, so they come out
                  IDENTICAL; only Thor (2.60 GHz) is faster, by 1.30x.

    cores_x_freq  cpu_cores * cpu_freq_ghz / 1000
                  Also credits core count, which is what separates Orin (12
                  cores) from Xavier (8 cores) and widens the spread to 1.95x.
                  Assumes training scales perfectly across cores.

Two files are produced:

  tiers_cpu.csv          Derived straight from the CPU columns of the baseline
                         tiers.csv, per --metric (1.30x spread by default).

  tiers_cpu_gap20pct.csv Forced to an exact 1.2x (20%) spread, using the same
                         construction as make_tier_gaps.py: the slowest tier is
                         anchored, the fastest is slowest * gap, and the middle
                         tier sits at the arithmetic midpoint.

Absolute magnitudes here do not matter: generate_sleep_times.py is invoked with
--anchor-tier/--anchor-sleep, which rescales everything so only the ratios
between tiers survive.

Usage:
    python make_cpu_tiers.py
    python make_cpu_tiers.py --gap 1.5
    python make_cpu_tiers.py --metric cores_x_freq
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
DEFAULT_BASE_CSV = _MODULE_DIR / "input" / "tiers.csv"
DEFAULT_OUTDIR = _MODULE_DIR / "input"

SLOW_TIER = "xavier"
MID_TIER = "orin"
FAST_TIER = "thor"


def cpu_throughput(row: dict[str, str], metric: str = "freq") -> float:
    """CPU-throughput proxy for one tier row (see --metric in the docstring)."""
    if metric == "freq":
        return float(row["cpu_freq_ghz"]) / 1000.0
    if metric == "cores_x_freq":
        return float(row["cpu_cores"]) * float(row["cpu_freq_ghz"]) / 1000.0
    raise ValueError(f"Unknown metric '{metric}'.")


def forced_gap_throughput(slowest: float, gap: float) -> dict[str, float]:
    """Same shape as make_tier_gaps.py: anchor slowest, midpoint in the middle."""
    fastest = slowest * gap
    return {
        SLOW_TIER: slowest,
        MID_TIER: (slowest + fastest) / 2.0,
        FAST_TIER: fastest,
    }


def write_tiers_csv(
    base_rows: list[dict[str, str]],
    fieldnames: list[str],
    throughput: dict[str, float],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in base_rows:
            new_row = dict(row)
            new_row["gpu_tflops_fp16"] = f"{throughput[row['tier_id']]:g}"
            writer.writerow(new_row)
    return output_path


def _report(label: str, throughput: dict[str, float], path: Path) -> None:
    ratio = throughput[FAST_TIER] / throughput[SLOW_TIER]
    specs = "  ".join(f"{t}={throughput[t]:g}" for t in (SLOW_TIER, MID_TIER, FAST_TIER))
    print(f"{label:<14} {specs}  spread={ratio:.2f}x  -> {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--gap", type=float, default=1.2, help="Forced fast/slow ratio.")
    parser.add_argument(
        "--metric",
        choices=("freq", "cores_x_freq"),
        default="freq",
        help="What counts as CPU throughput (see module docstring).",
    )
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    with args.base_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        base_rows = list(reader)

    # 1. Straight from the CPU spec columns.
    spec = {row["tier_id"]: cpu_throughput(row, args.metric) for row in base_rows}
    out = write_tiers_csv(base_rows, fieldnames, spec, args.outdir / "tiers_cpu.csv")
    _report(args.metric, spec, out)

    # 2. Forced to the requested gap, anchored on the slowest tier's spec value.
    forced = forced_gap_throughput(spec[SLOW_TIER], args.gap)
    pct = round((args.gap - 1.0) * 100)
    out = write_tiers_csv(
        base_rows, fieldnames, forced, args.outdir / f"tiers_cpu_gap{pct}pct.csv"
    )
    _report(f"forced {args.gap:g}x", forced, out)


if __name__ == "__main__":
    main()
