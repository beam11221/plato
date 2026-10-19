"""
Multi-seed aggregation for the gpu/cpu/baseline heterogeneity sweep.

`plot_time_to_accuracy.py` answers the sweep's question for ONE seed; its
``RUN_DIR_RE`` does not even match the ``_s<seed>`` directories.  This module adds
the seed axis and nothing else — the crossing logic, the evaluated-row
subsampling and the effective-gap calculation are imported from that script and
from ``summarize_sweep.py`` rather than re-derived, so a number here and the same
number there always agree.

Seed layout (HANDOFF § 4c)
--------------------------
Seed 8888 lives in ``runs/gap_sweep`` with no suffix and 10 000 rounds; seeds 1234
and 5678 live in ``runs/gap_sweep_ms`` with a ``_s<seed>`` suffix and 5 000 rounds.
The configs are otherwise byte-identical (verified: the only diff is ``rounds``),
and both metrics here are prefix properties, so the longer run is truncated to the
common horizon rather than excluded.

Time is PLATO's **simulated** clock.  ``figures/clean_time/`` only covers the
seed-8888 runs, so the reported ``elapsed_time`` is used for every seed — mixing
the two sources across seeds would put a host-cost correction on one third of the
sample and call the result an average.

Two metrics
-----------
time-to-accuracy   hours to the first smoothed crossing of a target, per seed,
                   then the mean over seeds.  A crossing is a prefix property, so
                   a seed that ran longer is not advantaged.
time budget        smoothed accuracy at a fixed simulated-hour budget.  Robust
                   where the crossing is not (``fixed_budget.py`` docstring): a
                   window of evaluations decides it, not one, and there is no
                   division by a vanishing slope on the plateau.

Only cells with every requested seed present are aggregated; partial cells are
reported separately rather than averaged over a different n than their neighbours.

Usage:
    python3 multiseed.py                       # tables
    python3 multiseed.py --plots               # tables + figures/multiseed/
    python3 multiseed.py --targets 50 55 60
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from plot_time_to_accuracy import (
    ARM_COLOR,
    ARM_ORDER,
    DEFAULT_CLIENTS_CSV,
    DEFAULT_SLEEP_DIR,
    effective_gap,
    first_crossing,
    load_curve,
)

_MODULE_DIR = Path(__file__).resolve().parent
_RUNS = _MODULE_DIR.parent / "runs"

# Seed 8888 is the original single-seed sweep; the other two are the _ms sweep.
SEED_LOCATION = {
    8888: (_RUNS / "gap_sweep", ""),
    1234: (_RUNS / "gap_sweep_ms", "_s1234"),
    5678: (_RUNS / "gap_sweep_ms", "_s5678"),
}
DEFAULT_SEEDS = [8888, 1234, 5678]
GAPS = ["01", "10", "20", "30", "40"]
LOG_DIR = _MODULE_DIR / "logs"


@dataclass
class Cell:
    """One (arm, gap) cell across seeds."""

    arm: str
    gap: str
    curves: dict[int, pd.DataFrame] = field(default_factory=dict)
    running: set[int] = field(default_factory=set)

    @property
    def seeds(self) -> list[int]:
        return sorted(self.curves)

    @property
    def horizon(self) -> float:
        """Simulated hours every seed of this cell covers."""
        return min(float(df["hours"].iloc[-1]) for df in self.curves.values())


def _results_csv(run_dir: Path) -> Path | None:
    cands = sorted(
        (run_dir / "results").glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    cands = [p for p in cands if not p.stem.endswith(("_binned", "_accuracy_bins"))]
    return cands[0] if cands else None


def collect(seeds: list[int], arms: list[str], gaps: list[str], smooth: int) -> dict:
    cells: dict[tuple[str, str], Cell] = {}
    for arm in arms:
        for gap in gaps:
            cell = Cell(arm=arm, gap=gap)
            for seed in seeds:
                root, suffix = SEED_LOCATION[seed]
                name = f"{arm}_xa20_gap{gap}{suffix}"
                run_dir = root / name
                if not run_dir.is_dir():
                    continue
                csv_path = _results_csv(run_dir)
                if csv_path is None:
                    continue
                # stop_at=None: truncation to a common horizon happens per cell,
                # against the other seeds, not against a fixed accuracy.
                df = load_curve(csv_path, stop_at=None, smooth=smooth)
                if len(df) < 50:
                    continue
                cell.curves[seed] = df
                if not (LOG_DIR / f"{name}.done").exists():
                    cell.running.add(seed)
            if cell.curves:
                cells[(arm, gap)] = cell
    return cells


def tta_table(cells: dict, targets: list[float], seeds: list[int]) -> pd.DataFrame:
    rows = []
    for (arm, gap), cell in cells.items():
        for target in targets:
            per_seed = {}
            for seed, df in cell.curves.items():
                hit = first_crossing(df, target)
                per_seed[seed] = None if hit is None else hit[1] / 3600.0
            got = [v for v in per_seed.values() if v is not None]
            rows.append(
                {
                    "arm": arm,
                    "gap": gap,
                    "target": target,
                    "n_seeds": len(cell.curves),
                    "n_crossed": len(got),
                    "complete": len(cell.curves) == len(seeds),
                    "mean_hours": float(np.mean(got)) if got else None,
                    "std_hours": float(np.std(got, ddof=1)) if len(got) > 1 else None,
                    "min_hours": float(np.min(got)) if got else None,
                    "max_hours": float(np.max(got)) if got else None,
                    **{f"h_s{s}": per_seed.get(s) for s in seeds},
                    "running": ",".join(str(s) for s in sorted(cell.running)) or "",
                }
            )
    return pd.DataFrame(rows)


def _acc_at(df: pd.DataFrame, hours: float) -> float | None:
    if hours > float(df["hours"].iloc[-1]):
        return None
    return float(np.interp(hours, df["hours"].to_numpy(), df["smooth_pct"].to_numpy()))


def budget_table(cells: dict, fractions: list[float], seeds: list[int]) -> pd.DataFrame:
    """Accuracy at fixed simulated-time budgets.

    The budget grid is per GAP, not per cell: comparing arms means reading them at
    the same clock time, so the horizon is the shortest any (arm, seed) of that gap
    covers.  Otherwise the fastest arm is scored at a budget the slowest never saw.
    """
    rows = []
    by_gap: dict[str, list[Cell]] = {}
    for (_, gap), cell in cells.items():
        by_gap.setdefault(gap, []).append(cell)

    for gap, gap_cells in by_gap.items():
        horizon = min(c.horizon for c in gap_cells)
        for frac in fractions:
            budget = horizon * frac
            for cell in gap_cells:
                per_seed = {s: _acc_at(df, budget) for s, df in cell.curves.items()}
                got = [v for v in per_seed.values() if v is not None]
                rows.append(
                    {
                        "arm": cell.arm,
                        "gap": gap,
                        "budget_hours": budget,
                        "frac": frac,
                        "n_seeds": len(got),
                        "complete": len(cell.curves) == len(seeds),
                        "mean_acc": float(np.mean(got)) if got else None,
                        "std_acc": float(np.std(got, ddof=1)) if len(got) > 1 else None,
                        **{f"a_s{s}": per_seed.get(s) for s in seeds},
                    }
                )
    return pd.DataFrame(rows)


def eff_gaps(gaps: list[str], sleep_dir: Path, clients_csv: Path) -> dict[str, float]:
    out = {}
    for gap in gaps:
        try:
            val = effective_gap(int(gap), 20, 1.0, sleep_dir, clients_csv)
        except Exception:
            val = None
        if val is not None:
            out[gap] = val
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--arms", nargs="+", default=ARM_ORDER, choices=ARM_ORDER)
    p.add_argument("--gaps", nargs="+", default=GAPS)
    p.add_argument("--targets", nargs="+", type=float, default=[50.0, 55.0])
    p.add_argument("--smooth", type=int, default=25, help="evaluations in the rolling median")
    p.add_argument("--fractions", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--out-dir", type=Path, default=_MODULE_DIR / "figures" / "multiseed")
    p.add_argument("--plots", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cells = collect(args.seeds, args.arms, args.gaps, args.smooth)
    if not cells:
        print("no runs found", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"seeds requested: {args.seeds}\n")
    print("coverage (seeds present per cell; * = still running)")
    print(f"  {'arm':<9}" + "".join(f"{g:>8}" for g in args.gaps))
    for arm in args.arms:
        line = f"  {arm:<9}"
        for gap in args.gaps:
            cell = cells.get((arm, gap))
            if cell is None:
                line += f"{'—':>8}"
            else:
                mark = "*" if cell.running else ""
                line += f"{str(len(cell.curves)) + mark:>8}"
        print(line)
    print()

    tta = tta_table(cells, args.targets, args.seeds)
    bud = budget_table(cells, args.fractions, args.seeds)
    tta.to_csv(args.out_dir / "tta_multiseed.csv", index=False)
    bud.to_csv(args.out_dir / "budget_multiseed.csv", index=False)

    eg = eff_gaps(args.gaps, DEFAULT_SLEEP_DIR, DEFAULT_CLIENTS_CSV)

    for target in args.targets:
        print(f"--- time to {target:.0f}% accuracy (simulated hours, mean over seeds) ---")
        print(f"  {'gap':>5} {'eff':>6} " + "".join(f"{a:>22}" for a in args.arms))
        for gap in args.gaps:
            line = f"  {gap:>5} {eg.get(gap, float('nan')):>6.2f} "
            for arm in args.arms:
                sel = tta[(tta.arm == arm) & (tta.gap == gap) & (tta.target == target)]
                if sel.empty or pd.isna(sel.mean_hours.iloc[0]):
                    line += f"{'—':>22}"
                    continue
                r = sel.iloc[0]
                spread = "" if pd.isna(r.std_hours) else f" ±{r.std_hours:.2f}"
                line += f"{f'{r.mean_hours:.2f}{spread} (n={int(r.n_crossed)})':>22}"
            print(line)
        print()

    print("--- accuracy at a fixed simulated-time budget (mean over seeds) ---")
    for gap in args.gaps:
        sel = bud[bud.gap == gap]
        if sel.empty:
            continue
        print(f"  G={gap} (effective {eg.get(gap, float('nan')):.2f}x)")
        print(f"    {'budget h':>9} " + "".join(f"{a:>20}" for a in args.arms))
        for frac in args.fractions:
            sub = sel[sel.frac == frac]
            if sub.empty:
                continue
            line = f"    {sub.budget_hours.iloc[0]:9.3f} "
            for arm in args.arms:
                row = sub[sub.arm == arm]
                if row.empty or pd.isna(row.mean_acc.iloc[0]):
                    line += f"{'—':>20}"
                    continue
                r = row.iloc[0]
                spread = "" if pd.isna(r.std_acc) else f" ±{r.std_acc:.2f}"
                line += f"{f'{r.mean_acc:.2f}{spread} (n={int(r.n_seeds)})':>20}"
            print(line)
        print()

    print(f"wrote {args.out_dir}/tta_multiseed.csv and budget_multiseed.csv")

    if args.plots:
        from multiseed_plots import render

        render(cells, tta, bud, eg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
