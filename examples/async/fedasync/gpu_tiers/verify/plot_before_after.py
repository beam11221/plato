"""
Before/after figure for the flush-accounting fix — gpu arm, G=40.

Both traces come from the same cell of the sweep, the same config and the same
seed. The only difference is whether ``servers/base.py`` records the clients its
staleness flush merges, so the difference in the two lines is the bug.

    python3 plot_before_after.py
    python3 plot_before_after.py --out /tmp/fix.png
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).parent
_GPU_TIERS = _HERE.parent
sys.path.insert(0, str(_GPU_TIERS))

from plot_time_to_accuracy import (  # noqa: E402
    GRID,
    INK,
    INK_2,
    INK_MUTED,
    SURFACE,
    concurrency_trace,
)

BEFORE = _GPU_TIERS.parent / "runs" / "gap_sweep_prebugfix" / "gpu_xa20_gap40.log"
AFTER = _GPU_TIERS / "logs" / "verify_gpu_xa20_gap40.log"

# Slots 8 (red) and 3 (aqua) of the reference palette — before/after is a status
# contrast, not two members of a series, so it does not take the arm colours.
COLOR_BEFORE = "#e34948"
COLOR_AFTER = "#1baf7a"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--before", type=Path, default=BEFORE)
    parser.add_argument("--after", type=Path, default=AFTER)
    parser.add_argument("--out", type=Path, default=_GPU_TIERS / "figures" / "fix_before_after.png")
    parser.add_argument("--max-conc", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    series = []
    for label, path, color in (
        ("before the fix", args.before, COLOR_BEFORE),
        ("after the fix", args.after, COLOR_AFTER),
    ):
        if not path.exists():
            raise SystemExit(f"missing log: {path}")
        trace, lost = concurrency_trace(path)
        if not trace:
            raise SystemExit(f"no aggregation events in {path}")
        series.append((label, trace, lost, color))

    # Compare over the shorter run's span so the x range means the same thing.
    span = min(len(trace) for _, trace, _, _ in series)

    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)

    ax.axhline(args.max_conc, color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 4)), zorder=1)
    ax.annotate(
        f"max_concurrency = {args.max_conc}",
        xy=(0.99, args.max_conc), xycoords=("axes fraction", "data"),
        ha="right", va="bottom", fontsize=8.5, color=INK_MUTED,
    )

    for label, trace, lost, color in series:
        cut = trace[:span]
        ax.plot(range(1, len(cut) + 1), cut, color=color, linewidth=2.0, zorder=3, label=label)
        tail = cut[len(cut) // 5:]
        ax.annotate(
            f"{label} → {statistics.median(tail):.0f}",
            xy=(len(cut), cut[-1]), xytext=(7, 0), textcoords="offset points",
            va="center", fontsize=9, color=color, fontweight="bold",
        )
        print(
            f"{label:<16} rounds={len(trace):>5}  steady-state median={statistics.median(tail):>4.0f}"
            f"  min={min(tail):>3}  extra clients merged by flushes={lost}"
        )

    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_xlabel("round", color=INK_2, fontsize=10)
    ax.set_ylabel("clients training at once", color=INK_2, fontsize=10)
    ax.set_xscale("log")
    ax.set_ylim(0, args.max_conc + 1.2)
    ax.margins(x=0.14)
    ax.set_title(
        "One line of bookkeeping — gpu arm, G=40, same config and seed",
        color=INK, fontsize=13, fontweight="bold", loc="left", pad=14,
    )
    legend = ax.legend(frameon=False, fontsize=9, loc="center left")
    for text in legend.get_texts():
        text.set_color(INK_2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
