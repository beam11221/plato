"""
Figures for the multi-seed sweep — rendered by ``multiseed.py --plots``.

Kept separate from the aggregation so the tables can be produced on a box without
matplotlib, and so the numbers a figure draws are the same objects the CSVs were
written from rather than a re-read of them.

Colours are ``plot_time_to_accuracy.ARM_COLOR`` unchanged — the sweep's existing
arm palette, which is the documented default categorical palette's first three
slots (validated adjacent- and all-pairs, both modes).  Text wears ink tokens; a
colour appears only on a mark.

The seed axis is drawn, never hidden:

* a cell with every seed gets a solid line and a filled marker;
* a cell short of that gets a dashed line, a hollow marker and an ``n=`` label,
  so a single-seed baseline is never read as a three-seed mean.

Spread is the seed **range** (min..max) on the curve panels and ±1 SD on the
summary panels.  With n=3 an SD is a weak estimate, so the band is there to show
that the seeds overlap or do not — not to support a significance claim.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from plot_time_to_accuracy import ARM_COLOR, ARM_ORDER  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

ARM_SHORT = {
    "gpu": "gpu — belief = reality",
    "cpu": "cpu — belief wrong",
    "baseline": "baseline — no predictor",
}


def _style(ax, xlabel: str, ylabel: str, title: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)


def _legend(fig, arms: list[str], note: str | None = None) -> None:
    handles = [
        Line2D([], [], color=ARM_COLOR[a], lw=2, marker="o", ms=7, label=ARM_SHORT[a])
        for a in arms
    ]
    if note:
        handles.append(Line2D([], [], color=INK_MUTED, lw=2, ls="--", label=note))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
    )


# --------------------------------------------------------------------------- #
# Figure 1 — the summary: hours to target vs effective gap
# --------------------------------------------------------------------------- #


def _stagger(labels: list[tuple[str, float]], min_sep: float) -> dict[str, float]:
    """Nudge direct labels apart along y so two arms ending at the same value do
    not print on top of each other.  Returns the y to draw each label at."""
    ordered = sorted(labels, key=lambda t: t[1])
    out: dict[str, float] = {}
    prev = None
    for name, y in ordered:
        if prev is not None and y - prev < min_sep:
            y = prev + min_sep
        out[name] = y
        prev = y
    return out


def fig_tta_vs_gap(tta: pd.DataFrame, eg: dict, targets: list[float], arms: list[str],
                   out: Path, n_full: int, note: str = "") -> None:
    fig, axes = plt.subplots(1, len(targets), figsize=(5.9 * len(targets), 5.0), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    axes = axes[0]

    for ax, target in zip(axes, targets):
        ends: list[tuple[str, float, float, str]] = []
        for arm in arms:
            sel = tta[(tta.arm == arm) & (tta.target == target)].copy()
            sel = sel[sel.mean_hours.notna()]
            if sel.empty:
                continue
            sel["eff"] = sel.gap.map(eg)
            sel = sel.sort_values("eff")
            full = sel.n_crossed == n_full
            colour = ARM_COLOR[arm]

            # One line per arm; the dash marks the seed count, not the arm.
            ax.plot(sel.eff, sel.mean_hours, color=colour, lw=2, zorder=3,
                    ls="-" if full.all() else "--")
            err = sel.std_hours.fillna(0.0)
            ax.fill_between(sel.eff, sel.mean_hours - err, sel.mean_hours + err,
                            color=colour, alpha=0.13, lw=0, zorder=2)
            for _, r in sel.iterrows():
                solid = r.n_crossed == n_full
                ax.plot(r.eff, r.mean_hours, marker="o", ms=8, zorder=4,
                        color=colour if solid else SURFACE,
                        markeredgecolor=colour, markeredgewidth=2)
            # The seed count rides on the arm's direct label rather than on every
            # point: per-point n= tags collide with the labels and with each other,
            # and the count is a property of the arm here, not of one gap.
            counts = sorted(set(int(v) for v in sel.n_crossed))
            tag = "" if counts == [n_full] else (
                f"  n={counts[0]}" if len(counts) == 1 else f"  n={counts[0]}–{counts[-1]}"
            )
            last = sel.iloc[-1]
            ends.append((arm + tag, float(last.eff), float(last.mean_hours), colour))

        _style(ax, "effective gap  (c + s_slow) / (c + s_fast)",
               "simulated hours to target",
               f"Time to {target:.0f}% accuracy")
        ax.set_ylim(bottom=0)
        ax.set_xlim(0.2, max(eg.values()) * 1.19)

        # Direct labels at the right end — ink text, the mark carries identity.
        # Staggered so arms that finish at the same time stay readable.
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        placed = _stagger([(a, y) for a, _, y, _ in ends], span * 0.062)
        for arm, x_end, y_end, colour in ends:
            y_lab = placed[arm]
            ax.annotate(arm, (x_end, y_lab), textcoords="offset points",
                        xytext=(11, 0), va="center", fontsize=9.5, color=INK_2)
            if abs(y_lab - y_end) > span * 0.012:  # leader line back to the point
                ax.plot([x_end, x_end + (ax.get_xlim()[1] - x_end) * 0.16],
                        [y_end, y_lab], color=colour, lw=0.9, alpha=0.55, zorder=2)

    fig.suptitle("Time-to-accuracy across the tier-speed gap — mean over seeds",
                 color=INK, fontsize=13, x=0.015, ha="left", y=0.975)
    fig.text(0.015, 0.928,
             "Simulated clock. Band = ±1 SD over seeds. Hollow marker + n= : fewer seeds than the rest."
             + (f"  {note}  A crossing already reached is final." if note else ""),
             color=INK_MUTED, fontsize=9, ha="left", va="top")
    _legend(fig, arms, "fewer seeds")
    fig.tight_layout(rect=[0, 0.09, 1, 0.915])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2 — the time-budget curves, one panel per gap
# --------------------------------------------------------------------------- #


def fig_budget_curves(cells: dict, eg: dict, gaps: list[str], arms: list[str],
                      out: Path, n_full: int, targets: list[float], note: str = "") -> None:
    gaps = [g for g in gaps if any((a, g) in cells for a in arms)]
    fig, axes = plt.subplots(1, len(gaps), figsize=(3.05 * len(gaps), 5.6),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    axes = axes[0]

    for ax, gap in zip(axes, gaps):
        present = [cells[(a, gap)] for a in arms if (a, gap) in cells]
        horizon = min(c.horizon for c in present)
        grid = np.linspace(0.02, horizon, 220)

        for cell in present:
            colour = ARM_COLOR[cell.arm]
            stack = np.vstack([
                np.interp(grid, df["hours"].to_numpy(), df["smooth_pct"].to_numpy())
                for df in cell.curves.values()
            ])
            mean = stack.mean(axis=0)
            full = len(cell.curves) == n_full
            if len(stack) > 1:
                ax.fill_between(grid, stack.min(axis=0), stack.max(axis=0),
                                color=colour, alpha=0.14, lw=0, zorder=2)
            ax.plot(grid, mean, color=colour, lw=2, zorder=3, ls="-" if full else "--")
            if not full:
                ax.annotate(f"n={len(cell.curves)}", (grid[-1], mean[-1]),
                            textcoords="offset points", xytext=(-3, -13),
                            ha="right", fontsize=7.5, color=INK_MUTED)

        # Target lines are labelled on the right, clear of the y-axis ticks.
        for target in targets:
            ax.axhline(target, color=GRID, lw=1, ls=":", zorder=1)
            if gap == gaps[-1]:
                ax.annotate(f"{target:.0f}%", (1.0, target),
                            xycoords=("axes fraction", "data"),
                            textcoords="offset points", xytext=(5, -3),
                            fontsize=8, color=INK_MUTED, annotation_clip=False)

        _style(ax, "simulated hours", "accuracy (%)" if gap == gaps[0] else "",
               f"G={gap}  ·  {eg.get(gap, float('nan')):.1f}x effective")
        ax.set_xlim(0, horizon)

    # Zoom to where the arms actually separate.  The 30->45 ramp is over in the
    # first few minutes and identical for every arm; showing it compresses the
    # decision region into the top fifth of the panel.
    axes[0].set_ylim(44, 60)

    fig.suptitle("Accuracy at a fixed simulated-time budget — mean over seeds",
                 color=INK, fontsize=13, x=0.012, ha="left", y=0.978)
    fig.text(0.012, 0.938,
             "Each panel runs to the longest budget every arm and seed covers. "
             "Band = seed min..max. Dashed + n= : fewer seeds. y-axis zoomed to 44–60%."
             + (f"  {note}" if note else ""),
             color=INK_MUTED, fontsize=9, ha="left", va="top")
    _legend(fig, arms, "fewer seeds")
    fig.tight_layout(rect=[0, 0.085, 1, 0.925])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3 — gpu minus cpu against the G=1 noise floor
# --------------------------------------------------------------------------- #


def fig_delta_vs_floor(bud: pd.DataFrame, eg: dict, gaps: list[str], out: Path) -> None:
    """The one panel that says whether the effect clears the noise.

    At G=1 the tiers are the same speed, so neither predictor can exploit a speed
    signal and whatever ``gpu - cpu`` remains there is trajectory noise.  A bar at
    another gap counts only if it clears that band.
    """
    wide = bud.pivot_table(index=["gap", "frac"], columns="arm", values="mean_acc")
    if "gpu" not in wide or "cpu" not in wide:
        return
    wide["delta"] = wide["gpu"] - wide["cpu"]
    wide = wide.reset_index()

    floor_rows = wide[wide.gap == "01"].delta.dropna()
    floor = float(floor_rows.abs().max()) if len(floor_rows) else 0.0

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    fig.patch.set_facecolor(SURFACE)

    plot_gaps = [g for g in gaps if g in set(wide.gap)]
    # The G=1 group IS the floor, not a measurement against it.
    if "01" in plot_gaps:
        i = plot_gaps.index("01")
        ax.axvspan(i - 0.5, i + 0.5, color=GRID, alpha=0.45, zorder=0)
    fracs = sorted(wide.frac.unique())
    x = np.arange(len(plot_gaps), dtype=float)
    width = 0.8 / max(1, len(fracs))

    # One hue, lightness stepped by budget: the budget fraction is ORDINAL, so it
    # takes a ramp, not the categorical slots (which are spoken for by the arms).
    shades = ["#a9c8ee", "#7aabe4", "#4b8dda", "#2a78d6"]
    for i, frac in enumerate(fracs):
        vals, xs = [], []
        for j, gap in enumerate(plot_gaps):
            row = wide[(wide.gap == gap) & (wide.frac == frac)]
            if row.empty or pd.isna(row.delta.iloc[0]):
                continue
            vals.append(float(row.delta.iloc[0]))
            xs.append(x[j] - 0.4 + width * (i + 0.5))
        ax.bar(xs, vals, width=width * 0.88, color=shades[i % len(shades)],
               zorder=3, label=f"{frac:.0%} of budget",
               edgecolor=SURFACE, linewidth=2)

    ax.axhspan(-floor, floor, color=GRID, alpha=0.8, zorder=1)
    ax.axhline(0, color=INK_MUTED, lw=1, zorder=2)
    ax.annotate(f"±{floor:.2f} pts — the G=1 noise floor",
                (-0.42, -floor), textcoords="offset points", xytext=(0, -13),
                ha="left", fontsize=8.5, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"G={g}\n{eg.get(g, float('nan')):.1f}x" for g in plot_gaps])
    if "01" in plot_gaps:
        ax.get_xticklabels()[plot_gaps.index("01")].set_color(INK_MUTED)
        ax.annotate("control — tiers equal speed,\nso this spread IS the floor",
                    (plot_gaps.index("01"), 1.0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(0, -30),
                    ha="center", fontsize=8, color=INK_MUTED)
    _style(ax, "nominal gap / effective gap", "gpu − cpu accuracy (points)",
           "Does the right predictor actually help? gpu minus cpu at equal time budget")
    # Legend below the axes: the tallest bar owns the top-left corner.
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=len(fracs),
                    loc="upper center", bbox_to_anchor=(0.5, -0.16))
    leg.set_title("simulated-time budget", prop={"size": 9})
    leg.get_title().set_color(INK_MUTED)

    fig.text(0.012, 0.015,
             "Positive = gpu ahead. Bars inside the grey band are indistinguishable from seed noise.",
             color=INK_MUTED, fontsize=9)
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _inflight_note(cells: dict) -> str:
    """Runs still training when the figure was made.  A crossing already reached is
    final, but a curve can still grow to the right, so the reader is told which."""
    live = sorted(f"{c.arm} G={c.gap}" for c in cells.values() if c.running)
    return f"Still running at render time: {', '.join(live)}." if live else ""


def render(cells: dict, tta: pd.DataFrame, bud: pd.DataFrame, eg: dict, args) -> None:
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    n_full = len(args.seeds)
    arms = [a for a in ARM_ORDER if a in args.arms and any(k[0] == a for k in cells)]
    note = _inflight_note(cells)

    fig_tta_vs_gap(tta, eg, args.targets, arms, out_dir / "tta_vs_gap_multiseed.png",
                   n_full, note)
    fig_budget_curves(cells, eg, args.gaps, arms, out_dir / "budget_curves_multiseed.png",
                      n_full, args.targets, note)
    fig_delta_vs_floor(bud, eg, args.gaps, out_dir / "budget_delta_vs_floor.png")
    print(f"wrote figures to {out_dir}/")
