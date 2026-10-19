"""
Analysis for the FLEET-MIX sweep — same tier speeds, different fleet composition.

The gap sweep asks what happens when the hardware spread widens.  This one pins
the spread (G=30: xavier 20 s, orin 3.65 s, thor 0.667 s) and asks what happens
when the *proportions* change — 50/35/15, 70/20/10, 34/33/33, 15/35/50 — with the
same dataset, model and seed.  Runs live in ``runs/mix_sweep`` and were generated
by ``make_mix_sweep.sh``.

Everything measurable here moves for two quite different reasons, and the whole
point of the tables below is to keep them apart:

CAPACITY   A fleet with more fast clients finishes rounds sooner no matter who
           selects.  E[T] falls from 14.8 s (70/20/10) to 4.6 s (15/35/50), so a
           3.2x change in simulated time-to-target is available to a selector
           that does nothing at all.  The ``baseline`` arm measures exactly this,
           which is why it is in the sweep.

STEERING   What the selector adds on top.  Reported as the ratio to that mix's
           own baseline, never as raw hours.  A fleet's composition also sets how
           much steering is *available*: at 15/35/50 half the fleet is already
           thor, so preferring thor buys little; at 70/20/10 preferring it buys a
           lot but there are only 10 of them to prefer, and a client already in
           flight cannot be picked again.  Whether the advantage peaks in the
           middle or rises monotonically is the question this sweep exists to
           answer, and it is not answerable from the gap sweep at all.

Two more properties are worth stating because they constrain what may be
concluded:

* Every cell has **n = 1 seed** (8888).  HANDOFF § 4.13 puts evaluation noise
  alone at ~10 % of a first-crossing time, so a gpu-vs-cpu difference under
  ~1.15x at one mix is not resolvable.  A *trend across four mixes* is a stronger
  signal than any single cell, and that is what to read.
* Time is PLATO's simulated clock as reported, which carries this host's real
  serialisation cost (§ 4.8).  All twelve runs share one host here, unlike the
  gap sweep, so that term is common to every cell — but two runs were in flight
  at a time, so it is inflated relative to the gap sweep's numbers and the two
  sweeps' absolute hours are not interchangeable.

Usage:
    python3 mix_sweep.py                     # tables
    python3 mix_sweep.py --plots             # tables + figures/mix/
    python3 mix_sweep.py --targets 50 55
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODULE_DIR))

from analyze_steering import measure as measure_dispatches  # noqa: E402
from plot_time_to_accuracy import (  # noqa: E402
    ARM_COLOR,
    ARM_ORDER,
    first_crossing,
    load_curve,
)

RUNS_DIR = _MODULE_DIR.parent / "runs" / "mix_sweep"
LOG_DIR = _MODULE_DIR / "logs"
INPUT_DIR = _MODULE_DIR / "input"
SLEEP_DIR = _MODULE_DIR / "output"
OUT_DIR = _MODULE_DIR / "figures" / "mix"

TIER_ORDER = ("xavier", "orin", "thor")  # slowest -> fastest
RUN_RE = re.compile(
    r"^(?P<arm>gpu|cpu|baseline)_xa(?P<anchor>\d+)_gap(?P<gap>\d+)_(?P<mix>m[\d_]+)$"
)
# Concurrency the runs actually hold (check_run.py reports 9 against
# max_concurrency = 10); used only to state the capacity expectation.
CONCURRENCY = 9


@dataclass
class Cell:
    """One (arm, mix) run."""

    arm: str
    mix: str
    run_dir: Path
    curve: pd.DataFrame
    done: bool
    log: Path

    @property
    def horizon(self) -> float:
        return float(self.curve["hours"].iloc[-1])

    @property
    def rounds(self) -> int:
        return int(self.curve["round"].iloc[-1])


@dataclass
class Fleet:
    """The composition itself, read from the files the runs actually used."""

    mix: str
    counts: dict[str, int]
    sleeps: dict[str, float]
    order: int = field(default=0)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def fast_share(self) -> float:
        """Fraction of the fleet that is not the straggler tier."""
        return 1.0 - self.counts[TIER_ORDER[0]] / self.total

    @property
    def mean_sleep(self) -> float:
        """E[T] over the pool — what sets the round rate before any selection."""
        return sum(self.counts[t] * self.sleeps[t] for t in self.counts) / self.total

    @property
    def label(self) -> str:
        return "/".join(str(self.counts[t]) for t in TIER_ORDER)


def read_fleet(mix: str, anchor: int, gap: str) -> Fleet | None:
    """Counts from input/clients_<mix>.csv, sleeps from the run's own sleep CSV.

    Read rather than assumed: these two files are what the runs were pointed at,
    so a mislabelled cell shows up here as a contradiction instead of being
    quietly described by whatever this script believed the mix to be.
    """
    clients_csv = INPUT_DIR / f"clients_{mix}.csv"
    sleep_csv = SLEEP_DIR / f"client_sleep_times_xa{anchor}_gap{gap}_{mix}.csv"
    if not clients_csv.exists() or not sleep_csv.exists():
        return None
    with clients_csv.open(newline="") as f:
        tier_of = {int(r["client_id"]): r["tier_id"] for r in csv.DictReader(f)}
    with sleep_csv.open(newline="") as f:
        sleep_of = {int(r["client_id"]): float(r["sleep_time"]) for r in csv.DictReader(f)}
    counts = {t: 0 for t in TIER_ORDER}
    sleeps: dict[str, float] = {}
    for cid, tier in tier_of.items():
        counts[tier] += 1
        sleeps.setdefault(tier, sleep_of[cid])
    return Fleet(mix=mix, counts=counts, sleeps=sleeps)


def collect(runs_dir: Path, arms: list[str], smooth: int,
            clean_dir: Path | None = None) -> tuple[dict, dict]:
    """Load every (arm, mix) run.

    ``clean_dir`` switches the time axis to the reconstructed clock written by
    ``recompute_wall_time.py`` — every report charged the SAME floor instead of
    whatever the host really spent serialising 42.69 MB at that moment.  That is
    what makes cells run under different machine load comparable; on this sweep
    the floor is ~0.4 s per report, i.e. ~3 % of the simulated clock.  What it
    cannot undo is the effect of that load on the *decisions*: the selector reads
    ``wall_now``, so a contended run made its choices against the contended clock
    (HANDOFF § 4.8).
    """
    cells: dict[tuple[str, str], Cell] = {}
    fleets: dict[str, Fleet] = {}
    for run_dir in sorted(runs_dir.iterdir() if runs_dir.is_dir() else []):
        match = RUN_RE.match(run_dir.name)
        if not match or match["arm"] not in arms:
            continue
        results = sorted(
            (run_dir / "results").glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        results = [p for p in results if not p.stem.endswith(("_binned", "_accuracy_bins"))]
        if not results:
            continue
        clean_csv = (clean_dir / f"{run_dir.name}.csv") if clean_dir else None
        curve = load_curve(results[0], stop_at=None, smooth=smooth, clean_time=clean_csv)
        if clean_dir is not None and str(curve["time_source"].iloc[0]) != "clean":
            print(f"  ! {run_dir.name}: no reconstructed clock in {clean_dir.name}/ — "
                  f"using the reported one.  Run recompute_wall_time.py first.")
        if len(curve) < 20:
            continue
        mix = match["mix"]
        cells[(match["arm"], mix)] = Cell(
            arm=match["arm"],
            mix=mix,
            run_dir=run_dir,
            curve=curve,
            done=(LOG_DIR / f"{run_dir.name}.done").exists(),
            log=LOG_DIR / f"{run_dir.name}.log",
        )
        if mix not in fleets:
            fleet = read_fleet(mix, int(match["anchor"]), match["gap"])
            if fleet is not None:
                fleets[mix] = fleet
    # Order the mixes by how much of the fleet is NOT the straggler tier — the
    # sweep's x-axis.  Sorting by label would put 15/35/50 first and make every
    # figure read backwards.
    for order, mix in enumerate(sorted(fleets, key=lambda m: fleets[m].fast_share)):
        fleets[mix].order = order
    return cells, fleets


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def tta_table(cells: dict, fleets: dict, targets: list[float]) -> pd.DataFrame:
    """First smoothed crossing of each target, in rounds and in simulated hours.

    Both are reported because they answer different questions and the mix moves
    them in opposite directions: a slower fleet needs roughly the same number of
    rounds (mean staleness is C/M, which the composition barely touches) but pays
    more simulated seconds for each one.  A result that shows up in rounds is
    about learning; one that shows up only in hours is about scheduling.
    """
    rows = []
    for (arm, mix), cell in sorted(cells.items()):
        for target in targets:
            hit = first_crossing(cell.curve, target)
            rows.append(
                {
                    "arm": arm,
                    "mix": fleets[mix].label if mix in fleets else mix,
                    "mix_key": mix,
                    "order": fleets[mix].order if mix in fleets else 99,
                    "target": target,
                    "rounds": None if hit is None else hit[0],
                    "hours": None if hit is None else hit[1] / 3600.0,
                    "best_acc": float(cell.curve["smooth_pct"].max()),
                    "ran_rounds": cell.rounds,
                    "done": cell.done,
                }
            )
    return pd.DataFrame(rows)


def budget_table(cells: dict, fleets: dict) -> pd.DataFrame:
    """Smoothed accuracy at a fixed simulated-time budget.

    Two budgets, because there are two questions:

    ``within``  the shortest horizon among the arms of THAT mix — the fair
                arm-vs-arm read inside one composition.
    ``common``  the shortest horizon across every cell in the sweep — the only
                budget at which two different compositions can be compared at
                all, and necessarily set by the slowest fleet.
    """
    rows = []
    by_mix: dict[str, list[Cell]] = {}
    for (_, mix), cell in cells.items():
        by_mix.setdefault(mix, []).append(cell)
    common = min(c.horizon for group in by_mix.values() for c in group)

    for mix, group in by_mix.items():
        within = min(c.horizon for c in group)
        for cell in group:
            hours = cell.curve["hours"].to_numpy()
            acc = cell.curve["smooth_pct"].to_numpy()
            rows.append(
                {
                    "arm": cell.arm,
                    "mix": fleets[mix].label if mix in fleets else mix,
                    "mix_key": mix,
                    "order": fleets[mix].order if mix in fleets else 99,
                    "within_hours": within,
                    "acc_within": float(np.interp(within, hours, acc)),
                    "common_hours": common,
                    "acc_common": float(np.interp(common, hours, acc)),
                    "acc_final": float(acc[-1]),
                }
            )
    return pd.DataFrame(rows)


def steering_table(cells: dict, fleets: dict) -> pd.DataFrame:
    """Dispatch lift per tier — share dispatched over share of pool.

    >1 means the arm preferred that tier over what random selection would give.
    The pool share is the denominator, so the number is already normalised for
    the composition: a fleet with 50 thor clients hands thor 50 % of dispatches
    by doing nothing, and that reads as lift 1.00, not as steering.
    """
    rows = []
    for (arm, mix), cell in sorted(cells.items()):
        if not cell.log.exists() or mix not in fleets:
            continue
        fleet = fleets[mix]
        with (INPUT_DIR / f"clients_{mix}.csv").open(newline="") as f:
            tier_of = {int(r["client_id"]): r["tier_id"] for r in csv.DictReader(f)}
        disp, v_by_tier = measure_dispatches(cell.log, tier_of)
        total = sum(disp.values())
        if not total:
            continue
        for tier in TIER_ORDER:
            pool_share = fleet.counts[tier] / fleet.total
            if not pool_share:
                continue
            share = disp[tier] / total
            rows.append(
                {
                    "arm": arm,
                    "mix": fleet.label,
                    "mix_key": mix,
                    "order": fleet.order,
                    "tier": tier,
                    "dispatch_share": share,
                    "pool_share": pool_share,
                    "lift": share / pool_share,
                    "mean_V": float(np.mean(v_by_tier[tier])) if v_by_tier[tier] else None,
                    "n_dispatch": disp[tier],
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #


def _mix_order(frame: pd.DataFrame) -> list[str]:
    return [m for _, m in sorted({(r.order, r.mix_key) for r in frame.itertuples()})]


def print_fleets(fleets: dict) -> None:
    print("--- the four fleets (same tier speeds, G=30) ---")
    print(f"  {'mix':>10} {'non-straggler':>14} {'E[T] s':>8} {'T_round s':>10} {'capacity':>9}")
    slowest = max(f.mean_sleep for f in fleets.values())
    for mix in sorted(fleets, key=lambda m: fleets[m].order):
        f = fleets[mix]
        print(
            f"  {f.label:>10} {f.fast_share:>13.0%} {f.mean_sleep:>8.2f}"
            f" {f.mean_sleep / CONCURRENCY:>10.3f} {slowest / f.mean_sleep:>8.2f}x"
        )
    print(
        "  capacity = E[T] of the slowest fleet over this one's: the speed-up a\n"
        "  selector gets for free, before it selects anything.\n"
    )


def print_tta(tta: pd.DataFrame, targets: list[float], arms: list[str]) -> None:
    for target in targets:
        sel = tta[tta.target == target]
        if sel.empty:
            continue
        print(f"--- time to {target:.0f}% accuracy (simulated hours / round) ---")
        header = f"  {'mix':>10} " + "".join(f"{a:>20}" for a in arms)
        print(header + f"{'gpu/base':>10}{'cpu/base':>10}")
        for mix_key in _mix_order(sel):
            by_arm = {r.arm: r for r in sel[sel.mix_key == mix_key].itertuples()}
            line = f"  {next(iter(by_arm.values())).mix:>10} "
            for arm in arms:
                r = by_arm.get(arm)
                if r is None or r.hours is None or pd.isna(r.hours):
                    line += f"{'never crossed':>20}"
                else:
                    line += f"{f'{r.hours:.2f} h / r{int(r.rounds)}':>20}"
            base = by_arm.get("baseline")
            for arm in ("gpu", "cpu"):
                r = by_arm.get(arm)
                ok = (
                    base is not None and r is not None
                    and base.hours and r.hours
                    and not pd.isna(base.hours) and not pd.isna(r.hours)
                )
                line += f"{f'{r.hours / base.hours:.2f}x':>10}" if ok else f"{'-':>10}"
            print(line)
        print(
            "  gpu/base < 1 means the predictor beat random selection on that fleet.\n"
            "  Raw hours across mixes are mostly capacity (see the table above); the\n"
            "  ratios are the part that is about selection.\n"
        )


def print_budget(bud: pd.DataFrame, arms: list[str]) -> None:
    if bud.empty:
        return
    print("--- accuracy at a fixed simulated-time budget ---")
    print(
        f"  {'mix':>10} {'within h':>9} " + "".join(f"{a:>10}" for a in arms)
        + f"{'common h':>10}" + "".join(f"{a:>10}" for a in arms)
    )
    for mix_key in _mix_order(bud):
        sub = {r.arm: r for r in bud[bud.mix_key == mix_key].itertuples()}
        first = next(iter(sub.values()))
        line = f"  {first.mix:>10} {first.within_hours:>9.2f} "
        line += "".join(f"{sub[a].acc_within:>10.2f}" if a in sub else f"{'-':>10}" for a in arms)
        line += f"{first.common_hours:>10.2f}"
        line += "".join(f"{sub[a].acc_common:>10.2f}" if a in sub else f"{'-':>10}" for a in arms)
        print(line)
    print(
        "  'within' compares arms inside one fleet; 'common' is the only budget at\n"
        "  which two fleets can be compared, and the slowest fleet sets it.\n"
    )


def print_steering(steer: pd.DataFrame, arms: list[str]) -> None:
    if steer.empty:
        print("--- dispatch lift: no logs parsed yet ---\n")
        return
    print("--- dispatch lift per tier (share dispatched / share of pool) ---")
    for arm in arms:
        sub = steer[steer.arm == arm]
        if sub.empty:
            continue
        print(f"\n  [{arm}]")
        print(f"    {'mix':>10} " + "".join(f"{t:>10}" for t in TIER_ORDER) + f"{'thor/xav':>10}")
        for mix_key in _mix_order(sub):
            rows = {r.tier: r for r in sub[sub.mix_key == mix_key].itertuples()}
            line = f"    {next(iter(rows.values())).mix:>10} "
            line += "".join(
                f"{rows[t].lift:>10.2f}" if t in rows else f"{'-':>10}" for t in TIER_ORDER
            )
            slow, fast = TIER_ORDER[0], TIER_ORDER[-1]
            if slow in rows and fast in rows and rows[slow].lift:
                line += f"{rows[fast].lift / rows[slow].lift:>10.2f}"
            print(line)
    print(
        "\n  baseline is the control: it selects at random, so every lift should sit\n"
        "  at ~1.00 whatever the fleet.  A baseline that steers is a bug, not a\n"
        "  finding.  cpu cannot separate xavier from orin by construction (both\n"
        "  clock 2.00 GHz in tiers_cpu.csv), so a flat xavier/orin pair there is\n"
        "  expected — HANDOFF § 4.5.\n"
    )


# --------------------------------------------------------------------------- #
# Figures — house style from plot_time_to_accuracy.py (same arm colours, so a
# reader moving between the two sweeps' figures never has to re-learn them).
# --------------------------------------------------------------------------- #


def render(cells: dict, fleets: dict, tta: pd.DataFrame, steer: pd.DataFrame,
           targets: list[float], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot_time_to_accuracy import GRID, INK, INK_2, INK_MUTED, SURFACE, _style_axes

    order = sorted(fleets, key=lambda m: fleets[m].order)
    xs = list(range(len(order)))
    labels = [fleets[m].label for m in order]
    target = targets[-1]

    def _hours(arm: str, mix: str):
        row = tta[(tta.arm == arm) & (tta.mix_key == mix) & (tta.target == target)]
        if row.empty or row.hours.isna().all():
            return None
        return float(row.hours.iloc[0])

    # 1. Hours to target vs composition — absolute, then divided by that fleet's
    #    own baseline, because the absolute panel is dominated by capacity.
    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE)
    for arm in ARM_ORDER:
        pts = [(x, _hours(arm, m)) for x, m in zip(xs, order)]
        pts = [(x, h) for x, h in pts if h is not None]
        if pts:
            ax_abs.plot([p[0] for p in pts], [p[1] for p in pts], color=ARM_COLOR[arm],
                        linewidth=2.0, marker="o", markersize=8, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, label=arm, zorder=3)
    # The capacity-only prediction: hours scale with E[T] if nothing else changes.
    anchor = order[0]
    h0 = _hours("baseline", anchor)
    if h0 is not None:
        ax_abs.plot(xs, [h0 * fleets[m].mean_sleep / fleets[anchor].mean_sleep for m in order],
                    color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 4)), zorder=2,
                    label="capacity only (E[T])")
    _style_axes(ax_abs, "fleet  xavier/orin/thor", f"simulated hours to {target:.0f}%")
    ax_abs.set_xticks(xs, labels)
    ax_abs.set_title(f"time to {target:.0f}%: mostly capacity", color=INK, fontsize=11,
                     fontweight="bold", loc="left")
    ax_abs.legend(frameon=False, fontsize=9, labelcolor=INK_2)

    for arm in ("gpu", "cpu"):
        pts = []
        for x, mix in zip(xs, order):
            h, hb = _hours(arm, mix), _hours("baseline", mix)
            if h is not None and hb:
                pts.append((x, h / hb))
        if pts:
            ax_rel.plot([p[0] for p in pts], [p[1] for p in pts], color=ARM_COLOR[arm],
                        linewidth=2.0, marker="o", markersize=8, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, label=f"{arm} / baseline", zorder=3)
    ax_rel.axhline(1.0, color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 4)), zorder=1)
    ax_rel.annotate("no advantage", xy=(xs[-1], 1.0), xytext=(0, 5),
                    textcoords="offset points", ha="right", fontsize=8.5, color=INK_MUTED)
    _style_axes(ax_rel, "fleet  xavier/orin/thor", "hours / baseline hours")
    ax_rel.set_xticks(xs, labels)
    ax_rel.set_title("what the selector added", color=INK, fontsize=11,
                     fontweight="bold", loc="left")
    ax_rel.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out_dir / "tta_vs_mix.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 2. The curves themselves, one panel per fleet.
    ncol = min(2, len(order)) or 1
    nrow = (len(order) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 4.0 * nrow),
                             facecolor=SURFACE, squeeze=False)
    for ax, mix in zip(axes.flat, order):
        for arm in ARM_ORDER:
            cell = cells.get((arm, mix))
            if cell is None:
                continue
            ax.plot(cell.curve["hours"], cell.curve["accuracy_pct"], color=ARM_COLOR[arm],
                    linewidth=0.7, alpha=0.22, zorder=2)
            ax.plot(cell.curve["hours"], cell.curve["smooth_pct"], color=ARM_COLOR[arm],
                    linewidth=2.0, label=arm, zorder=3)
        ax.axhline(target, color=GRID, linewidth=1.0, linestyle=(0, (4, 4)), zorder=1)
        _style_axes(ax, "simulated hours", "accuracy %")
        ax.set_title(f"{fleets[mix].label}   E[T] = {fleets[mix].mean_sleep:.1f} s",
                     color=INK, fontsize=11, fontweight="bold", loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "curves_by_mix.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 3. Steering: does the lift survive the composition change?
    if not steer.empty:
        arms_present = [a for a in ARM_ORDER if a in set(steer.arm)]
        fig, axes = plt.subplots(1, len(arms_present), figsize=(5.2 * len(arms_present), 4.2),
                                 facecolor=SURFACE, squeeze=False)
        # Tier is an ORDERED quantity here (slow -> fast), so it gets one hue in
        # three steps rather than three categorical hues, which would imply the
        # tiers are unrelated categories.
        tier_step = {"xavier": "#9ec6f0", "orin": "#4f93dd", "thor": "#14508f"}
        for ax, arm in zip(axes.flat, arms_present):
            for tier in TIER_ORDER:
                pts = []
                for x, mix in zip(xs, order):
                    row = steer[(steer.arm == arm) & (steer.mix_key == mix) & (steer.tier == tier)]
                    if not row.empty:
                        pts.append((x, float(row.lift.iloc[0])))
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=tier_step[tier],
                            linewidth=2.0, marker="o", markersize=8, markeredgecolor=SURFACE,
                            markeredgewidth=1.5, label=tier, zorder=3)
            ax.axhline(1.0, color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 4)), zorder=1)
            _style_axes(ax, "fleet  xavier/orin/thor", "dispatch lift")
            ax.set_xticks(xs, labels)
            ax.set_title(arm, color=INK, fontsize=11, fontweight="bold", loc="left")
            ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
        fig.suptitle("lift = share dispatched / share of pool; 1.00 = random",
                     color=INK, fontsize=12, fontweight="bold", x=0.006, ha="left")
        fig.tight_layout()
        fig.savefig(out_dir / "steering_vs_mix.png", dpi=160, facecolor=SURFACE)
        plt.close(fig)

    print(f"wrote figures to {out_dir}/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--arms", nargs="+", default=ARM_ORDER, choices=ARM_ORDER)
    p.add_argument("--targets", nargs="+", type=float, default=[50.0, 55.0])
    p.add_argument("--smooth", type=int, default=25, help="evaluations in the rolling median")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--no-steering", action="store_true", help="skip the log parse (slow)")
    p.add_argument(
        "--time-source", choices=("reported", "clean"), default="reported",
        help="'clean' reads the reconstructed clock from <out-dir>/clean_time/, "
        "which charges every report a uniform floor instead of the host's own "
        "(recompute_wall_time.py writes it).  Use it whenever the cells did not "
        "all run under the same machine load.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    clean_dir = args.out_dir / "clean_time" if args.time_source == "clean" else None
    cells, fleets = collect(args.runs_dir, args.arms, args.smooth, clean_dir)
    if not cells:
        print(f"no runs found under {args.runs_dir}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    running = sorted(
        f"{c.arm}/{fleets[c.mix].label if c.mix in fleets else c.mix}"
        for c in cells.values() if not c.done
    )
    print(f"{len(cells)} run(s) under {args.runs_dir.name}"
          + (f"; STILL RUNNING: {', '.join(running)}" if running else "; all finished"))
    print(f"  time axis: {args.time_source}"
          + ("  (uniform floor, host load removed)" if args.time_source == "clean" else
             "  (PLATO's own clock, carries this host's serialisation cost)"))
    print("  (unfinished runs are listed so a mid-sweep table is never read as the result)\n")

    print_fleets(fleets)

    tta = tta_table(cells, fleets, args.targets)
    bud = budget_table(cells, fleets)
    tta.to_csv(args.out_dir / "tta_mix.csv", index=False)
    bud.to_csv(args.out_dir / "budget_mix.csv", index=False)

    print_tta(tta, args.targets, args.arms)
    print_budget(bud, args.arms)

    steer = pd.DataFrame()
    if not args.no_steering:
        steer = steering_table(cells, fleets)
        if not steer.empty:
            steer.to_csv(args.out_dir / "steering_mix.csv", index=False)
        print_steering(steer, args.arms)

    print(f"wrote {args.out_dir}/tta_mix.csv, budget_mix.csv"
          + (", steering_mix.csv" if not steer.empty else ""))

    if args.plots:
        render(cells, fleets, tta, steer, args.targets, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
