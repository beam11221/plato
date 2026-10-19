"""
Time-to-accuracy comparison across the heterogeneity sweep.

One script for the question the sweep exists to answer: **at a given tier-speed
gap G, which arm reaches a target accuracy first — `gpu` (predictor belief
matches reality), `cpu` (belief deliberately wrong), or `baseline` (no predictor
at all)?**

It merges three things that were previously done by hand, in three places:

- ``../bin_results.py:bin_by_accuracy`` — first round/elapsed_time at which each
  accuracy threshold is crossed.  That function reads **one** CSV and writes a
  sibling file; here the same crossing logic runs over every run of the sweep and
  lands in one long-format table.
- ``summarize_sweep.py`` — nominal gap G is not the heterogeneity the server
  actually sees.  Its ``per_tier_sleep`` / ``pool_mean_sleep`` are imported
  directly so the x-axis is the **effective** gap ``(c + s_slow)/(c + s_fast)``,
  the same number that script prints.
- the plots themselves, which did not exist.

Time is PLATO's **simulated** wall clock (``elapsed_time`` in results/*.csv), not
real seconds — with ``sleep_simulation = true`` the clients never really sleep,
they only report the tier sleep as training time.  That is the whole point: the
simulated clock is what a real deployment would have paid.

Outputs (into ``--out-dir``, default ``figures/``):

    tta_gap<G>.png     accuracy vs simulated hours, one line per arm — one figure
                       per gap.  Bold line is the rolling-median accuracy (see
                       --smooth), the faint one behind it is the raw per-round value.
    tta_vs_gap.png     hours to reach each target accuracy vs effective gap, one
                       line per arm.  The summary figure.
    hypothesis_cpu_vs_gpu.png
                       four panels asking whether the cpu arm falls behind as the
                       gap widens: the spread each predictor can see, the tier mix
                       it dispatches to, and the cost of the difference in rounds
                       and in hours — which do not agree.
    tta_summary.csv    long format: arm, gap, effective_gap, target, first_round,
                       hours, and the ratio against the reference arm.
    tier_mix.csv       per run, the share of dispatches that went to each tier
                       (parsed from the logs, cached by log size).

Usage:
    python3 plot_time_to_accuracy.py
    python3 plot_time_to_accuracy.py --targets 45 50 55 --reference baseline
    python3 plot_time_to_accuracy.py --gaps 01 10 --stop-at none
    python3 plot_time_to_accuracy.py --runs-dir ../runs/gap_sweep --out-dir /tmp/figs

Runs still in flight are included and drawn dashed, labelled "(running)" — a run
is considered finished when ``logs/<run>.done`` exists (written by run_sweep.sh).
So this is safe to run mid-sweep to watch the comparison take shape.

Why --stop-at defaults to 0.6
-----------------------------
``target_accuracy`` was 0.7 until 2026-08-09 and never fired, so the two gap01
runs kept training long past 0.6 while every later run early-stops there.  The
crossing times are unaffected (a crossing is a prefix property), but the *curves*
would show one arm running hours longer for no reason.  Truncating every run at
its first 0.6 crossing puts them back on the same footing.  ``--stop-at none``
plots the raw trajectories.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from recompute_wall_time import dispatch_queues, parse_schedule
from summarize_sweep import (
    DEFAULT_FLOOR_SECONDS,
    TIER_ORDER,
    load_client_tiers,
    per_tier_sleep,
    predictor_gap,
)

_MODULE_DIR = Path(__file__).parent
DEFAULT_RUNS_DIR = _MODULE_DIR.parent / "runs" / "gap_sweep"
DEFAULT_OUT_DIR = _MODULE_DIR / "figures"
DEFAULT_LOG_DIR = _MODULE_DIR / "logs"
DEFAULT_SLEEP_DIR = _MODULE_DIR / "output"
DEFAULT_CLIENTS_CSV = _MODULE_DIR / "input" / "clients.csv"

RUN_DIR_RE = re.compile(r"^(?P<arm>gpu|cpu|baseline)_xa(?P<anchor>\d+)_gap(?P<gap>\d+)$")

# Categorical slots 1-3 of the reference palette — the three that validate on the
# all-pairs list.  Colour follows the arm, never its rank, so a missing arm never
# repaints the others.
ARM_ORDER = ["gpu", "cpu", "baseline"]
ARM_COLOR = {"gpu": "#2a78d6", "cpu": "#eb6834", "baseline": "#1baf7a"}
ARM_LABEL = {
    "gpu": "gpu — belief = reality",
    "cpu": "cpu — belief wrong (GHz)",
    "baseline": "baseline — random, no predictor",
}

INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


class Run:
    """One sweep run: its results CSV plus the metadata needed to place it."""

    def __init__(self, path: Path, arm: str, anchor: int, gap: int, csv_path: Path, done: bool):
        self.path = path
        self.arm = arm
        self.anchor = anchor
        self.gap = gap
        self.csv_path = csv_path
        self.done = done
        self.frame: pd.DataFrame | None = None
        self.effective_gap: float | None = None
        self.perceived: float | None = None
        self.mix: dict | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def label(self) -> str:
        return ARM_LABEL[self.arm] + ("" if self.done else "  (running)")


def _results_csv(run_dir: Path) -> Path | None:
    """The run's results CSV — PLATO names it <pid>.csv, one per run.

    A restarted run leaves more than one; the newest is the live one.
    """
    candidates = sorted(
        (run_dir / "results").glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    candidates = [p for p in candidates if not p.stem.endswith(("_binned", "_accuracy_bins"))]
    if not candidates:
        return None
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        print(f"  ! {run_dir.name}: {len(candidates)} result CSVs ({names}) — using the newest")
    return candidates[0]


def discover_runs(runs_dir: Path, log_dir: Path, arms: list[str], gaps: list[str] | None) -> list[Run]:
    runs: list[Run] = []
    for run_dir in sorted(runs_dir.iterdir() if runs_dir.is_dir() else []):
        match = RUN_DIR_RE.match(run_dir.name)
        if not match or match["arm"] not in arms:
            continue
        if gaps is not None and match["gap"] not in gaps:
            continue
        csv_path = _results_csv(run_dir)
        if csv_path is None:
            print(f"  ! {run_dir.name}: no results CSV yet — skipped")
            continue
        runs.append(
            Run(
                path=run_dir,
                arm=match["arm"],
                anchor=int(match["anchor"]),
                gap=int(match["gap"]),
                csv_path=csv_path,
                done=(log_dir / f"{run_dir.name}.done").exists(),
            )
        )
    return runs


# --------------------------------------------------------------------------- #
# Loading and the crossing logic (generalised from bin_results.bin_by_accuracy)
# --------------------------------------------------------------------------- #


def _evaluated_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per actual evaluation, dropping the carried-forward repeats.

    The cadence is inferred: accuracy changes only on evaluated rounds, so the
    modal spacing between changes is `eval_interval`, and those rounds share a
    common residue mod that interval.  Two consecutive evaluations that happen to
    return the identical accuracy would hide a change, which is why the residue —
    not the change itself — selects the rows.

    Falls back to the input unchanged when no repeat structure is present (a run
    that really did evaluate every round).
    """
    if len(df) < 20:
        return df.reset_index(drop=True)
    changes = df.loc[df["accuracy"].diff().fillna(1) != 0, "round"]
    if len(changes) < 5 or len(changes) > 0.9 * len(df):
        return df.reset_index(drop=True)  # already one row per evaluation
    spacing = changes.diff().dropna()
    if spacing.empty:
        return df.reset_index(drop=True)
    interval = int(spacing.mode().iloc[0])
    if interval <= 1:
        return df.reset_index(drop=True)
    residue = int((changes % interval).mode().iloc[0])
    kept = df[df["round"] % interval == residue]
    # Guard: never return a degenerate frame if the inference misfires.
    return (kept if len(kept) >= 10 else df).reset_index(drop=True)


def load_curve(
    csv_path: Path, stop_at: float | None, smooth: int, clean_time: Path | None = None
) -> pd.DataFrame:
    """Evaluated rounds only, accuracy as a percentage, plus the smoothed curve.

    PLATO writes a row per round but evaluates on a cadence, leaving
    ``accuracy = 0`` / empty ``valid_loss`` on the rest; those rows are not
    measurements and would drag the curve to zero.

    ``smooth`` is a centred rolling **median** over that many evaluations.  It is
    not cosmetic: per-round accuracy here swings ±5 points, so "first round at or
    above the target" read off the raw series is one noisy sample — a lucky spike
    can move a crossing by hundreds of rounds and flip an arm comparison.  The
    median ignores spikes without shifting the level the way a mean would.
    ``--smooth 1`` restores the raw reading (what bin_results.py does).
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["accuracy"])
    df = df[df["accuracy"] > 0]
    df["round"] = df["round"].astype(int)
    df = df.sort_values("round").reset_index(drop=True)

    # PLATO writes a row per ROUND but evaluates on `eval_interval`, carrying the
    # last accuracy forward on the rounds in between — 8 995 of 9 990 consecutive
    # differences are exactly 0 in a 10 000-round run.  Those repeats are not
    # measurements.  Left in, `smooth` counts rows rather than evaluations, so the
    # documented "25 evaluations" window actually spans 2.5 of them and the median
    # barely moves; the noise it exists to suppress survives into every crossing.
    # Subsample to the evaluated rounds first, inferring the cadence from the data
    # rather than trusting the config (a resumed run can shift the phase).
    df = _evaluated_rows(df)

    # Accuracy is stored as a fraction here, but older runs used percent.
    # Same normalisation bin_results.py does.
    if not df.empty and df["accuracy"].max() <= 1.0:
        df["accuracy_pct"] = df["accuracy"] * 100.0
    else:
        df["accuracy_pct"] = df["accuracy"]

    df["smooth_pct"] = (
        df["accuracy_pct"].rolling(max(1, smooth), center=True, min_periods=1).median()
    )
    # The reported elapsed_time carries the host's real serialisation and transfer
    # cost inside the simulated clock, and the arms did not share a host — see
    # recompute_wall_time.py.  Prefer the reconstructed clock when it is available.
    df["time_source"] = "reported"
    if clean_time is not None and clean_time.exists():
        clean = pd.read_csv(clean_time)[["round", "clean_elapsed"]]
        merged = df.merge(clean, on="round", how="left")
        if merged["clean_elapsed"].notna().mean() > 0.98:
            df = merged
            df["elapsed_time"] = df["clean_elapsed"]
            df["time_source"] = "clean"

    df["hours"] = df["elapsed_time"] / 3600.0

    if stop_at is not None:
        crossed = df.index[df["accuracy_pct"] >= stop_at * 100.0]
        if len(crossed):
            df = df.loc[: crossed[0]]
    return df


def first_crossing(df: pd.DataFrame, target_pct: float) -> tuple[int, float] | None:
    """(round, elapsed_time seconds) of the first row whose smoothed accuracy
    reaches target_pct.  Same definition the plotted bold line shows, so a
    crossing you read off the figure and one in the table always agree."""
    reached = df[df["smooth_pct"] >= target_pct]
    if reached.empty:
        return None
    row = reached.iloc[0]
    return int(row["round"]), float(row["elapsed_time"])


# --------------------------------------------------------------------------- #
# Effective gap (from summarize_sweep)
# --------------------------------------------------------------------------- #


def effective_gap(gap: int, anchor: int, floor: float, sleep_dir: Path, clients_csv: Path) -> float | None:
    """(c + s_slowest) / (c + s_fastest) for this gap's sleep-time file.

    The nominal G controls only the injected sleep; PLATO charges every client an
    additive floor c on top of it, so this — not G — is the sweep's real x-axis.
    """
    sleep_csv = sleep_dir / f"client_sleep_times_xa{anchor}_gap{gap:02d}.csv"
    if not sleep_csv.exists() or not clients_csv.exists():
        return None
    sleeps = per_tier_sleep(sleep_csv, load_client_tiers(clients_csv))
    slow, fast = sleeps[TIER_ORDER[0]], sleeps[TIER_ORDER[-1]]
    return (floor + slow) / (floor + fast)


# --------------------------------------------------------------------------- #
# What each arm actually selected, and what its predictor could see
# --------------------------------------------------------------------------- #

SELECTION_RE = re.compile(r"\[GpuTierSleep\] round=\d+ client_id=\d+ tier_id=(\w+) .*?sleep_time=([\d.]+)")


def tier_mix(run: Run, log_dir: Path, cache: Path) -> dict | None:
    """Share of dispatches that went to each tier, and their mean sleep time.

    Read from the ``[GpuTierSleep] round=…`` line the client logs on every
    dispatch — the record of what the selector *did*, as opposed to what it
    believed.  With 50/35/15 clients per tier, an arm that cannot tell the tiers
    apart lands on 0.50/0.35/0.15; anything else is steering.

    The logs run to tens of MB, so results are cached in ``figures/tier_mix.csv``
    keyed by run name and the log's size — a growing log re-parses, a finished
    one does not.
    """
    log_path = log_dir / f"{run.name}.log"
    if not log_path.exists():
        return None
    size = log_path.stat().st_size

    rows = {}
    if cache.exists():
        with cache.open(newline="") as f:
            rows = {r["run"]: r for r in csv.DictReader(f)}
        hit = rows.get(run.name)
        if hit and int(hit["log_size"]) == size:
            return {k: (v if k == "run" else float(v)) for k, v in hit.items()}

    counts: dict[str, int] = {}
    total, sleep_sum = 0, 0.0
    with log_path.open(errors="replace") as f:
        for line in f:
            match = SELECTION_RE.search(line)
            if match:
                counts[match[1]] = counts.get(match[1], 0) + 1
                sleep_sum += float(match[2])
                total += 1
    if not total:
        return None

    record = {
        "run": run.name,
        "log_size": size,
        "dispatches": total,
        "mean_sleep": sleep_sum / total,
        **{f"share_{tier}": counts.get(tier, 0) / total for tier in TIER_ORDER},
    }
    rows[run.name] = record
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        writer.writeheader()
        writer.writerows(rows.values())
    return record


def perceived_gap(run: Run, input_dir: Path) -> float | None:
    """Slowest/fastest ratio the arm's predictor believes in.

    The `gpu` arm reads the same tier file the clients sleep by, so its belief
    tracks reality.  The `cpu` arm reads ``tiers_cpu.csv`` — CPU GHz, a fixed
    1.30x spread whatever the real gap is.  `baseline` has no predictor at all.
    """
    if run.arm == "baseline":
        return None
    name = "tiers_cpu.csv" if run.arm == "cpu" else f"tiers_geo_gap{run.gap:02d}.csv"
    path = input_dir / name
    return predictor_gap(path) if path.exists() else None


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def _style_axes(ax, xlabel: str, ylabel: str) -> None:
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


def plot_curves_for_gap(
    runs: list[Run], targets: list[float], out_path: Path, xmax: float | None = None
) -> None:
    """Accuracy vs simulated wall clock, one line per arm, for a single gap."""
    gap = runs[0].gap
    eff = runs[0].effective_gap
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=160)
    fig.patch.set_facecolor(SURFACE)

    for target in targets:
        ax.axhline(target, color=GRID, linewidth=1.0, linestyle=(0, (4, 4)), zorder=1)
        # Left edge, not right: the right margin belongs to the direct labels.
        ax.annotate(
            f"{target:g}%",
            xy=(0.004, target),
            xycoords=("axes fraction", "data"),
            ha="left",
            va="bottom",
            fontsize=8,
            color=INK_MUTED,
        )

    for run in sorted(runs, key=lambda r: ARM_ORDER.index(r.arm)):
        df = run.frame
        if df is None or df.empty:
            continue
        color = ARM_COLOR[run.arm]
        ax.plot(df["hours"], df["accuracy_pct"], color=color, linewidth=0.7, alpha=0.22, zorder=2)
        ax.plot(
            df["hours"],
            df["smooth_pct"],
            color=color,
            linewidth=2.0,
            linestyle="-" if run.done else (0, (6, 3)),
            label=run.label,
            zorder=3,
            solid_capstyle="round",
        )
        # Direct label at the end of the line — identity without reading the legend.
        ax.annotate(
            run.arm,
            xy=(df["hours"].iloc[-1], df["smooth_pct"].iloc[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    _style_axes(ax, "simulated wall clock (hours)", "accuracy (%, rolling median)")
    subtitle = f"nominal gap G={gap}"
    if eff is not None:
        subtitle += f"  ·  effective gap {eff:.2f}×"
    ax.set_title(
        f"Time to accuracy — {subtitle}",
        color=INK,
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=14,
    )
    legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK_2)

    if xmax is not None:
        ax.set_xlim(right=xmax)
    # Room on the right for the direct labels, which sit outside the last point.
    ax.margins(x=0.06)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


# Ordinal ramp (sequential blue, steps 250/450/650) — target accuracy is an
# ordered variable, so it gets a one-hue ramp rather than categorical hues.
TARGET_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def _ramp(index: int, count: int) -> str:
    if count <= 1:
        return TARGET_RAMP[len(TARGET_RAMP) // 2]
    position = index / (count - 1) * (len(TARGET_RAMP) - 1)
    return TARGET_RAMP[round(position)]



def concurrency_trace(log_path: Path) -> tuple[list[int], int]:
    """Clients in flight after each round, and how many extra clients the
    staleness flushes merged (the count that used to be slots lost outright).

    The async server replaces exactly the clients it processed
    (``base.py:589`` — ``choose_clients(selectable, len(current_processed_clients))``),
    and in practice that is one per round.  When a staleness flush merges several
    at once, the next round still dispatches one, so the difference is gone for
    good: nothing in the loop ever refills the pipeline back to
    ``max_concurrency``.  Verified in the logs — round 13 of gpu_xa20_gap40 merges
    six clients and round 14 selects one.

    This matters more than it sounds.  Simulated seconds per round is roughly a
    job's length divided by how many jobs overlap, so an arm that quietly loses
    its parallelism pays for every client serially — which is what makes the
    wall-clock axis at wide gaps a statement about pipeline depth rather than
    about selection quality.
    """
    dispatch, merges, _ = parse_schedule(log_path)
    queues = dispatch_queues(dispatch)
    dispatched: dict[int, int] = {}
    for rounds in dispatch.values():
        for rnd, _ in rounds:
            dispatched[rnd] = dispatched.get(rnd, 0) + 1

    consumed: dict[int, int] = {}
    lost = 0
    for rnd, entries in merges:
        matched = 0
        for client_id, _ in entries:
            queue = queues.get(client_id)
            if queue:
                queue.popleft()
                matched += 1
        consumed[rnd] = matched
        if matched > 1:
            lost += matched - 1

    trace, live = [], 0
    for rnd in range(1, (max(consumed) if consumed else 0) + 1):
        live += dispatched.get(rnd, 0) - consumed.get(rnd, 0)
        trace.append(live)
    return trace, lost


def plot_concurrency(runs: list[Run], log_dir: Path, out_path: Path) -> None:
    """How much of the configured concurrency each arm actually keeps."""
    traces = {}
    for run in runs:
        log_path = log_dir / f"{run.name}.log"
        if log_path.exists():
            traces[(run.arm, run.gap)] = concurrency_trace(log_path)
    if not traces:
        return

    gaps = sorted({gap for _, gap in traces})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)

    # -- left: the collapse over rounds, at the widest gap ------------------ #
    ax = axes[0]
    widest = gaps[-1]
    for arm in ARM_ORDER:
        entry = traces.get((arm, widest))
        if not entry:
            continue
        trace = entry[0][:1500]
        ax.plot(range(1, len(trace) + 1), trace, color=ARM_COLOR[arm], linewidth=1.8, zorder=3)
        ax.annotate(
            f"{arm} → {trace[-1]}", xy=(len(trace), trace[-1]), xytext=(6, 0),
            textcoords="offset points", va="center", fontsize=9,
            color=ARM_COLOR[arm], fontweight="bold",
        )
    _style_axes(ax, "round", "clients training at once")
    ax.set_title(
        f"the pipeline drains and never refills  ·  G={widest}",
        color=INK, fontsize=11, fontweight="bold", loc="left",
    )
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.12)

    # -- right: where each arm settles, by gap ------------------------------ #
    ax = axes[1]
    position = {gap: index for index, gap in enumerate(gaps)}
    for arm in ARM_ORDER:
        points = []
        for gap in gaps:
            entry = traces.get((arm, gap))
            if not entry or not entry[0]:
                continue
            tail = entry[0][len(entry[0]) // 5:]
            points.append((position[gap], statistics.median(tail)))
        if not points:
            continue
        ax.plot(
            [p[0] for p in points], [p[1] for p in points], color=ARM_COLOR[arm],
            linewidth=2.0, marker="o", markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=2.0, zorder=3,
        )
        ax.annotate(
            arm, xy=points[-1], xytext=(7, 0), textcoords="offset points", va="center",
            fontsize=9, color=ARM_COLOR[arm], fontweight="bold",
        )
    _style_axes(ax, "", "steady-state clients training at once")
    ax.set_title(
        "what is left of max_concurrency = 10", color=INK, fontsize=11, fontweight="bold", loc="left"
    )
    ax.set_xticks(range(len(gaps)))
    ax.set_xticklabels([f"G={gap}" for gap in gaps], fontsize=8.5)
    ax.set_xlim(-0.35, len(gaps) - 1 + 0.45)
    ax.set_ylim(0, 10.5)
    ax.axhline(10, color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 4)), zorder=1)
    ax.annotate(
        "configured max_concurrency", xy=(0.5, 9.85), xycoords=("axes fraction", "data"),
        ha="center", va="top", fontsize=8.5, color=INK_MUTED,
    )

    fig.suptitle(
        "Effective concurrency — why the wall-clock axis is not about selection",
        color=INK, fontsize=13, fontweight="bold", x=0.006, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_hypothesis(
    runs: list[Run], records: list[dict], targets: list[float], out_path: Path
) -> None:
    """Four panels testing one claim, left to right, premise to consequence.

    The claim under test: the `cpu` arm should fall behind as the gap widens,
    because CPU clock speed barely separates the tiers while their real speeds
    separate more and more.

    Panel 1 is the premise (what each predictor can see), panel 2 is what that
    does to selection, panels 3 and 4 are what it costs — in rounds, then in
    simulated wall clock.  Splitting the cost in two is the point: the two
    answers differ, and a single time-to-accuracy number hides that.

    The x axis is the five sweep configurations, evenly spaced.  Effective gap
    saturates (1.0, 5.5, 7.3, 8.2, 8.8), so plotting it to scale crushes the top
    four points together and the tick labels collide; the tick text carries both
    numbers instead.  Slopes here are therefore ordinal, not quantitative.
    """
    steps = sorted({(r.gap, r.effective_gap) for r in runs if r.effective_gap is not None})
    if not steps:
        return
    position = {gap: index for index, (gap, _) in enumerate(steps)}
    labels = [f"G={gap}\n{eff:.1f}×" for gap, eff in steps]

    fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)

    # -- 1. what the predictor can see ------------------------------------- #
    ax = axes[0]
    ax.plot(
        [position[g] for g, _ in steps], [eff for _, eff in steps],
        color=INK_MUTED, linewidth=1.5, linestyle=(0, (5, 4)), zorder=2,
    )
    ax.annotate(
        "reality", xy=(len(steps) - 1, steps[-1][1]), xytext=(-4, 7),
        textcoords="offset points", ha="right", fontsize=8.5, color=INK_MUTED,
    )
    for arm in ("gpu", "cpu"):
        points = sorted(
            (position[r.gap], r.perceived) for r in runs if r.arm == arm and r.perceived
        )
        if not points:
            continue
        ax.plot(
            [p[0] for p in points], [p[1] for p in points], color=ARM_COLOR[arm], linewidth=2.0,
            marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=3,
        )
        ax.annotate(
            arm, xy=points[-1], xytext=(7, 0), textcoords="offset points", va="center",
            fontsize=9, color=ARM_COLOR[arm], fontweight="bold",
        )
    _style_axes(ax, "", "spread the predictor believes in (×)")
    ax.set_title("1 · the signal", color=INK, fontsize=11, fontweight="bold", loc="left")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 40])
    ax.set_yticklabels(["1×", "2×", "5×", "10×", "20×", "40×"])

    # -- 2. what it selects ------------------------------------------------- #
    ax = axes[1]
    ax.axhline(0.50, color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 4)), zorder=1)
    ax.annotate(
        "pool share — no steering",
        xy=(0.97, 0.498), xycoords=("axes fraction", "data"), ha="right", va="top",
        fontsize=8.5, color=INK_MUTED,
    )
    for arm in ARM_ORDER:
        points = sorted(
            (position[r.gap], r.mix[f"share_{TIER_ORDER[0]}"])
            for r in runs
            if r.arm == arm and r.mix and r.gap in position
        )
        if not points:
            continue
        ax.plot(
            [p[0] for p in points], [p[1] for p in points], color=ARM_COLOR[arm], linewidth=2.0,
            marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=3,
            linestyle="-" if len(points) > 2 else (0, (6, 3)),
        )
        ax.annotate(
            arm, xy=points[-1], xytext=(7, 0), textcoords="offset points", va="center",
            fontsize=9, color=ARM_COLOR[arm], fontweight="bold",
        )
    _style_axes(ax, "", f"share of dispatches to {TIER_ORDER[0]} (slowest tier)")
    ax.set_title("2 · what it picks", color=INK, fontsize=11, fontweight="bold", loc="left")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")

    # -- 3 & 4. what it costs ---------------------------------------------- #
    for index, (ax, field, unit) in enumerate(
        ((axes[2], "first_round", "rounds"), (axes[3], "hours", "hours"))
    ):
        ax.axhspan(1.0, 10.0, color=ARM_COLOR["cpu"], alpha=0.05, zorder=0)
        ax.axhline(1.0, color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 4)), zorder=1)
        for order, target in enumerate(targets):
            points = []
            for gap, eff in steps:
                pair = {
                    r["arm"]: r[field]
                    for r in records
                    if r["target_pct"] == target and r["gap"] == gap and r[field]
                    and r["arm"] in ("cpu", "gpu")
                }
                if len(pair) == 2:
                    points.append((position[gap], pair["cpu"] / pair["gpu"]))
            if not points:
                continue
            ax.plot(
                [p[0] for p in points], [p[1] for p in points], color=_ramp(order, len(targets)),
                linewidth=2.0, marker="o", markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2.0, zorder=3, label=f"target {target:g}%",
            )
        _style_axes(ax, "", f"cpu ÷ gpu · {unit} to target")
        ax.set_title(
            f"{3 + index} · cost in {unit}", color=INK, fontsize=11, fontweight="bold", loc="left"
        )
        # Log scale: on a ratio axis 2x and 0.5x are the same size of effect, and
        # a linear axis would draw the first as four times the second.
        ax.set_yscale("log")
        ax.set_yticks([0.5, 0.7, 1.0, 1.4, 2.0])
        ax.set_yticklabels(["0.5×", "0.7×", "1.0×", "1.4×", "2.0×"])
        # A log axis relabels its minor ticks too, leaving a stray "6 x 10^-1".
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(0.45, 2.2)
        ax.annotate(
            "cpu worse ↑", xy=(0.02, 0.97), xycoords="axes fraction", fontsize=8.5,
            color=ARM_COLOR["cpu"], va="top",
        )
        ax.annotate(
            "cpu better ↓", xy=(0.02, 0.03), xycoords="axes fraction", fontsize=8.5,
            color=INK_MUTED, va="bottom",
        )
        if index == 0:
            legend = ax.legend(frameon=False, fontsize=8.5, loc="lower right", handlelength=1.4)
            for text in legend.get_texts():
                text.set_color(INK_2)

    for ax in axes:
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_xlim(-0.35, len(steps) - 1 + 0.45)

    fig.suptitle(
        "Does the cpu arm lose ground as heterogeneity widens?",
        color=INK, fontsize=13, fontweight="bold", x=0.006, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_tta_vs_gap(records: list[dict], targets: list[float], out_path: Path) -> None:
    """Hours to reach each target vs effective gap — one panel per target."""
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), dpi=160, squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    # Every panel shares one x range so the panels can be read against each other;
    # a single-gap sweep would otherwise get an axis zoomed onto floating-point dust.
    xs = sorted({r["x"] for r in records})
    span = (xs[-1] - xs[0]) or 1.0
    xlim = (xs[0] - 0.08 * span, xs[-1] + 0.08 * span)

    for index, (ax, target) in enumerate(zip(axes[0], targets)):
        for arm in ARM_ORDER:
            points = sorted(
                (r for r in records if r["arm"] == arm and r["target_pct"] == target and r["hours"] is not None),
                key=lambda r: r["x"],
            )
            if not points:
                continue
            ax.plot(
                [p["x"] for p in points],
                [p["hours"] for p in points],
                color=ARM_COLOR[arm],
                linewidth=2.0,
                marker="o",
                markersize=8,
                markeredgecolor=SURFACE,
                markeredgewidth=2.0,
                label=ARM_LABEL[arm],
                zorder=3,
                solid_capstyle="round",
            )
        _style_axes(ax, "effective gap (×)", "hours to reach target" if index == 0 else "")
        ax.set_title(f"target {target:g}%", color=INK, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylim(bottom=0)
        ax.set_xlim(*xlim)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:.2f}" for x in xs])

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        legend = fig.legend(
            handles,
            labels,
            frameon=False,
            fontsize=9,
            loc="lower center",
            ncols=len(handles),
            bbox_to_anchor=(0.5, -0.02),
        )
        for text in legend.get_texts():
            text.set_color(INK_2)

    fig.suptitle(
        "Time to accuracy vs heterogeneity",
        color=INK,
        fontsize=13,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _stop_at(value: str) -> float | None:
    if value.lower() in {"none", "off", ""}:
        return None
    return float(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="where .done markers live")
    parser.add_argument("--sleep-dir", type=Path, default=DEFAULT_SLEEP_DIR)
    parser.add_argument("--clients-csv", type=Path, default=DEFAULT_CLIENTS_CSV)
    parser.add_argument("--arms", nargs="+", default=ARM_ORDER, choices=ARM_ORDER)
    parser.add_argument("--gaps", nargs="+", default=None, help="e.g. --gaps 01 10 20 (default: all found)")
    parser.add_argument(
        "--targets",
        nargs="+",
        type=float,
        default=[40.0, 45.0, 50.0, 55.0],
        help="target accuracies in %% (default: 40 45 50 55)",
    )
    parser.add_argument(
        "--reference",
        default="baseline",
        choices=ARM_ORDER,
        help="arm the speedup ratio is taken against (default: baseline; falls back to gpu)",
    )
    parser.add_argument(
        "--stop-at",
        type=_stop_at,
        default=0.6,
        help="truncate every curve at its first crossing of this accuracy fraction, "
        "or 'none' (default: 0.6 — matches target_accuracy)",
    )
    parser.add_argument("--floor", type=float, default=DEFAULT_FLOOR_SECONDS, help="additive per-client floor c")
    parser.add_argument(
        "--xmax",
        type=float,
        default=None,
        help="clip the per-gap curves at this many simulated hours — the interesting "
        "part is the climb, and a saturated tail can be most of the axis",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=25,
        help="centred rolling-median window, in evaluations, used for the plotted "
        "curve and every crossing (default: 25; 1 = raw, what bin_results.py reads)",
    )
    parser.add_argument("--input-dir", type=Path, default=_MODULE_DIR / "input")
    parser.add_argument(
        "--time-source",
        choices=("clean", "reported"),
        default="clean",
        help="'clean' uses the clock rebuilt by recompute_wall_time.py, which charges "
        "every arm the same per-report cost; 'reported' uses PLATO's elapsed_time, "
        "which includes each host's real payload handling (default: clean, falling "
        "back to reported per run when no reconstruction exists)",
    )
    parser.add_argument(
        "--clean-time-dir", type=Path, default=_MODULE_DIR / "figures" / "clean_time"
    )
    parser.add_argument(
        "--no-tier-mix",
        action="store_true",
        help="skip parsing the server logs for what each arm selected (the slow part)",
    )
    parser.add_argument("--no-plots", action="store_true", help="table only")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"Runs dir: {args.runs_dir}")
    runs = discover_runs(args.runs_dir, args.log_dir, args.arms, args.gaps)
    if not runs:
        raise SystemExit(f"No sweep runs found under {args.runs_dir}.")

    for run in runs:
        clean_path = None if args.time_source == "reported" else args.clean_time_dir / f"{run.name}.csv"
        run.frame = load_curve(run.csv_path, args.stop_at, args.smooth, clean_path)
        run.effective_gap = effective_gap(
            run.gap, run.anchor, args.floor, args.sleep_dir, args.clients_csv
        )
        run.perceived = perceived_gap(run, args.input_dir)
        if not args.no_tier_mix:
            run.mix = tier_mix(run, args.log_dir, args.out_dir / "tier_mix.csv")

    sources = {run.frame["time_source"].iloc[0] for run in runs if run.frame is not None and not run.frame.empty}
    if args.time_source == "clean" and "reported" in sources:
        missing = [r.name for r in runs if r.frame is not None and not r.frame.empty
                   and r.frame["time_source"].iloc[0] == "reported"]
        print(f"  ! no reconstructed clock for {', '.join(missing)} — using PLATO's reported "
              f"elapsed_time there.  Run recompute_wall_time.py to fix.")
    print(f"Time axis: {'/'.join(sorted(sources))} simulated clock")

    stop_note = "none" if args.stop_at is None else f"{args.stop_at:.2f}"
    print(f"Found {len(runs)} run(s); floor c = {args.floor:.2f} s; curves truncated at {stop_note}\n")

    # ---- long-format table ------------------------------------------------ #
    records: list[dict] = []
    for run in runs:
        df = run.frame
        for target in args.targets:
            crossing = first_crossing(df, target) if df is not None else None
            records.append(
                {
                    "run": run.name,
                    "arm": run.arm,
                    "gap": run.gap,
                    "effective_gap": run.effective_gap,
                    "x": run.effective_gap if run.effective_gap is not None else float(run.gap),
                    "target_pct": target,
                    "first_round": crossing[0] if crossing else None,
                    "seconds": crossing[1] if crossing else None,
                    "hours": crossing[1] / 3600.0 if crossing else None,
                    "best_pct": float(df["accuracy_pct"].max()) if df is not None and not df.empty else None,
                    "rounds_run": int(df["round"].iloc[-1]) if df is not None and not df.empty else 0,
                    "finished": run.done,
                }
            )

    # Ratio against the reference arm, per (gap, target).  <1 means faster.
    for (gap, target), group in _group_by(records, lambda r: (r["gap"], r["target_pct"])).items():
        ref_arm = args.reference if any(r["arm"] == args.reference and r["hours"] for r in group) else "gpu"
        ref = next((r["hours"] for r in group if r["arm"] == ref_arm and r["hours"]), None)
        for record in group:
            record["ref_arm"] = ref_arm if ref else None
            record["ratio_vs_ref"] = record["hours"] / ref if ref and record["hours"] else None

    _print_table(records, args.targets)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "tta_summary.csv"
    fields = [
        "run", "arm", "gap", "effective_gap", "target_pct", "first_round", "seconds", "hours",
        "ref_arm", "ratio_vs_ref", "best_pct", "rounds_run", "finished",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"\nWrote {summary_path}")

    if args.no_plots:
        return

    for gap, gap_runs in sorted(_group_by(runs, lambda r: r.gap).items()):
        out_path = args.out_dir / f"tta_gap{gap:02d}.png"
        plot_curves_for_gap(gap_runs, args.targets, out_path, xmax=args.xmax)
        print(f"Wrote {out_path}")

    out_path = args.out_dir / "tta_vs_gap.png"
    plot_tta_vs_gap(records, args.targets, out_path)
    print(f"Wrote {out_path}")

    if any(r.arm == "cpu" for r in runs) and any(r.arm == "gpu" for r in runs):
        out_path = args.out_dir / "hypothesis_cpu_vs_gpu.png"
        plot_hypothesis(runs, records, args.targets, out_path)
        print(f"Wrote {out_path}")

    if not args.no_tier_mix:
        out_path = args.out_dir / "concurrency.png"
        plot_concurrency(runs, args.log_dir, out_path)
        print(f"Wrote {out_path}")


def _group_by(items, key):
    grouped: dict = {}
    for item in items:
        grouped.setdefault(key(item), []).append(item)
    return grouped


def _print_table(records: list[dict], targets: list[float]) -> None:
    """Hours-to-target per arm, one block per gap."""
    for gap, group in sorted(_group_by(records, lambda r: r["gap"]).items()):
        eff = next((r["effective_gap"] for r in group if r["effective_gap"] is not None), None)
        header = f"G={gap:02d}" + (f"  (effective {eff:.2f}x)" if eff else "")
        print(header)
        print(
            f"  {'arm':<10}{'status':<12}{'best%':>7}{'rounds':>8}"
            + "".join(f"{('h@' + f'{t:g}%'):>14}" for t in targets)
        )
        for arm in ARM_ORDER:
            arm_rows = [r for r in group if r["arm"] == arm]
            if not arm_rows:
                continue
            first = arm_rows[0]
            status = "done" if first["finished"] else "running"
            cells = []
            for target in targets:
                row = next((r for r in arm_rows if r["target_pct"] == target), None)
                if row is None or row["hours"] is None:
                    cells.append(f"{'—':>12}")
                else:
                    ratio = row.get("ratio_vs_ref")
                    cell = f"{row['hours']:.2f}"
                    if ratio and row["ref_arm"] != arm:
                        cell += f" ({ratio:.2f}x)"
                    cells.append(f"{cell:>14}")
            best = first["best_pct"]
            print(
                f"  {arm:<10}{status:<12}{(best if best is not None else math.nan):>7.2f}"
                f"{first['rounds_run']:>8}" + "".join(cells)
            )
        print()


if __name__ == "__main__":
    sys.exit(main())
