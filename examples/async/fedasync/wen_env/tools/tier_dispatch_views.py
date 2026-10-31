"""
How many vehicles of each GPU tier each method dispatches in every time window, vs simulated time.
Several views of the same counts, so they can be compared side by side:

    lines       one panel per method, one line per tier (dispatches / bin); band = min-max over seeds
    bytier      one panel per tier, one line per method: who picks more Thor (Orin, Xavier) and when
    pervehicle  as `lines`, divided by the tier's fleet size (dispatches / vehicle / bin);
                uniform random selection puts all three tiers on the same level
    share       one panel per method, share of each tier in the window (100 % stacked); dotted = fleet
    heatmap     rows = method x tier, columns = time windows, colour = mean dispatches / bin

    /workspace/plato/.venv/bin/python wen_env/tools/tier_dispatch_views.py \
        [--scenario S1] [--methods ours,wen_fresh,polaris,random] [--bin 60] [--smooth 3] \
        [--horizon auto] [--views lines,bytier,pervehicle,share,heatmap]

Counts come from tier_dispatch_timeline.load_counts (t_dispatch binned, logged tier, every bin present,
common horizon = earliest last dispatch over all runs). Lines are the seed mean; smoothing (centered
rolling mean over --smooth bins) is for display only and is applied to each seed before the band.
Writes <scenario>_tier_view_<view>.pdf / .png and <scenario>_tier_view_counts.csv (unsmoothed) to
wen_env/evidence/figs_<scenario>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tier_dispatch_timeline import METHODS, SEEDS, TIER_COLOR, TIER_LABEL, TIERS, load_counts

HERE = Path(__file__).resolve().parent
FLEET = {"xavier": 15, "orin": 10, "thor": 5}  # n30_x15-o10-t5
METHOD_COLOR = {"ours": "#2a78d6", "wen_fresh": "#eda100", "polaris": "#d6336c",
                "random": "#1baf7a"}  # as in notebooks/s1_curves_selection.ipynb
VIEWS = ("lines", "bytier", "pervehicle", "share", "heatmap")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def series(per_seed: pd.DataFrame, k: str, t: str, smooth: int, scale: float = 1.0) -> np.ndarray:
    """(seeds x bins) counts of one method and tier, smoothed per seed, divided by scale."""
    g = per_seed[(per_seed["method"] == k) & (per_seed["tier"] == t)]
    m = g.pivot(index="seed", columns="bin", values="count").reindex(SEEDS)
    return m.T.rolling(smooth, center=True, min_periods=1).mean().T.to_numpy() / scale


def band(ax, x, ys, color, label=None, lw=1.6):
    ax.fill_between(x, ys.min(0), ys.max(0), color=color, alpha=0.18, linewidth=0)
    ax.plot(x, ys.mean(0), color=color, lw=lw, label=label)


def finish(fig, out, stem):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / stem}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scenario", default="S1", choices=("S1", "S2"))
    ap.add_argument("--methods", default="ours,wen_fresh,polaris,random",
                    help=f"comma-separated, from {list(METHODS)}")
    ap.add_argument("--bin", type=float, default=60.0, help="window width in simulated seconds")
    ap.add_argument("--smooth", type=int, default=3, help="display smoothing in bins (1 = off)")
    ap.add_argument("--horizon", default="auto", help="seconds, or 'auto'")
    ap.add_argument("--views", default=",".join(VIEWS), help=f"comma-separated, from {VIEWS}")
    ap.add_argument("--out", type=Path, default=None, help="output directory")
    a = ap.parse_args()

    keys = a.methods.split(",")
    views = a.views.split(",")
    for k in keys:
        if k not in METHODS:
            raise SystemExit(f"unknown method {k!r}")
    for v in views:
        if v not in VIEWS:
            raise SystemExit(f"unknown view {v!r}")
    sc = a.scenario
    out = a.out or HERE.parent / "evidence" / f"figs_{sc.lower()}"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{sc.lower()}_tier_view"

    per_seed, horizon = load_counts(sc, keys, a.bin, a.horizon, verbose=False)
    per_seed.to_csv(out / f"{stem}_counts.csv", index=False)
    n_bins = per_seed["bin"].max() + 1
    x = (np.arange(n_bins) + 0.5) * a.bin
    unit = f"{a.bin:.0f} s"
    title = {k: METHODS[k][1] for k in keys}
    n = len(keys)

    if "lines" in views or "pervehicle" in views:
        for view in [v for v in ("lines", "pervehicle") if v in views]:
            fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 3.4), sharex=True, sharey=True)
            for i, (ax, k) in enumerate(zip(np.atleast_1d(axes), keys)):
                for t in TIERS:
                    scale = FLEET[t] if view == "pervehicle" else 1.0
                    lab = f"{TIER_LABEL[t]} ({FLEET[t]} veh.)" if view == "pervehicle" else TIER_LABEL[t]
                    band(ax, x, series(per_seed, k, t, a.smooth, scale), TIER_COLOR[t], lab)
                ax.set_title(f"({chr(97 + i)}) {title[k]}", loc="left", fontsize=10)
                ax.set_xlabel("Simulated time (s)")
                ax.set_xlim(0, horizon)
                ax.set_ylim(0, None)
                ax.grid(alpha=0.3)
            axes = np.atleast_1d(axes)
            axes[0].set_ylabel(f"Dispatches / vehicle / {unit}" if view == "pervehicle"
                               else f"Dispatches / {unit}")
            axes[-1].legend(loc="upper right", fontsize=8.5)
            finish(fig, out, f"{stem}_{view}")

    if "bytier" in views:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True, sharey=True)
        for ax, t in zip(axes, TIERS):
            for k in keys:
                band(ax, x, series(per_seed, k, t, a.smooth), METHOD_COLOR[k], title[k])
            ax.set_title(f"{TIER_LABEL[t]} ({FLEET[t]} vehicles)", loc="left", fontsize=10,
                         color=TIER_COLOR[t])
            ax.set_xlabel("Simulated time (s)")
            ax.set_xlim(0, horizon)
            ax.set_ylim(0, None)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(f"Dispatches / {unit}")
        axes[-1].legend(loc="upper right", fontsize=8.5)
        finish(fig, out, f"{stem}_bytier")

    if "share" in views:
        fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 3.4), sharex=True, sharey=True)
        fleet_cum = np.cumsum([FLEET[t] for t in TIERS]) / sum(FLEET.values()) * 100
        for i, (ax, k) in enumerate(zip(np.atleast_1d(axes), keys)):
            m = np.array([series(per_seed, k, t, a.smooth).mean(0) for t in TIERS])
            tot = m.sum(0)
            sh = np.divide(m, tot, out=np.zeros_like(m), where=tot > 0) * 100
            ax.stackplot(x, sh, colors=[TIER_COLOR[t] for t in TIERS],
                         labels=[TIER_LABEL[t] for t in TIERS], linewidth=0, alpha=0.9)
            for c in fleet_cum[:-1]:
                ax.axhline(c, color="white", lw=1.0, ls=":")
            ax.set_title(f"({chr(97 + i)}) {title[k]}", loc="left", fontsize=10)
            ax.set_xlabel("Simulated time (s)")
            ax.set_xlim(0, horizon)
            ax.set_ylim(0, 100)
        axes = np.atleast_1d(axes)
        axes[0].set_ylabel(f"Share of dispatches in the {unit} window (%)")
        h, l = axes[-1].get_legend_handles_labels()
        axes[-1].legend(h[::-1], l[::-1], loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8.5)
        fig.text(0.01, -0.03, "Bands stacked Xavier (bottom), Orin, Thor (top); dotted lines = fleet "
                 "split 50 / 83 %.", fontsize=8, color="#52514e")
        finish(fig, out, f"{stem}_share")

    if "heatmap" in views:
        mean = per_seed.groupby(["method", "tier", "bin"])["count"].mean()
        rows, labels = [], []
        for k in keys:
            for t in TIERS:
                rows.append(mean.loc[(k, t)].reindex(range(n_bins)).to_numpy())
                labels.append(f"{title[k]} · {TIER_LABEL[t]}")
        z = np.array(rows)
        fig, ax = plt.subplots(figsize=(13, 0.32 * len(rows) + 1.3))
        im = ax.imshow(z, aspect="auto", cmap="viridis", interpolation="nearest",
                       extent=(0, horizon, len(rows) - 0.5, -0.5))
        ax.set_yticks(range(len(rows)), labels, fontsize=8.5)
        for j in range(3, len(rows), 3):
            ax.axhline(j - 0.5, color="white", lw=2)
        ax.set_xlabel("Simulated time (s)")
        ax.spines[["left", "bottom"]].set_visible(False)
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label(f"Dispatches / {unit} (mean of {len(SEEDS)} seeds)")
        finish(fig, out, f"{stem}_heatmap")

    # numbers behind the figures: mean dispatches / bin per method and tier, early vs late
    q = n_bins // 4
    print(f"\nmean dispatches / {unit} (seed mean, unsmoothed), first vs last quarter of [0, {horizon:.0f}) s")
    print("| method | " + " | ".join(f"{TIER_LABEL[t]} first / last" for t in TIERS) + " | total |")
    print("|---|---|---|---|---|")
    for k in keys:
        cells = []
        for t in TIERS:
            s = per_seed[(per_seed["method"] == k) & (per_seed["tier"] == t)]
            mb = s.groupby("bin")["count"].mean()
            cells.append(f"{mb.iloc[:q].mean():.2f} / {mb.iloc[-q:].mean():.2f}")
        tot = per_seed[per_seed["method"] == k].groupby("bin")["count"].sum().mean() / len(SEEDS)
        print(f"| {title[k]} | " + " | ".join(cells) + f" | {tot:.2f} |")


if __name__ == "__main__":
    main()
