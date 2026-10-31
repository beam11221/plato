"""
Tier selection over time: dispatches per time bin and GPU tier, one stacked-area panel per method.

    /workspace/plato/.venv/bin/python wen_env/tools/tier_dispatch_timeline.py \
        [--scenario S1] [--methods ours,wen_fresh,random] [--bin 60] [--smooth 3] [--horizon auto]

Reads runs/wen_env/<cell>_s<seed>/dispatch_log.csv (seed 4126 from _phaseA/ when it is there, as
wen_ptable.run_dir). Each dispatch is binned by t_dispatch (simulation time) and counted by its logged
tier. Horizon "auto" = the earliest last dispatch over all runs of all methods, rounded down to a whole
bin, so every bin averages the same seeds. Writes, to wen_env/evidence/figs_<scenario>/:
    <scenario>_tier_timeline.pdf / .png
    <scenario>_tier_timeline_per_seed.csv   (unsmoothed counts per method, seed, bin, tier)
    <scenario>_tier_timeline_mean.csv       (unsmoothed seed means per method, bin, tier)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "runs" / "wen_env"
SEEDS = (4126, 4127, 4128)
TIERS = ("xavier", "orin", "thor")  # stacked bottom -> top
TIER_LABEL = {"xavier": "Xavier", "orin": "Orin", "thor": "Thor"}
TIER_COLOR = {"xavier": "#e8382c", "orin": "#d97a3f", "thor": "#2f6b34"}
# key -> (cell suffix, panel title)
METHODS = {
    "ours": ("gpu", "Ours"),
    "wen_fresh": ("wen_fresh", "RA-MaOVSM (Wen et al.)"),
    "random": ("random", "Random"),
    "polaris": ("polaris", "Polaris"),
}
# existing bar chart (S1, seed 4126, up to TTA@90 %), for the validation table
REFERENCE = {"ours": (45.8, 33.2, 20.9), "wen_fresh": (19.9, 79.0, 1.1), "random": (46.7, 35.8, 17.4)}


def run_dir(cell: str, seed: int) -> Path:
    p = RUNS / "_phaseA" / f"{cell}_s{seed}"
    return p if seed == 4126 and p.exists() else RUNS / f"{cell}_s{seed}"


def load_counts(sc: str, keys: list[str], bin_s: float, horizon_arg: str = "auto",
                verbose: bool = True) -> tuple[pd.DataFrame, float]:
    """Dispatch counts per (method, seed, bin, tier), every bin present (0 when empty), and the horizon."""
    logs, ends = {}, {}
    for k in keys:
        for s in SEEDS:
            d = run_dir(f"{sc}_{METHODS[k][0]}", s)
            dl = pd.read_csv(d / "dispatch_log.csv", usecols=["t_dispatch", "tier", "veh_id"])
            if dl["t_dispatch"].isna().any() or dl["tier"].isna().any():
                raise SystemExit(f"{d}: missing t_dispatch or tier")
            bad = set(dl["tier"]) - set(TIERS)
            if bad:
                raise SystemExit(f"{d}: unknown tiers {bad}")
            logs[(k, s)] = dl
            ends[(k, s)] = dl["t_dispatch"].max()
            if verbose:
                print(f"{k:10s} s{s}  {d.relative_to(RUNS)}  dispatches {len(dl)}  "
                      f"last dispatch {ends[(k, s)]:.1f} s")

    if horizon_arg == "auto":
        first = min(ends, key=ends.get)
        horizon = np.floor(ends[first] / bin_s) * bin_s
        print(f"\nhorizon: {horizon:.0f} s (earliest last dispatch {ends[first]:.1f} s in "
              f"{first[0]} s{first[1]}, rounded down to a whole {bin_s:.0f} s bin)")
    else:
        horizon = float(horizon_arg)
        print(f"\nhorizon: {horizon:.0f} s (given)")
    n_bins = int(round(horizon / bin_s))
    edges = np.arange(n_bins + 1) * bin_s

    rows = []
    for (k, s), dl in logs.items():
        dl = dl[dl["t_dispatch"] < horizon]
        b = np.floor(dl["t_dispatch"] / bin_s).astype(int)
        for t in TIERS:
            cnt = np.bincount(b[dl["tier"] == t], minlength=n_bins)[:n_bins]
            rows.append(pd.DataFrame({"method": k, "seed": s, "bin": np.arange(n_bins),
                                      "t_start": edges[:-1], "tier": t, "count": cnt}))
    return pd.concat(rows, ignore_index=True), horizon


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scenario", default="S1", choices=("S1", "S2"))
    ap.add_argument("--methods", default="ours,wen_fresh,random",
                    help=f"comma-separated, from {list(METHODS)}")
    ap.add_argument("--bin", type=float, default=60.0, help="bin width in simulated seconds")
    ap.add_argument("--smooth", type=int, default=3, help="centered rolling mean over this many bins "
                    "(display only; 1 = off)")
    ap.add_argument("--horizon", default="auto", help="seconds, or 'auto'")
    ap.add_argument("--out", type=Path, default=None, help="output directory")
    a = ap.parse_args()

    keys = a.methods.split(",")
    for k in keys:
        if k not in METHODS:
            raise SystemExit(f"unknown method {k!r}")
    sc = a.scenario
    out = a.out or HERE.parent / "evidence" / f"figs_{sc.lower()}"

    per_seed, horizon = load_counts(sc, keys, a.bin, a.horizon)
    edges = np.arange(int(round(horizon / a.bin)) + 1) * a.bin
    mean =(per_seed.groupby(["method", "bin", "t_start", "tier"], sort=False)["count"]
            .mean().rename("mean_count").reset_index())

    out.mkdir(parents=True, exist_ok=True)
    stem = f"{sc.lower()}_tier_timeline"
    per_seed.to_csv(out / f"{stem}_per_seed.csv", index=False)
    mean.to_csv(out / f"{stem}_mean.csv", index=False)

    # plot
    fig, axes = plt.subplots(1, len(keys), figsize=(14 * len(keys) / 3, 3.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    mid = edges[:-1] + a.bin / 2
    for i, (ax, k) in enumerate(zip(axes, keys)):
        m = mean[mean["method"] == k]
        ys = [m[m["tier"] == t]["mean_count"].rolling(a.smooth, center=True, min_periods=1).mean().values
              for t in TIERS]
        ax.stackplot(mid, ys, colors=[TIER_COLOR[t] for t in TIERS],
                     labels=[TIER_LABEL[t] for t in TIERS], linewidth=0)
        ax.set_title(f"(b{i + 1}) {METHODS[k][1]}", loc="left", fontsize=11)
        ax.set_xlabel("Wall-clock time (s)")
        ax.set_xlim(0, horizon)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(f"Dispatches / {a.bin:.0f} s")
    axes[0].set_ylim(0, None)
    axes[-1].legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")

    # validation: overall tier share of the unsmoothed, horizon-cut counts (all seeds summed)
    print(f"\ntier share over [0, {horizon:.0f}) s, seeds {SEEDS} pooled  (reference: existing bar chart)")
    print("| Method | Xavier | Orin | Thor | reference X / O / T | max diff (pp) |")
    print("|---|---|---|---|---|---|")
    for k in keys:
        tot = per_seed[per_seed["method"] == k].groupby("tier")["count"].sum().reindex(TIERS)
        sh = tot / tot.sum() * 100
        ref = REFERENCE.get(k)
        diff = max(abs(x - r) for x, r in zip(sh, ref)) if ref else None
        flag = "" if diff is None else f"{diff:.1f}" + (" **> 1 pp**" if diff > 1 else "")
        refs = " / ".join(f"{r:.1f}" for r in ref) if ref else "—"
        print(f"| {METHODS[k][1]} | " + " | ".join(f"{x:.1f}" for x in sh) + f" | {refs} | {flag or '—'} |")
    print(f"\nwrote {out / stem}.pdf/.png, {stem}_per_seed.csv, {stem}_mean.csv")


if __name__ == "__main__":
    main()
