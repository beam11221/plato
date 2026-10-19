"""
Analysis for the COMMUNICATION-LATENCY sweep (BRIEF_comm_sweep.md).

The H-sweep and fleet-mix sweep both model training time as compute only.  This
sweep adds the uplink transfer cost C = D/B on top, holding the fleet at
50/35/15 and the tier-speed gap at G=30, and asks where the GPU-TOPS predictor's
advantage collapses as bandwidth falls — because an additive constant common to
every client pushes every ratio toward 1 (BRIEF § 1).

    B (Mbps)   inf   100    50    20    10
    C (s)        0  3.42  6.83 17.08 34.15

B=inf is the reused `m50_35_15` cell of the fleet-mix sweep at G=30
(runs/mix_sweep/*_m50_35_15) — it was already run under identical settings
(t_round_model="queue", rounds=5000, target_accuracy=1.01, seed 8888) and is NOT
regenerated here (BRIEF § 3).  Everything else lives in runs/comm_sweep/.

Two things this script keeps apart, same discipline as mix_sweep.py:

MEASURED    dispatch share per tier, read from the log — what the selector
            actually did, immune to any predictor bug.
PREDICTED   `pref` = sqrt(1+tau_hat_xavier)/sqrt(1+tau_hat_thor), computed from
            gpu_tier_sleep directly against each cell's own rendered config —
            the mechanism's strength, checkable independently of the run.

Belief-vs-reality ablation (BRIEF § 3): at B=10, `gpu` ran twice — the ordinary
cell (belief has C) and `..._b010nc` (belief does not).  Reported separately,
never averaged together.

Usage:
    python3 comm_sweep.py                     # tables
    python3 comm_sweep.py --plots             # tables + figures/comm/
    python3 comm_sweep.py --targets 45 50 55
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import tomllib
from dataclasses import dataclass
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

COMM_RUNS_DIR = _MODULE_DIR.parent / "runs" / "comm_sweep"
CONTROL_RUNS_DIR = _MODULE_DIR.parent / "runs" / "mix_sweep"  # B=inf, reused
CONFIG_DIR = _MODULE_DIR / "configs_comm"
LOG_DIR = _MODULE_DIR / "logs"
INPUT_DIR = _MODULE_DIR / "input"
OUT_DIR = _MODULE_DIR / "figures" / "comm"

TIER_ORDER = ("xavier", "orin", "thor")
GAP = "30"
ANCHOR = 20
MIX = "m50_35_15"
PAYLOAD_MBIT = 341.52

# label -> (B in Mbps, run-name suffix, config path, run dir).  "inf" is the
# reused mix-sweep control; the rest live under runs/comm_sweep/.
BANDWIDTHS: dict[str, float | None] = {
    "inf": None, "100": 100.0, "50": 50.0, "20": 20.0, "10": 10.0,
}
ABLATION_LABEL = "10nc"  # gpu only, B=10, comm_in_predictor=false


def run_name(arm: str, label: str) -> str:
    if label == "inf":
        return f"{arm}_xa{ANCHOR}_gap{GAP}_{MIX}"
    if label == ABLATION_LABEL:
        return f"{arm}_xa{ANCHOR}_gap{GAP}_{MIX}_b010nc"
    return f"{arm}_xa{ANCHOR}_gap{GAP}_{MIX}_b{label.zfill(3)}"


def run_dir_for(name: str, label: str) -> Path:
    return (CONTROL_RUNS_DIR if label == "inf" else COMM_RUNS_DIR) / name


@dataclass
class Cell:
    arm: str
    label: str  # bandwidth label: "inf", "100", "50", "20", "10", "10nc"
    run_dir: Path
    curve: pd.DataFrame
    done: bool
    log: Path

    @property
    def rounds(self) -> int:
        return int(self.curve["round"].iloc[-1])


def comm_latency(mbps: float | None) -> float:
    return 0.0 if mbps is None else PAYLOAD_MBIT / mbps


def predicted_pref(arm: str, mbps: float | None, comm_in_predictor: bool = True) -> float:
    """pref = sqrt(1+tau_hat_xavier)/sqrt(1+tau_hat_thor), via gpu_tier_sleep directly.

    Loads the cell's OWN rendered config (or synthesises the B=inf/ablation
    equivalents make_comm_sweep.py would have written) through PLATO's real
    Config() and gpu_tier_sleep functions, rather than re-deriving the queue
    formula by hand — so this number and the run it describes can never drift
    out of sync with each other.  ``arm`` selects the BELIEF: "gpu" reads the
    true tier file (tiers_geo_gap30.csv), "cpu" reads tiers_cpu.csv — same
    template, same [gpu_tier] comm keys, only the predictor's view differs.
    """
    import make_configs

    sys.path.insert(0, str(_MODULE_DIR.parents[3]))
    from plato.config import Config
    import gpu_tier_sleep as gts

    _, text, _ = make_configs.render(
        arm, GAP, ANCHOR, root=make_configs.DEFAULT_ROOT,
        mix=MIX, t_round_model="queue", target_accuracy=1.01, rounds=5000,
        comm_bandwidth_mbps=mbps, payload_mbit=PAYLOAD_MBIT,
        comm_in_predictor=comm_in_predictor,
    )
    tmp = _MODULE_DIR / ".comm_sweep_pref_probe.toml"
    tmp.write_text(text)
    Config._instance = None
    gts._computer = None
    gts._predictor_scale = None
    gts._comm_latency = None
    gts._comm_in_predictor = None
    gts._t_round_logged = False
    sys.argv = ["comm_sweep.py", "-c", str(tmp)]
    Config()

    client_ids = list(range(1, 101))
    sizes = {c: 500 for c in client_ids}
    t_round = gts.compute_pool_T_round_seconds(client_ids, sizes, 1, clients_per_round=10)
    tau_xavier = gts.seconds_to_rounds(gts.gpu_tops_predict_seconds(1, 500, 1), t_round)
    tau_thor = gts.seconds_to_rounds(gts.gpu_tops_predict_seconds(86, 500, 1), t_round)
    tmp.unlink(missing_ok=True)
    import math
    return math.sqrt(1 + tau_xavier) / math.sqrt(1 + tau_thor)


def collect(arms: list[str], smooth: int) -> dict[tuple[str, str], Cell]:
    cells: dict[tuple[str, str], Cell] = {}
    labels = list(BANDWIDTHS) + [ABLATION_LABEL]
    for label in labels:
        for arm in arms:
            if label == ABLATION_LABEL and arm != "gpu":
                continue
            name = run_name(arm, label)
            run_dir = run_dir_for(name, label)
            results_dir = run_dir / "results"
            if not results_dir.is_dir():
                continue
            results = sorted(results_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            results = [p for p in results if not p.stem.endswith(("_binned", "_accuracy_bins"))]
            if not results:
                continue
            curve = load_curve(results[0], stop_at=None, smooth=smooth)
            if len(curve) < 20:
                continue
            cells[(arm, label)] = Cell(
                arm=arm, label=label, run_dir=run_dir, curve=curve,
                done=(LOG_DIR / f"{name}.done").exists(),
                log=LOG_DIR / f"{name}.log",
            )
    return cells


def tta_table(cells: dict, targets: list[float]) -> pd.DataFrame:
    rows = []
    for (arm, label), cell in sorted(cells.items()):
        for target in targets:
            hit = first_crossing(cell.curve, target)
            rows.append({
                "arm": arm, "label": label,
                "mbps": BANDWIDTHS.get(label if label != ABLATION_LABEL else "10"),
                "target": target,
                "rounds": None if hit is None else hit[0],
                "hours": None if hit is None else hit[1] / 3600.0,
                "best_acc": float(cell.curve["smooth_pct"].max()),
                "ran_rounds": cell.rounds,
                "done": cell.done,
            })
    return pd.DataFrame(rows)


def steering_table(cells: dict) -> pd.DataFrame:
    with (INPUT_DIR / f"clients_{MIX}.csv").open(newline="") as f:
        tier_of = {int(r["client_id"]): r["tier_id"] for r in csv.DictReader(f)}
    pool_total = len(tier_of)
    pool_share = {t: sum(1 for v in tier_of.values() if v == t) / pool_total for t in TIER_ORDER}

    rows = []
    for (arm, label), cell in sorted(cells.items()):
        if not cell.log.exists():
            continue
        disp, v_by_tier = measure_dispatches(cell.log, tier_of)
        total = sum(disp.values())
        if not total:
            continue
        mbps = BANDWIDTHS.get(label if label != ABLATION_LABEL else "10")
        comm_in_pred = label != ABLATION_LABEL
        pref = predicted_pref(arm, mbps, comm_in_pred) if arm in ("gpu", "cpu") else None
        for tier in TIER_ORDER:
            share = disp[tier] / total
            rows.append({
                "arm": arm, "label": label, "mbps": mbps,
                "comm_in_predictor": comm_in_pred,
                "tier": tier,
                "dispatch_share": share,
                "pool_share": pool_share[tier],
                "lift": share / pool_share[tier] if pool_share[tier] else None,
                "mean_V": float(np.mean(v_by_tier[tier])) if v_by_tier[tier] else None,
                "pref_xavier_over_thor": pref,
            })
    return pd.DataFrame(rows)


def _label_order(labels) -> list[str]:
    def key(label: str):
        if label == "inf":
            return (0, 0)
        if label == ABLATION_LABEL:
            return (1, -float(ABLATION_LABEL.rstrip("nc")))
        return (1, -float(label))
    return sorted(set(labels), key=key)


def print_cells(cells: dict) -> None:
    print("--- comm-latency sweep: G=30, fleet 50/35/15, C=D/B added to both belief & reality ---")
    for label in _label_order(BANDWIDTHS.keys() | {ABLATION_LABEL}):
        mbps = BANDWIDTHS.get(label if label != ABLATION_LABEL else "10")
        c = comm_latency(mbps)
        tag = " (ablation: comm_in_predictor=false)" if label == ABLATION_LABEL else ""
        arms_here = sorted({a for (a, l) in cells if l == label})
        print(f"  B={label:>4} Mbps  C={c:7.4f}s  arms present: {arms_here or 'none yet'}{tag}")
    print()


def print_tta(tta: pd.DataFrame, targets: list[float], arms: list[str]) -> None:
    for target in targets:
        sel = tta[tta.target == target]
        if sel.empty:
            continue
        print(f"--- time to {target:.0f}% accuracy (simulated hours / round) ---")
        header = f"  {'B (Mbps)':>10} " + "".join(f"{a:>20}" for a in arms)
        print(header)
        for label in _label_order(sel.label):
            by_arm = {r.arm: r for r in sel[sel.label == label].itertuples()}
            line = f"  {label:>10} "
            for arm in arms:
                r = by_arm.get(arm)
                if r is None or r.hours is None or pd.isna(r.hours):
                    line += f"{'never crossed' if r is not None else '-':>20}"
                else:
                    line += f"{f'{r.hours:.2f} h / r{int(r.rounds)}':>20}"
            print(line)
        print()


def print_steering(steer: pd.DataFrame, arms: list[str]) -> None:
    if steer.empty:
        print("--- dispatch lift: no logs parsed yet ---\n")
        return
    print("--- dispatch share % and pref (xavier-over-thor speed-term ratio) ---")
    for arm in arms:
        sub = steer[steer.arm == arm]
        if sub.empty:
            continue
        print(f"\n  [{arm}]")
        print(f"    {'B (Mbps)':>10} " + "".join(f"{t:>10}" for t in TIER_ORDER)
              + f"{'pref':>8}")
        for label in _label_order(sub.label):
            rows = {r.tier: r for r in sub[sub.label == label].itertuples()}
            pref = next(iter(rows.values())).pref_xavier_over_thor
            line = f"    {label:>10} "
            line += "".join(f"{rows[t].dispatch_share * 100:>9.1f}%" if t in rows else f"{'-':>10}"
                             for t in TIER_ORDER)
            line += f"{pref:>8.2f}" if pref is not None else f"{'-':>8}"
            print(line)
    print(
        "\n  pref -> 1.0 as B falls means the speed signal is being diluted, not\n"
        "  degraded (BRIEF § 1): the additive comm floor is common to every tier.\n"
    )


def render(cells: dict, tta: pd.DataFrame, steer: pd.DataFrame, targets: list[float],
           out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot_time_to_accuracy import INK, INK_2, INK_MUTED, SURFACE, _style_axes

    order = _label_order(BANDWIDTHS.keys())  # inf, 100, 50, 20, 10 — no ablation on this axis
    xs = list(range(len(order)))
    x_labels = [("∞" if lb == "inf" else lb) for lb in order]
    target = targets[-1]

    # 1. Headline: dispatch share vs B.
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURFACE)
    tier_color = {"xavier": "#9ec6f0", "orin": "#4f93dd", "thor": "#14508f"}
    gpu = steer[steer.arm == "gpu"]
    for tier in TIER_ORDER:
        pts = []
        for x, label in zip(xs, order):
            row = gpu[(gpu.label == label) & (gpu.tier == tier)]
            if not row.empty:
                pts.append((x, float(row.dispatch_share.iloc[0]) * 100))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=tier_color[tier],
                    linewidth=2.0, marker="o", markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=tier, zorder=3)
    _style_axes(ax, "uplink bandwidth B (Mbps)", "dispatch share %")
    ax.set_xticks(xs, x_labels)
    ax.set_title("gpu arm: dispatch share vs bandwidth", color=INK, fontsize=11,
                 fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out_dir / "dispatch_share_vs_bandwidth.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 2. pref vs B, gpu and cpu together.
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURFACE)
    for arm in ("gpu", "cpu"):
        sub = steer[steer.arm == arm]
        pts = []
        for x, label in zip(xs, order):
            row = sub[sub.label == label]
            if not row.empty and pd.notna(row.pref_xavier_over_thor.iloc[0]):
                pts.append((x, float(row.pref_xavier_over_thor.iloc[0])))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=ARM_COLOR[arm],
                    linewidth=2.0, marker="o", markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=arm, zorder=3)
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 4)), zorder=1)
    _style_axes(ax, "uplink bandwidth B (Mbps)", "pref (xavier-over-thor speed term)")
    ax.set_xticks(xs, x_labels)
    ax.set_title("predicted mechanism strength vs bandwidth", color=INK, fontsize=11,
                 fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out_dir / "pref_vs_bandwidth.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 3. Hours and rounds to target vs B, one panel each.
    fig, (ax_h, ax_r) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE)
    for arm in ARM_ORDER:
        pts_h, pts_r = [], []
        for x, label in zip(xs, order):
            row = tta[(tta.arm == arm) & (tta.label == label) & (tta.target == target)]
            if row.empty or pd.isna(row.hours.iloc[0]):
                continue
            pts_h.append((x, float(row.hours.iloc[0])))
            pts_r.append((x, float(row.rounds.iloc[0])))
        if pts_h:
            ax_h.plot([p[0] for p in pts_h], [p[1] for p in pts_h], color=ARM_COLOR[arm],
                      linewidth=2.0, marker="o", markersize=8, markeredgecolor=SURFACE,
                      markeredgewidth=1.5, label=arm, zorder=3)
        if pts_r:
            ax_r.plot([p[0] for p in pts_r], [p[1] for p in pts_r], color=ARM_COLOR[arm],
                      linewidth=2.0, marker="o", markersize=8, markeredgecolor=SURFACE,
                      markeredgewidth=1.5, label=arm, zorder=3)
    _style_axes(ax_h, "uplink bandwidth B (Mbps)", f"simulated hours to {target:.0f}%")
    ax_h.set_xticks(xs, x_labels)
    ax_h.set_title("time (system + statistical)", color=INK, fontsize=11, fontweight="bold", loc="left")
    ax_h.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    _style_axes(ax_r, "uplink bandwidth B (Mbps)", f"rounds to {target:.0f}%")
    ax_r.set_xticks(xs, x_labels)
    ax_r.set_title("rounds (statistical only)", color=INK, fontsize=11, fontweight="bold", loc="left")
    ax_r.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out_dir / "tta_vs_bandwidth.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 4. The A-vs-B ablation at B=10: two curves, dispatch share bar comparison.
    a_cell, b_cell = cells.get(("gpu", "10")), cells.get(("gpu", ABLATION_LABEL))
    if a_cell is not None and b_cell is not None:
        fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURFACE)
        for cell, label, color in ((a_cell, "A: belief has C", "#2a78d6"),
                                    (b_cell, "B: belief lacks C", "#c0392b")):
            ax.plot(cell.curve["hours"], cell.curve["accuracy_pct"], color=color,
                    linewidth=0.7, alpha=0.22, zorder=2)
            ax.plot(cell.curve["hours"], cell.curve["smooth_pct"], color=color,
                    linewidth=2.0, label=label, zorder=3)
        _style_axes(ax, "simulated hours", "accuracy %")
        ax.set_title("belief-vs-reality ablation at B=10 Mbps", color=INK, fontsize=11,
                     fontweight="bold", loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_b010.png", dpi=160, facecolor=SURFACE)
        plt.close(fig)

    print(f"wrote figures to {out_dir}/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--arms", nargs="+", default=ARM_ORDER, choices=ARM_ORDER)
    p.add_argument("--targets", nargs="+", type=float, default=[45.0, 50.0, 55.0])
    p.add_argument("--smooth", type=int, default=25, help="evaluations in the rolling median")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--plots", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cells = collect(args.arms, args.smooth)
    if not cells:
        print("no runs found yet under runs/comm_sweep/ or runs/mix_sweep/", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print_cells(cells)

    running = sorted(f"{a}/{l}" for (a, l), c in cells.items() if not c.done)
    print(f"{len(cells)} run(s) found"
          + (f"; STILL RUNNING: {', '.join(running)}" if running else "; all finished"))
    print("  (unfinished runs are listed so a mid-sweep table is never read as the result)\n")

    tta = tta_table(cells, args.targets)
    tta.to_csv(args.out_dir / "tta_comm.csv", index=False)
    print_tta(tta, args.targets, args.arms)

    steer = steering_table(cells)
    if not steer.empty:
        steer.to_csv(args.out_dir / "steering_comm.csv", index=False)
    print_steering(steer, args.arms)

    print(f"wrote {args.out_dir}/tta_comm.csv"
          + (", steering_comm.csv" if not steer.empty else ""))

    if args.plots:
        render(cells, tta, steer, args.targets, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
