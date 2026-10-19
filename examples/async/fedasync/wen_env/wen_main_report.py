"""
Prompt C, Step 6: mechanical analysis of the main runs (no interpretation).

Per cell (mean ± std over the seeds present; one seed -> the single value):
    TTA@80%            online-clock elapsed time of the first EVALUATED round with
                       accuracy >= 0.80 (accuracy is evaluated every
                       eval_interval rounds; other rows carry the last value)
    plateau            mean accuracy of the evaluated rounds among the last 200
    s/round            whole run, and before / after the 80% crossing
    energy to 80%      ground-truth J of every dispatch that finished at or
                       before TTA@80%
    dispatch share     per tier in rounds 1-500, 1,500-2,000, 3,500-4,000
    explored share     share of explored picks (egreedy cells)
    S2                 waste fraction (dispatches, GPU-s, J), Stage 0 rejection rate

Curves: accuracy vs online clock, one panel per scenario, mean with min-max band.

The P-table needs the v3 prediction text (docs/predictions_v3.md) and is filled
by hand from these numbers; this script does not word or judge predictions.

Usage:
    python wen_env/wen_main_report.py --runs runs/wen_env/S1_* runs/wen_env/S2_* \
        --out wen_env/evidence/main_runs_seed4126.md --plot wen_env/evidence/curves.png
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import wen_clock_replay

TARGET = 0.80
TIERS = ("xavier", "orin", "thor")
WINDOWS = ((1, 500), (1500, 2000), (3500, 4000))
SEED = re.compile(r"_s(\d{4})(?:_|$)")
STAGE0 = re.compile(r"\[WenStage0\] summary .*candidates=(\d+) rejected=(\d+)")


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def cell_of(run: Path) -> str:
    return SEED.sub("", run.name)


def accuracy_curve(run: Path) -> list[tuple[int, float, float]]:
    """(round, elapsed, accuracy) for rounds where accuracy was actually evaluated."""
    results = next((run / "results").glob("*.csv"))
    rows = _rows(results)
    eval_interval = 10
    out = []
    for r in rows:
        rnd = int(r["round"])
        if rnd % eval_interval == 0:
            out.append((rnd, float(r["elapsed_time"]), float(r["accuracy"])))
    return out


def one_run(run: Path) -> dict:
    curve = accuracy_curve(run)
    d = _rows(run / "dispatch_log.csv")
    last_round = curve[-1][0] if curve else 0
    total_t = curve[-1][1] if curve else 0.0
    crossing = next(((rnd, t) for rnd, t, a in curve if a >= TARGET), None)
    tail = [a for rnd, _, a in curve if rnd > last_round - 200]
    out = {
        "run": run.name, "rounds": last_round, "tta": crossing[1] if crossing else None,
        "tta_round": crossing[0] if crossing else None,
        "plateau": st.fmean(tail) if tail else None,
        "s_per_round": total_t / last_round if last_round else None,
    }
    if crossing:
        rnd, t = crossing
        out["s_before"] = t / rnd
        out["s_after"] = (total_t - t) / (last_round - rnd) if last_round > rnd else None
        out["energy_to_80"] = sum(float(r["E_train"]) + float(r["E_up"]) for r in d
                                  if float(r["t_finish_clean"]) <= t)
    else:
        out["s_before"] = out["s_after"] = out["energy_to_80"] = None
    for lo, hi in WINDOWS:
        sub = [r for r in d if lo <= int(r["round"]) <= hi]
        out[f"share_{lo}_{hi}"] = {t: (sum(r["tier"] == t for r in sub) / len(sub)
                                       if sub else None) for t in TIERS}
    explored = [r for r in d if r.get("explored", "") != ""]
    out["explored"] = (sum(r["explored"] == "1" for r in explored) / len(explored)
                       if explored else None)
    if d and d[0]["scenario"] == "S2":
        wasted = [r for r in d if r["accepted"] == "0"]
        out["waste_dispatch"] = len(wasted) / len(d)
        out["waste_gpu_s"] = sum(float(r["T_train_true"]) for r in wasted)
        out["waste_j"] = sum(float(r["E_train"]) + float(r["E_up"]) for r in wasted)
        out["stage0_rejected"] = None
        for log in run.glob("server_*.log"):
            for line in log.read_text(errors="replace").splitlines():
                m = STAGE0.search(line)
                if m:
                    out["stage0_rejected"] = int(m.group(2)) / max(int(m.group(1)), 1)
    out["clock_ok"] = wen_clock_replay.replay(run)["max_abs_diff_s"] < 1e-6
    out["curve"] = curve
    return out


def _agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.4g}"
    return f"{st.fmean(vals):.4g} ± {st.stdev(vals):.2g}"


def render(cells: dict[str, list[dict]]) -> str:
    out = ["# Main runs — mechanical numbers (no interpretation)", "",
           "| cell | seeds | TTA@80% (s) | crossing round | plateau | s/round | "
           "s/round before / after 80% | energy to 80% (J) | explored | clock ok |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for cell, runs in sorted(cells.items()):
        out.append(
            f"| {cell} | {len(runs)} | {_agg(r['tta'] for r in runs)} | "
            f"{_agg(r['tta_round'] for r in runs)} | {_agg(r['plateau'] for r in runs)} | "
            f"{_agg(r['s_per_round'] for r in runs)} | "
            f"{_agg(r['s_before'] for r in runs)} / {_agg(r['s_after'] for r in runs)} | "
            f"{_agg(r['energy_to_80'] for r in runs)} | {_agg(r['explored'] for r in runs)} | "
            f"{all(r['clock_ok'] for r in runs)} |")
    out += ["", "## Dispatch share per tier (xavier / orin / thor)", "",
            "| cell | rounds 1–500 | 1,500–2,000 | 3,500–4,000 |", "|---|---|---|---|"]
    for cell, runs in sorted(cells.items()):
        parts = []
        for lo, hi in WINDOWS:
            parts.append(" / ".join(_agg(r[f"share_{lo}_{hi}"][t] for r in runs)
                                    for t in TIERS))
        out.append(f"| {cell} | " + " | ".join(parts) + " |")
    s2 = {c: r for c, r in cells.items() if c.startswith("S2")}
    if s2:
        out += ["", "## S2 waste and Stage 0", "",
                "| cell | waste (dispatches) | waste GPU-s | waste J | Stage 0 rejected |",
                "|---|---|---|---|---|"]
        for cell, runs in sorted(s2.items()):
            out.append(f"| {cell} | {_agg(r['waste_dispatch'] for r in runs)} | "
                       f"{_agg(r['waste_gpu_s'] for r in runs)} | "
                       f"{_agg(r['waste_j'] for r in runs)} | "
                       f"{_agg(r['stage0_rejected'] for r in runs)} |")
    return "\n".join(out) + "\n"


def plot(cells: dict[str, list[dict]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    scenarios = sorted({c.split("_")[0] for c in cells})
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7 * len(scenarios), 4.5),
                             squeeze=False)
    for ax, sc in zip(axes[0], scenarios):
        for cell, runs in sorted(cells.items()):
            if not cell.startswith(sc):
                continue
            grid = np.linspace(0, min(r["curve"][-1][1] for r in runs), 400)
            ys = np.array([np.interp(grid, [p[1] for p in r["curve"]],
                                     [p[2] for p in r["curve"]]) for r in runs])
            ax.plot(grid, ys.mean(axis=0), label=cell.removeprefix(sc + "_"))
            if len(runs) > 1:
                ax.fill_between(grid, ys.min(axis=0), ys.max(axis=0), alpha=0.2)
        ax.axhline(TARGET, color="grey", lw=0.8, ls="--")
        ax.set_title(sc)
        ax.set_xlabel("online clock (s)")
        ax.set_ylabel("test accuracy")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    cells: dict[str, list[dict]] = defaultdict(list)
    for run in args.runs:
        cells[cell_of(run)].append(one_run(run))
    text = render(cells)
    print(text)
    if args.out:
        args.out.write_text(text)
    if args.plot:
        plot(cells, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
