"""
Prompt C, Step 4: the pilot gate on S1 random and S1 gpu (seed 4126, 4,000 rounds).

| Gate    | Pass condition                                                         |
|---------|------------------------------------------------------------------------|
| Plateau | accuracy flat over the last 500 rounds: |slope| of a linear fit       |
|         | < 0.5 points (percentage points) per 1,000 rounds                     |
| Target  | the run crosses 80% before round 3,500                                 |
| Trace   | last dispatch time < 30,000 s (and no TraceEndError in the log)        |
| Clock   | online clock == replayed clock (max |diff| < 1e-6 s)                   |

Accuracy is evaluated every eval_interval = 10 rounds; only evaluated rounds
enter the fit and the crossing. Nothing here changes a setting -- a failing
gate is reported, not fixed.

Usage:
    python wen_env/wen_pilot_gate.py runs/wen_env/S1_random_s4126 runs/wen_env/S1_gpu_s4126 \
        --out wen_env/evidence/pilot_gate.md --plot wen_env/evidence/pilot_curves.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import wen_clock_replay

TARGET = 0.80
TARGET_BEFORE_ROUND = 3500
PLATEAU_WINDOW = 500
PLATEAU_MAX_SLOPE = 0.5          # percentage points per 1,000 rounds
TRACE_END_S = 30_000.0
EVAL_INTERVAL = 10


def curve(run: Path) -> list[tuple[int, float, float]]:
    f = next(p for p in (run / "results").glob("*.csv"))
    with f.open(newline="") as fh:
        return [(int(r["round"]), float(r["elapsed_time"]), float(r["accuracy"]))
                for r in csv.DictReader(fh) if int(r["round"]) % EVAL_INTERVAL == 0]


def gates(run: Path) -> dict:
    pts = curve(run)
    last = pts[-1][0]
    window = [(r, a) for r, _, a in pts if r > last - PLATEAU_WINDOW]
    slope = np.polyfit([r for r, _ in window], [100 * a for _, a in window], 1)[0] * 1000
    crossing = next((r for r, _, a in pts if a >= TARGET), None)
    with (run / "dispatch_log.csv").open(newline="") as fh:
        last_dispatch = max(float(r["t_dispatch"]) for r in csv.DictReader(fh))
    trace_err = any("TraceEndError" in p.read_text(errors="replace")
                    for p in run.glob("server_*.log"))
    clock = wen_clock_replay.replay(run)["max_abs_diff_s"]
    return {
        "run": run.name, "rounds": last,
        "plateau_slope": slope, "plateau_ok": abs(slope) < PLATEAU_MAX_SLOPE,
        "plateau_mean": float(np.mean([a for _, a in window])),
        "crossing_round": crossing,
        "target_ok": crossing is not None and crossing < TARGET_BEFORE_ROUND,
        "last_dispatch_s": last_dispatch,
        "trace_ok": last_dispatch < TRACE_END_S and not trace_err,
        "clock_diff": clock, "clock_ok": clock < 1e-6,
        "curve": pts,
    }


def render(results: list[dict]) -> str:
    ok = lambda b: "PASS" if b else "**FAIL**"  # noqa: E731
    out = ["# Prompt C, Step 4: pilot gate (seed 4126, 4,000 rounds)", "",
           "| run | rounds | plateau slope (pts / 1,000 rounds) | plateau | "
           "80% crossing round | target | last dispatch (s) | trace | clock diff (s) | clock |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        out.append(
            f"| {r['run']} | {r['rounds']} | {r['plateau_slope']:+.3f} "
            f"(mean acc {r['plateau_mean']:.4f}) | {ok(r['plateau_ok'])} | "
            f"{r['crossing_round'] if r['crossing_round'] is not None else 'never'} | "
            f"{ok(r['target_ok'])} | {r['last_dispatch_s']:.1f} | {ok(r['trace_ok'])} | "
            f"{r['clock_diff']:.1e} | {ok(r['clock_ok'])} |")
    passed = all(r[k] for r in results for k in ("plateau_ok", "target_ok", "trace_ok",
                                                  "clock_ok"))
    out += ["", f"**Gate: {'PASS — Phase A may start' if passed else 'FAIL — stop and report'}**"]
    return "\n".join(out) + "\n"


def plot(results: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for r in results:
        axes[0].plot([p[1] for p in r["curve"]], [p[2] for p in r["curve"]], label=r["run"])
        axes[1].plot([p[0] for p in r["curve"]], [p[2] for p in r["curve"]], label=r["run"])
    for ax, xl in zip(axes, ("online clock (s)", "round")):
        ax.axhline(TARGET, color="grey", ls="--", lw=0.8)
        ax.set_xlabel(xl)
        ax.set_ylabel("test accuracy")
        ax.legend()
    axes[1].axvline(TARGET_BEFORE_ROUND, color="grey", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    results = [gates(r) for r in args.runs]
    text = render(results)
    print(text)
    if args.out:
        args.out.write_text(text)
    if args.plot:
        plot(results, args.plot)
    return 0 if "PASS —" in text else 1


if __name__ == "__main__":
    raise SystemExit(main())
