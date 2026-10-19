"""
Accuracy at a fixed simulated-time budget — the crossing-free comparison.

`time to reach accuracy X` reads a single noisy evaluation and divides by a slope
that collapses as the curve flattens, so it carries ±10 % of pure evaluation
noise at 55 % and becomes meaningless at 60 % (HANDOFF § 4.13).  Inverting the
axes removes the problem entirely:

    instead of  "how long to reach accuracy X"   (one evaluation decides it)
    ask         "what accuracy at time T"        (a window of evaluations does)

Two things make this robust where the crossing is not:

1. **Smoothing is available.**  A crossing cannot be smoothed — it is defined by
   a first exceedance, so any smoothing changes what is being measured.  An
   accuracy read at a fixed time can be a rolling median over `--window`
   evaluations, cutting the eval noise by ~sqrt(window).
2. **No division by a vanishing slope.**  On the plateau the metric simply
   reports that both arms plateaued, which is the truth, rather than amplifying
   noise without bound.

What it does NOT fix: trajectory variance — the two arms draw different clients
in a different order, and that is a real difference in the model's path, not
measurement error.  The G=1 control bounds it: the arms have no speed signal to
exploit there, so whatever gap remains at G=1 is the floor for the others.

Usage:
    python3 fixed_budget.py                 # table + calibration
    python3 fixed_budget.py --json out.json # data for the chart
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from plot_time_to_accuracy import _evaluated_rows

RUNS = "/workspace/plato/examples/async/fedasync/runs/gap_sweep"
GAPS = ["01", "10", "20", "30", "40"]
EFFECTIVE = {"01": 1.0, "10": 5.5, "20": 7.33, "30": 8.25, "40": 8.8}


def load(arm: str, gap: str, window: int):
    files = sorted(glob.glob(f"{RUNS}/{arm}_xa20_gap{gap}/results/*.csv"))
    if not files:
        return None
    df = pd.read_csv(files[-1]).sort_values("round").reset_index(drop=True)
    if "accuracy" not in df or len(df) < 200:
        return None
    # PLATO carries the last accuracy forward on non-evaluated rounds, so a
    # window measured in ROWS spans a tenth of the evaluations it claims to.
    # Subsample to real evaluations first (same helper the plot script uses).
    df = df[df.accuracy > 0]
    df = _evaluated_rows(df.reset_index(drop=True))
    if len(df) < 50:
        return None
    # Rolling median over `window` evaluations: kills the eval noise, keeps the
    # trend.  centre=True so the value is not shifted in time.
    df["acc_s"] = df.accuracy.rolling(window, center=True, min_periods=3).median()
    df["hours"] = df.elapsed_time / 3600.0
    return df


def acc_at(df: pd.DataFrame, hours: float):
    """Smoothed accuracy at a simulated-time budget; None past the run's end."""
    if hours > df.hours.iloc[-1]:
        return None
    return float(np.interp(hours, df.hours.to_numpy(), df.acc_s.to_numpy()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--window", type=int, default=15,
                    help="evaluations in the rolling median (default 15 = 150 rounds)")
    ap.add_argument("--json", help="write the chart's data here")
    args = ap.parse_args()

    runs = {}
    for gap in GAPS:
        for arm in ("gpu", "cpu", "baseline"):
            d = load(arm, gap, args.window)
            if d is not None:
                runs[(arm, gap)] = d

    print(f"Accuracy at a fixed simulated-time budget "
          f"(rolling median over {args.window} evals)\n")

    out = {"gaps": [], "window": args.window}
    for gap in GAPS:
        g, c = runs.get(("gpu", gap)), runs.get(("cpu", gap))
        if g is None:
            continue
        # Budgets up to where BOTH arms still have data, so every comparison is
        # like-for-like rather than extrapolated.
        end = g.hours.iloc[-1] if c is None else min(g.hours.iloc[-1], c.hours.iloc[-1])
        budgets = [round(end * f, 4) for f in (0.25, 0.5, 0.75, 1.0)]
        print(f"--- G={gap}  (effective {EFFECTIVE[gap]}x, both arms cover "
              f"{end:.3f} h) ---")
        print(f"  {'budget h':>9} {'gpu':>8} {'cpu':>8} {'gpu - cpu':>10}")
        rows = []
        for b in budgets:
            ga, ca = acc_at(g, b), (acc_at(c, b) if c is not None else None)
            d = None if (ga is None or ca is None) else ga - ca
            print(f"  {b:9.3f} {('—' if ga is None else f'{ga:.4f}'):>8}"
                  f" {('—' if ca is None else f'{ca:.4f}'):>8}"
                  f" {('—' if d is None else f'{d:+.4f}'):>10}")
            rows.append({"hours": b, "gpu": ga, "cpu": ca, "delta": d})
        out["gaps"].append({"gap": gap, "effective": EFFECTIVE[gap],
                            "end_hours": round(end, 4), "budgets": rows,
                            "curves": {
                                arm: [[round(h, 5), round(a, 5)]
                                      for h, a in zip(df.hours, df.acc_s)
                                      if not np.isnan(a)][::6]
                                for arm, df in (("gpu", g), ("cpu", c))
                                if df is not None}})
        print()

    # Calibration: G=1 has no speed signal, so its delta is the floor.
    g1 = next((x for x in out["gaps"] if x["gap"] == "01"), None)
    if g1:
        deltas = [abs(r["delta"]) for r in g1["budgets"] if r["delta"] is not None]
        if deltas:
            floor = max(deltas)
            print(f"Calibration — at G=1 neither arm can exploit a speed difference,")
            print(f"so its spread is the floor: |gpu - cpu| reaches {floor:.4f} accuracy.")
            print(f"A difference at another gap counts only if it clears that.\n")
            out["floor"] = round(floor, 5)
            for x in out["gaps"]:
                if x["gap"] == "01":
                    continue
                d = [r["delta"] for r in x["budgets"] if r["delta"] is not None]
                if not d:
                    continue
                mx = max(d, key=abs)
                verdict = ("gpu ahead" if mx > floor else
                           "cpu ahead" if mx < -floor else "inside the floor")
                print(f"  G={x['gap']}: largest gpu-cpu = {mx:+.4f}  ->  {verdict}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
