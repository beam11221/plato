"""Re-measure the eval noise without letting curvature inflate it.

The first pass fitted a straight line over +/-400 rounds and called the residual
"noise".  Where the curve is genuinely bending (the steep early part), that
residual contains curvature too, so sigma was overstated exactly where the slope
is largest.

Two curvature-free estimators, both local:

  MAD-detrend : residual around a ROLLING MEDIAN (window 15 evals = 150 rounds),
                which tracks a bending trend instead of forcing a line through it.
  lag-1       : sigma = std(diff(acc)) / sqrt(2).  Consecutive evals sit 10 rounds
                apart, so the trend contributes almost nothing to the difference
                and this is nearly pure measurement noise.

If both agree, the noise is real and not an artefact of the fit.
"""
import glob

import numpy as np
import pandas as pd

from plot_time_to_accuracy import _evaluated_rows

RUNS = "/workspace/plato/examples/async/fedasync/runs/gap_sweep"
TARGETS = [0.45, 0.50, 0.55, 0.57, 0.58]
WIN = 400  # in ROUNDS (the window filter compares round numbers) = 40 evaluations


def load(run):
    f = sorted(glob.glob(f"{RUNS}/{run}/results/*.csv"))
    if not f:
        return None
    df = pd.read_csv(f[-1]).sort_values("round").reset_index(drop=True)
    if "accuracy" not in df or len(df) < 500:
        return None
    # Drop the carried-forward repeats: accuracy is written every round but
    # evaluated every `eval_interval`, and 90 % of consecutive differences are
    # exactly 0.  Left in, they deflate every noise estimate by ~3x.
    df = df[df.accuracy > 0]
    return _evaluated_rows(df.reset_index(drop=True))


rows = []
for d in sorted(glob.glob(f"{RUNS}/*")):
    run = d.split("/")[-1]
    df = load(run)
    if df is None:
        continue
    spr = (df.elapsed_time.iloc[-1] - df.elapsed_time.iloc[0]) / (
        df["round"].iloc[-1] - df["round"].iloc[0])
    for t in TARGETS:
        hit = df[df.accuracy >= t]
        if hit.empty:
            continue
        r0 = hit["round"].iloc[0]
        w = df[(df["round"] >= r0 - WIN) & (df["round"] <= r0 + WIN)]
        if len(w) < 12:
            continue
        y = w.accuracy.to_numpy(float)
        x = w["round"].to_numpy(float)

        # trend: rolling median tracks curvature; slope from its endpoints
        med = pd.Series(y).rolling(15, center=True, min_periods=5).median()
        sigma_mad = float((y - med).std(ddof=1))
        sigma_lag1 = float(np.diff(y).std(ddof=1) / np.sqrt(2))
        # local slope from the smoothed trend, robust to the noise
        good = med.notna().to_numpy()
        slope = np.polyfit(x[good], med[good].to_numpy(), 1)[0]

        sigma = max(sigma_mad, sigma_lag1)
        sig_rounds = sigma / abs(slope) if abs(slope) > 1e-12 else np.inf
        rows.append(dict(run=run, target=f"{t:.0%}", round=int(r0),
                         slope_1k=slope * 1000, mad=sigma_mad, lag1=sigma_lag1,
                         amb_rounds=sig_rounds,
                         amb_hours=sig_rounds * spr / 3600,
                         hours=float(hit.elapsed_time.iloc[0]) / 3600))

d = pd.DataFrame(rows)
print("Curvature-free noise, and the crossing ambiguity it implies")
print("mad / lag1 = two independent noise estimates (accuracy units)")
print("amb_hours  = +/- simulated hours the crossing can slide on noise alone\n")
for t in [f"{x:.0%}" for x in TARGETS]:
    sub = d[d.target == t]
    if sub.empty:
        continue
    print(f"--- target {t} ---")
    s = sub[["run", "round", "slope_1k", "mad", "lag1", "amb_rounds",
             "amb_hours", "hours"]]
    print(s.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    share = (sub.amb_hours / sub.hours)
    print(f"  ambiguity as a share of the measured time: "
          f"median {share.median():.0%}, worst {share.max():.0%}")
    # what separation between two arms would actually be detectable
    print(f"  => a gpu-vs-cpu ratio is only meaningful beyond about "
          f"{1 + (share.median() * np.sqrt(2)):.2f}x\n")
