"""
Cross-experiment comparison: does mobility change which speed estimator wins?

`analyze.py` reports one cell against its own no-mobility control. This script
asks the question that spans all four runs at once:

    In a world where clients stand still, the GPU-spec predictor beats the
    reactive one. Is that still true when the clients move?

It compares MAE of tau_hat (in rounds) for the `gpu` and `observed` arms across
the 2x2 of {n20, n100} x {still, moving}, and reports the gap with a standard
error so "they are the same" and "one is better" can be told apart.

Reading the bias column
-----------------------
Bias is mean(actual - predicted). It is the whole story here:

  * `observed` measures tau directly, so its bias is ~0 in every condition.
  * `gpu` predicts tau_hat = T_i / T_round from a speed model that knows nothing
    about mobility. Under mobility a round is only produced when an upload
    succeeds, so rounds arrive (1 - waste) times slower than reports do while the
    prediction does not move -- the arm over-predicts, and the more waste there
    is, the deeper the bias.

So any advantage `observed` shows under mobility is at least partly an artifact
of the ROUND DEFINITION this experiment chose, not evidence that a reactive
estimator is more accurate. The debiased column exists to separate the two: it
subtracts each arm's own mean bias before taking the absolute error, i.e. it asks
"how well would this predictor do if someone corrected its constant offset?".

Usage:
    python3 compare_experiments.py
"""

from __future__ import annotations

import csv
import glob
import math
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEDASYNC = HERE.parent

PREDICTION_RE = re.compile(
    r"cid=(\d+)\s+predicted=([\d.]+) rounds\s+actual=([\d.]+) rounds"
)
WASTE_RE = re.compile(r"waste round=\d+ cycles=\d+ total=\d+/\d+ \(([\d.]+)%\)")

# The no-mobility control runs, per cell.
CONTROL = {
    "n20": "../runs/surgical_sweep/{arm}_xa100_gap15/server_*.log",
    "n100": "../runs/surgical_n100/{arm}_xa100_gap15_n100/server_*.log",
}


def errors(path: str) -> list[float]:
    """Signed prediction errors (actual - predicted), in rounds."""
    text = Path(path).read_text(errors="replace")
    return [float(a) - float(p) for _, p, a in PREDICTION_RE.findall(text)]


def summarise(signed: list[float]) -> dict:
    """MAE, its standard error, the bias, and the MAE after removing the bias."""
    bias = st.mean(signed)
    absolute = [abs(x) for x in signed]
    debiased = [abs(x - bias) for x in signed]
    return {
        "mae": st.mean(absolute),
        "se": st.stdev(absolute) / math.sqrt(len(absolute)),
        "bias": bias,
        "debiased": st.mean(debiased),
        "n": len(absolute),
    }


def waste_pct(cell: str, arm: str) -> float | None:
    log = HERE / "logs" / f"{arm}_{cell}.log"
    if not log.exists():
        return None
    hits = WASTE_RE.findall(log.read_text(errors="replace"))
    return float(hits[-1]) if hits else None


def load(cell: str, arm: str, moving: bool) -> dict | None:
    if moving:
        path = HERE / "logs" / f"{arm}_{cell}.log"
        if not path.exists():
            return None
        return summarise(errors(str(path)))
    found = glob.glob(str(HERE / CONTROL[cell].format(arm=arm)))
    return summarise(errors(found[0])) if found else None


def main() -> None:
    print("MAE of tau-hat, in rounds. Lower is better; bias is mean(actual - predicted).\n")
    header = (
        f"{'cell':<6}{'condition':<9} {'arm':<10}"
        f"{'MAE':>8}{'SE':>7}{'bias':>8}{'MAE debiased':>14}{'n':>8}"
    )
    print(header)
    print("-" * len(header))

    for cell in ("n20", "n100"):
        for moving in (False, True):
            for arm in ("gpu", "observed"):
                s = load(cell, arm, moving)
                if s is None:
                    continue
                print(
                    f"{cell:<6}{'moving' if moving else 'still':<9} {arm:<10}"
                    f"{s['mae']:>8.3f}{s['se']:>7.3f}{s['bias']:>+8.2f}"
                    f"{s['debiased']:>14.3f}{s['n']:>8}"
                )
        print()

    print("\nIs the gap between the two arms real? (gap = observed - gpu)\n")
    print(f"{'cell':<6}{'condition':<9}{'raw gap':>18}   {'verdict':<24}{'debiased gap':>14}")
    print("-" * 73)
    for cell in ("n20", "n100"):
        for moving in (False, True):
            g, o = load(cell, "gpu", moving), load(cell, "observed", moving)
            if not g or not o:
                continue
            gap = o["mae"] - g["mae"]
            se = math.sqrt(g["se"] ** 2 + o["se"] ** 2)
            sigma = abs(gap) / se
            winner = "observed better" if gap < 0 else "gpu better"
            verdict = f"{sigma:.1f} SE, {winner}" if sigma > 2 else f"{sigma:.1f} SE, tie"
            print(
                f"{cell:<6}{'moving' if moving else 'still':<9}"
                f"{gap:>+13.3f} ± {se:.3f}   {verdict:<24}"
                f"{o['debiased'] - g['debiased']:>+14.3f}"
            )

    print("\nWaste, which drives the gpu arm's bias under mobility:\n")
    for cell in ("n20", "n100"):
        parts = [
            f"{arm} {waste_pct(cell, arm):.1f}%"
            for arm in ("gpu", "observed")
            if waste_pct(cell, arm) is not None
        ]
        print(f"  {cell}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
