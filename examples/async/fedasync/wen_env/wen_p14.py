"""
Prompt D, Step 5: P14a-d (predictions v3.1 addendum), per seed, held / not held.

Comparisons are the literal addendum statements:
  P14a  cpu_cal's s/round gain over random >= half of location-only's gain
  P14b  gpu s/round < cpu_cal s/round
  P14c  energy to 80%: gpu < cpu_cal
  P14d  cpu_cal s/round < cpu s/round
Gain = (random - arm) / random, s/round over the whole run, same seed.

Usage:
    python wen_env/wen_p14.py --out <file>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import wen_main_report as R
from wen_ptable import SEEDS, run_dir

CELLS = ("S1_random", "S1_cpu", "S1_cpu_cal", "S1_gpu", "S1_location_only")


def load():
    return {c: {s: R.one_run(run_dir(c, s)) for s in SEEDS} for c in CELLS}


def gain(d, cell, s):
    r = d["S1_random"][s]["s_per_round"]
    return (r - d[cell][s]["s_per_round"]) / r


def checks(d):
    rows = [
        ("P14a", "cpu_cal gain over random ≥ ½ × location-only gain",
         lambda s: (gain(d, "S1_cpu_cal", s) >= 0.5 * gain(d, "S1_location_only", s),
                    f"cpu_cal {gain(d,'S1_cpu_cal',s):+.1%} vs ½·loc {0.5*gain(d,'S1_location_only',s):+.1%}")),
        ("P14b", "gpu s/round < cpu_cal s/round",
         lambda s: (d["S1_gpu"][s]["s_per_round"] < d["S1_cpu_cal"][s]["s_per_round"],
                    f"{d['S1_gpu'][s]['s_per_round']:.3f} vs {d['S1_cpu_cal'][s]['s_per_round']:.3f}")),
        ("P14c", "energy to 80%: gpu < cpu_cal",
         lambda s: (d["S1_gpu"][s]["energy_to_80"] < d["S1_cpu_cal"][s]["energy_to_80"],
                    f"{d['S1_gpu'][s]['energy_to_80']/1e3:.1f} vs {d['S1_cpu_cal'][s]['energy_to_80']/1e3:.1f} kJ")),
        ("P14d", "cpu_cal s/round < cpu s/round",
         lambda s: (d["S1_cpu_cal"][s]["s_per_round"] < d["S1_cpu"][s]["s_per_round"],
                    f"{d['S1_cpu_cal'][s]['s_per_round']:.3f} vs {d['S1_cpu'][s]['s_per_round']:.3f}")),
    ]
    return [(p, txt, [fn(s) for s in SEEDS]) for p, txt, fn in rows]


def render(d) -> str:
    L = ["| P | statement checked | seed 4126 | seed 4127 | seed 4128 | verdict |",
         "|---|---|---|---|---|---|"]
    for p, txt, res in checks(d):
        cells = [("✓ " if ok else "✗ ") + t for ok, t in res]
        verdict = "**held**" if all(ok for ok, _ in res) else "**not held**"
        L.append(f"| {p} | {txt} | " + " | ".join(cells) + f" | {verdict} |")
    return "\n".join(L) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = render(load())
    print(text)
    if args.out:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
