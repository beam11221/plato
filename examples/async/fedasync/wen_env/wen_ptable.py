"""
Prompt C, Step 6: per-seed directions for predictions v3, and held / not held.

v3 rule: a prediction holds if the direction is the same in all 3 seeds.
Magnitudes are not predicted. Each check below is the literal comparison in
docs/predictions_v3.md; the code never rewords a prediction.

Usage:
    python wen_env/wen_ptable.py --out wen_env/evidence/ptable_v1.md
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import wen_main_report as R

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs" / "wen_env"
SEEDS = (4126, 4127, 4128)


def run_dir(cell: str, seed: int) -> Path:
    if seed == 4126:
        p = RUNS / "_phaseA" / f"{cell}_s4126"
        if p.exists():
            return p
    return RUNS / f"{cell}_s{seed}"


def dispatch_rate(run: Path) -> float:
    rows = R._rows(run / "dispatch_log.csv")
    return len(rows) / max(float(r["t_finish_clean"]) for r in rows)


CELLS = ("S1_random", "S1_cpu", "S1_gpu", "S1_observed", "S1_gpu_oracle", "S1_gpu_eps0",
         "S1_gpu_prop", "S1_location_only", "S2_random", "S2_gpu", "S2_gpu_stage0",
         "S2_cpu_stage0")


def load() -> dict[str, dict[int, dict]]:
    data = {}
    for c in CELLS:
        data[c] = {}
        for s in SEEDS:
            d = run_dir(c, s)
            m = R.one_run(d)
            m["rate"] = dispatch_rate(d)
            data[c][s] = m
    return data


def gain(d, cell, s, base="S1_random", key="s_per_round"):
    return (d[base][s][key] - d[cell][s][key]) / d[base][s][key]


def checks(d):
    """(P, part, per-seed (bool, text)) for every directional statement."""
    out = []

    def add(p, part, fn):
        out.append((p, part, [fn(s) for s in SEEDS]))

    add("P1", "cpu's s/round gain over random < half of gpu's",
        lambda s: (gain(d, "S1_cpu", s) < 0.5 * gain(d, "S1_gpu", s),
                   f"cpu {gain(d,'S1_cpu',s):+.1%} vs gpu {gain(d,'S1_gpu',s):+.1%}"))
    add("P2", "gpu < cpu on s/round", lambda s: (
        d["S1_gpu"][s]["s_per_round"] < d["S1_cpu"][s]["s_per_round"],
        f"{d['S1_gpu'][s]['s_per_round']:.3f} vs {d['S1_cpu'][s]['s_per_round']:.3f}"))
    add("P2", "gpu < cpu on TTA@80%", lambda s: (
        d["S1_gpu"][s]["tta"] < d["S1_cpu"][s]["tta"],
        f"{d['S1_gpu'][s]['tta']:.0f} vs {d['S1_cpu'][s]['tta']:.0f} s"))
    add("P3", "gpu < random on s/round before 80%", lambda s: (
        d["S1_gpu"][s]["s_before"] < d["S1_random"][s]["s_before"],
        f"{d['S1_gpu'][s]['s_before']:.3f} vs {d['S1_random'][s]['s_before']:.3f}"))
    add("P3", "gpu < random on s/round, whole run", lambda s: (
        d["S1_gpu"][s]["s_per_round"] < d["S1_random"][s]["s_per_round"],
        f"{d['S1_gpu'][s]['s_per_round']:.3f} vs {d['S1_random'][s]['s_per_round']:.3f}"))
    add("P3", "gpu < random on TTA@80%", lambda s: (
        d["S1_gpu"][s]["tta"] < d["S1_random"][s]["tta"],
        f"{d['S1_gpu'][s]['tta']:.0f} vs {d['S1_random'][s]['tta']:.0f} s"))
    add("P4", "|plateau gpu − random| ≤ 1 point", lambda s: (
        abs(d["S1_gpu"][s]["plateau"] - d["S1_random"][s]["plateau"]) <= 0.01,
        f"{100*(d['S1_gpu'][s]['plateau']-d['S1_random'][s]['plateau']):+.2f} pts"))
    add("P5", "energy to 80%: gpu < random", lambda s: (
        d["S1_gpu"][s]["energy_to_80"] < d["S1_random"][s]["energy_to_80"],
        f"{d['S1_gpu'][s]['energy_to_80']/1e3:.1f} vs {d['S1_random'][s]['energy_to_80']/1e3:.1f} kJ"))
    for phase, key in (("before 80%", "s_before"), ("after 80%", "s_after")):
        add("P8", f"observed does not beat gpu: s/round {phase}",
            lambda s, key=key: (d["S1_observed"][s][key] >= d["S1_gpu"][s][key],
                                f"obs {d['S1_observed'][s][key]:.3f} vs gpu {d['S1_gpu'][s][key]:.3f}"))
    add("P8", "observed does not beat gpu: TTA@80%", lambda s: (
        d["S1_observed"][s]["tta"] >= d["S1_gpu"][s]["tta"],
        f"obs {d['S1_observed'][s]['tta']:.0f} vs gpu {d['S1_gpu'][s]['tta']:.0f} s"))
    add("P9", "Stage 0 reduces waste (gpu)", lambda s: (
        d["S2_gpu_stage0"][s]["waste_dispatch"] < d["S2_gpu"][s]["waste_dispatch"],
        f"{d['S2_gpu'][s]['waste_dispatch']:.1%} → {d['S2_gpu_stage0'][s]['waste_dispatch']:.2%}"))
    add("P10", "cpu + Stage 0 admits fewer dispatches / s than gpu + Stage 0", lambda s: (
        d["S2_cpu_stage0"][s]["rate"] < d["S2_gpu_stage0"][s]["rate"],
        f"{d['S2_cpu_stage0'][s]['rate']:.3f} vs {d['S2_gpu_stage0'][s]['rate']:.3f} /s"))
    add("P11", "Stage 0 does not lower the plateau (gpu)", lambda s: (
        d["S2_gpu_stage0"][s]["plateau"] >= d["S2_gpu"][s]["plateau"],
        f"{100*(d['S2_gpu_stage0'][s]['plateau']-d['S2_gpu'][s]['plateau']):+.2f} pts"))
    add("P12", "location-only gains over random (s/round)", lambda s: (
        d["S1_location_only"][s]["s_per_round"] < d["S1_random"][s]["s_per_round"],
        f"{gain(d,'S1_location_only',s):+.1%}"))
    add("P12", "gpu gains more than location-only (s/round)", lambda s: (
        d["S1_gpu"][s]["s_per_round"] < d["S1_location_only"][s]["s_per_round"],
        f"gpu {gain(d,'S1_gpu',s):+.1%} vs loc {gain(d,'S1_location_only',s):+.1%}"))
    add("P13", "ε = 0 s/round gain ≥ ε = 0.2", lambda s: (
        d["S1_gpu_eps0"][s]["s_per_round"] <= d["S1_gpu"][s]["s_per_round"],
        f"{d['S1_gpu_eps0'][s]['s_per_round']:.3f} vs {d['S1_gpu'][s]['s_per_round']:.3f}"))
    add("P13", "ε = 0 plateau ≤ ε = 0.2 plateau", lambda s: (
        d["S1_gpu_eps0"][s]["plateau"] <= d["S1_gpu"][s]["plateau"],
        f"{100*(d['S1_gpu_eps0'][s]['plateau']-d['S1_gpu'][s]['plateau']):+.2f} pts"))
    return out


def render(d) -> str:
    lines = ["# Predictions v3: held / not held (3 seeds; v3 rule = same direction in all 3)", "",
             "| P | statement checked | seed 4126 | seed 4127 | seed 4128 | verdict |",
             "|---|---|---|---|---|---|"]
    for p, part, res in checks(d):
        cells = [("✓ " if ok else "✗ ") + txt for ok, txt in res]
        verdict = "**held**" if all(ok for ok, _ in res) else "**not held**"
        lines.append(f"| {p} | {part} | " + " | ".join(cells) + f" | {verdict} |")
    # Non-directional / partial items, reported as numbers.
    gw = [d["S2_gpu_stage0"][s]["waste_dispatch"] for s in SEEDS]
    lines += ["", "Non-directional parts (numbers only):", "",
              f"- P7 gpu ≈ gpu_oracle: TTA@80% gpu {[round(d['S1_gpu'][s]['tta']) for s in SEEDS]} s, "
              f"oracle {[round(d['S1_gpu_oracle'][s]['tta']) for s in SEEDS]} s; s/round gpu "
              f"{[round(d['S1_gpu'][s]['s_per_round'],3) for s in SEEDS]}, oracle "
              f"{[round(d['S1_gpu_oracle'][s]['s_per_round'],3) for s in SEEDS]}.",
              f"- P10 'gpu + Stage 0 waste near zero': {', '.join(f'{w:.2%}' for w in gw)} of dispatches.",
              ]
    dp = [d["S1_gpu_prop"][s]["s_per_round"] - d["S1_random"][s]["s_per_round"] for s in SEEDS]
    spread = st.stdev([d["S1_random"][s]["s_per_round"] for s in SEEDS])
    lines.append(f"- P13 'proportional ≈ random on s/round (|Δ| within seed spread)': Δ per seed "
                 f"{', '.join(f'{x:+.3f}' for x in dp)} s; random's seed SD {spread:.3f} s; "
                 f"|mean Δ| = {abs(st.fmean(dp)):.3f} s.")
    lines.append("- P6, and P9 for Wen-faithful: not run (Wen selector is a later prompt).")
    return "\n".join(lines) + "\n"


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
