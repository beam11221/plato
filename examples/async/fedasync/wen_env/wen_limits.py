"""
Prompt E, Step 1: Wen's constraint limits T_max and E_max (decision W5).

Population: every vehicle (30) x every 1 s trace sample t = 1 .. 30,000 of
bundle_wen_closed_v1, taken at the sample positions themselves.

For each (i, t), with Wen's ESTIMATES only (|chi|^2 = 1):
    F1 = T_com + T_up,   T_com = b |S_i| rho / f_tier  (rho = 1,250)
    F2 = e rho b |S_i| f^2 + p T_up
Both come from `wen_estimators.CpuEstimator` -- the same object the `wen`
selector scores with -- so the limits and the selector cannot disagree about
F1 / F2. The predicted upload position is wrap(Loc_x + T_com * V_i) through
`Estimator.upload_hat` -> `wen_formulas.wrap_x`, the same function the gpu arm's
x_hat uses.

T_max = 90th percentile of F1, E_max = 90th percentile of F2 (numpy's default
linear interpolation), over all (i, t).

Prompt I (Study 2, S2-2b): `--family gpu` applies the same W5 rule to the gpu
family's F1 / F2 (GpuEstimator: T_hat_train(GPU) + T_hat_up, x_hat from the
GPU T_hat_train; E = P_tier * T_hat_train + p * T_hat_up) for the GPU plug-in.

Usage:
    python wen_env/wen_limits.py --json wen_env/evidence/wen_limits.json \
        --out wen_env/evidence/wen_limits.md
    python wen_env/wen_limits.py --family gpu --json wen_env/evidence/wen_limits_gpu.json \
        --out wen_env/evidence/wen_limits_gpu.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import wen_constants as K
from wen_estimators import CpuEstimator, GpuEstimator
from wen_fleet import Fleet
from wen_reported import ReportedVehicle
from wen_trace import TraceIndex, load_speeds, veh_id

HERE = Path(__file__).resolve().parent
FEDASYNC = HERE.parent
BUNDLE = FEDASYNC / "sumo_ver2" / "bundle_wen_closed_v1"
FLEET = FEDASYNC / "gpu_tiers_dataset" / "fleets" / "n30_x15-o10-t5_phys_p1667_j0p149_s4126"
TIERS = FEDASYNC / "gpu_tiers_dataset" / "tiers" / "tiers_wen.csv"
MODEL_BITS = 357_566_784
QUANTILE = 90
TIER_ORDER = ("xavier", "orin", "thor")


FAMILIES = {"cpu": CpuEstimator, "gpu": GpuEstimator}


def population(family: str = "cpu") -> dict[str, np.ndarray]:
    """F1, F2, tier and position at decision time for every (vehicle, sample)."""
    speeds = load_speeds(BUNDLE)
    trace = TraceIndex.from_bundle(BUNDLE, speeds)
    fleet = Fleet.load(FLEET, TIERS)
    est = FAMILIES[family](K.FLOPS_FWD_PER_IMAGE, MODEL_BITS)
    f1, f2, tier, xs = [], [], [], []
    for cid in fleet.client_ids:
        vid = veh_id(cid)
        spec = fleet.spec(cid)
        t0, x, _lap = trace.samples(vid)
        for k in range(len(x)):
            if t0 + k > 30_000:
                break
            rv = ReportedVehicle(cid, t0 + k, float(x[k]), speeds[vid], spec.tier_id,
                                 spec.tflops_fp16, spec.cpu_ghz, K.partition_size(cid))
            e = est.estimate(rv)
            f1.append(e.total)
            f2.append(e.E)
            tier.append(spec.tier_id)
            xs.append(float(x[k]))
    return {"F1": np.array(f1), "F2": np.array(f2), "tier": np.array(tier), "x": np.array(xs)}


def limits(pop) -> tuple[float, float]:
    return (float(np.percentile(pop["F1"], QUANTILE)),
            float(np.percentile(pop["F2"], QUANTILE)))


def table(pop, t_max, e_max, mask) -> list[str]:
    rows = ["| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | "
            "cut by both | total cut |", "|---|---|---|---|---|---|---|"]
    for tier in TIER_ORDER + ("fleet",):
        m = mask & ((pop["tier"] == tier) if tier != "fleet" else True)
        f1, f2 = pop["F1"][m], pop["F2"][m]
        t_cut, e_cut = f1 > t_max, f2 > e_max
        rows.append(
            f"| {tier} | {f1.min():.3f}–{f1.max():.3f} | {f2.min():.3f}–{f2.max():.3f} | "
            f"{np.mean(t_cut & ~e_cut):.1%} | {np.mean(e_cut & ~t_cut):.1%} | "
            f"{np.mean(t_cut & e_cut):.1%} | {np.mean(t_cut | e_cut):.1%} |")
    return rows


def qualitative_check(pop, t_max, e_max) -> list[str]:
    """Prompt E / I stop rule: E_max must not exclude every position of any tier
    (Prompt E named Thor), and each limit must cut at least one candidate."""
    problems = []
    for tier in TIER_ORDER:
        if np.all(pop["F2"][pop["tier"] == tier] > e_max):
            problems.append(f"E_max excludes every {tier.capitalize()} position")
    if not np.any(pop["F1"] > t_max):
        problems.append("T_max cuts no candidate")
    if not np.any(pop["F2"] > e_max):
        problems.append("E_max cuts no candidate")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--family", choices=sorted(FAMILIES), default="cpu")
    args = parser.parse_args()
    pop = population(args.family)
    t_max, e_max = limits(pop)
    problems = qualitative_check(pop, t_max, e_max)
    inside = (pop["x"] >= K.S2_COVERAGE_M[0]) & (pop["x"] <= K.S2_COVERAGE_M[1])
    text = "\n".join([
        f"# Wen constraint limits, {args.family} family F1 / F2 (W5 rule; "
        f"{'Prompt E Step 1' if args.family == 'cpu' else 'Prompt I Step 2.3'})", "",
        f"Population: {len(pop['F1']):,} (vehicle, 1 s trace sample) pairs, t = 1..30,000 s.",
        f"T_max = P{QUANTILE}(F1) = **{t_max!r} s**; E_max = P{QUANTILE}(F2) = **{e_max!r} J**.",
        f"Qualitative check: {'PASS' if not problems else 'FAIL — ' + '; '.join(problems)}.", "",
        "## All positions", "", *table(pop, t_max, e_max, np.ones_like(inside)), "",
        "## Positions inside [400, 1600] at decision time", "",
        *table(pop, t_max, e_max, inside), ""])
    print(text)
    args.json.write_text(json.dumps({"family": args.family, "t_max": t_max, "e_max": e_max,
                                     "quantile": QUANTILE,
                                     "population": int(len(pop["F1"])),
                                     "qualitative_problems": problems}, indent=2) + "\n")
    if args.out:
        args.out.write_text(text)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
