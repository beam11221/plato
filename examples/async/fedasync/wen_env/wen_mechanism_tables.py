"""
Prompt D, Step 4: log-only mechanism tables (no new runs). Mechanical; no conclusions.

Per arm, mean ± SD over the seeds present (3 when complete), from dispatch logs:
  1. ground-truth energy per dispatch (J): all, and before / after the 80% crossing
     (a dispatch counts as "before" if it finished at or before TTA@80%)
  2. mean upload distance d at training finish (m), and the share of dispatches
     whose training finishes inside x in [400, 1600]
  3. mean true T_train and true T_up per dispatch (s)
  4. dispatch share per tier, rounds 1-500 and the last 500 rounds of each run
  5. position-estimate error: mean |x_hat_finish - x_finish| (m), where
     x_hat_finish = wrap(x_dispatch + V_k * T_hat_train) is the arm's own predicted
     finish position and x_finish the trace's; circular distance on the 2,000 m
     loop. Arms without a T_hat (random, observed) have none.

Usage:
    python wen_env/wen_mechanism_tables.py --out wen_env/evidence/mechanism_tables_v1.md
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import wen_constants as K
import wen_main_report as R
from wen_ptable import SEEDS, run_dir

S1_ARMS = ("S1_random", "S1_cpu", "S1_cpu_cal", "S1_gpu", "S1_observed", "S1_gpu_oracle",
           "S1_gpu_eps0", "S1_gpu_prop", "S1_location_only")
S2_ARMS = ("S2_random", "S2_gpu", "S2_gpu_stage0", "S2_cpu_stage0")
TIERS = ("xavier", "orin", "thor")


def _f(r, k):
    return float(r[k])


def circ(a: float, b: float) -> float:
    d = abs(a - b) % K.ROAD_LENGTH_M
    return min(d, K.ROAD_LENGTH_M - d)


def per_run(run: Path) -> dict:
    d = R._rows(run / "dispatch_log.csv")
    tta = R.one_run(run)["tta"]
    energy = [_f(r, "E_train") + _f(r, "E_up") for r in d]
    before = [_f(r, "E_train") + _f(r, "E_up") for r in d
              if tta is not None and _f(r, "t_finish_clean") <= tta]
    after = [_f(r, "E_train") + _f(r, "E_up") for r in d
             if tta is not None and _f(r, "t_finish_clean") > tta]
    lo, hi = K.S2_COVERAGE_M
    last_round = max(int(r["round"]) for r in d)
    first = [r for r in d if int(r["round"]) <= 500]
    last = [r for r in d if int(r["round"]) > last_round - 500]
    errs = []
    for r in d:
        if r["T_hat_train"] == "":
            continue
        x_hat = (_f(r, "x_dispatch") + _f(r, "V_k") * _f(r, "T_hat_train")) % K.ROAD_LENGTH_M
        errs.append(circ(x_hat, _f(r, "x_finish")))
    return {
        "E_all": st.fmean(energy),
        "E_before": st.fmean(before) if before else None,
        "E_after": st.fmean(after) if after else None,
        "d": st.fmean(_f(r, "d_finish") for r in d),
        "inside": sum(lo <= _f(r, "x_finish") <= hi for r in d) / len(d),
        "T_train": st.fmean(_f(r, "T_train_true") for r in d),
        "T_up": st.fmean(_f(r, "T_up_true") for r in d),
        **{f"first_{t}": sum(r["tier"] == t for r in first) / len(first) for t in TIERS},
        **{f"last_{t}": sum(r["tier"] == t for r in last) / len(last) for t in TIERS},
        "pos_err": st.fmean(errs) if errs else None,
    }


def agg(vals, fmt="{:.2f}"):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return fmt.format(vals[0]) + " (n=1)"
    return f"{fmt.format(st.fmean(vals))} ± {fmt.format(st.stdev(vals))}"


def collect(arms):
    out = {}
    for arm in arms:
        runs = [run_dir(arm, s) for s in SEEDS]
        runs = [r for r in runs if (r / "dispatch_log.csv").exists()]
        if runs:
            out[arm] = [per_run(r) for r in runs]
    return out


def render(data: dict) -> str:
    L = ["# Log-only mechanism tables (Prompt D Step 4) — mean ± SD over seeds", ""]
    L += ["## 1. Ground-truth energy per dispatch (J)", "",
          "| arm | all | before 80% crossing | after 80% crossing |", "|---|---|---|---|"]
    for a, rs in data.items():
        L.append(f"| {a} | {agg([r['E_all'] for r in rs], '{:.1f}')} | "
                 f"{agg([r['E_before'] for r in rs], '{:.1f}')} | "
                 f"{agg([r['E_after'] for r in rs], '{:.1f}')} |")
    L += ["", "## 2. Upload distance and position at training finish", "",
          "| arm | mean d at training finish (m) | share finishing inside [400, 1600] |",
          "|---|---|---|"]
    for a, rs in data.items():
        L.append(f"| {a} | {agg([r['d'] for r in rs], '{:.0f}')} | "
                 f"{agg([r['inside'] for r in rs], '{:.3f}')} |")
    L += ["", "## 3. True time per dispatch (s)", "",
          "| arm | mean T_train | mean T_up |", "|---|---|---|"]
    for a, rs in data.items():
        L.append(f"| {a} | {agg([r['T_train'] for r in rs], '{:.3f}')} | "
                 f"{agg([r['T_up'] for r in rs], '{:.3f}')} |")
    L += ["", "## 4. Dispatch share per tier (xavier / orin / thor)", "",
          "| arm | rounds 1–500 | last 500 rounds of the run |", "|---|---|---|"]
    for a, rs in data.items():
        f = " / ".join(agg([r[f"first_{t}"] for r in rs], "{:.3f}") for t in TIERS)
        l = " / ".join(agg([r[f"last_{t}"] for r in rs], "{:.3f}") for t in TIERS)
        L.append(f"| {a} | {f} | {l} |")
    L += ["", "## 5. Position-estimate error: mean |x̂_finish − x_finish| (m)", "",
          "| arm | error |", "|---|---|"]
    for a, rs in data.items():
        L.append(f"| {a} | {agg([r['pos_err'] for r in rs], '{:.1f}')} |")
    return "\n".join(L) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = render({**collect(S1_ARMS), **collect(S2_ARMS)})
    print(text)
    if args.out:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
