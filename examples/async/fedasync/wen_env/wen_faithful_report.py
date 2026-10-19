"""
Prompt E reporting for the `wen` arm.

    smoke  <run> ...   Step 4 plumbing table (accuracy not read)
    tta    <run> ...   Step 6: TTA@80/85/90% (online clock) for any runs

Decision rows (wen_decisions.csv) are joined to dispatch rows in order: every
Wen decision is exactly one dispatch, and the join is checked (pick == client_id).
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import Counter
from pathlib import Path

import wen_constants as K
import wen_main_report as R

TIERS = ("xavier", "orin", "thor")
THRESHOLDS = (0.80, 0.85, 0.90)


def _rows(p: Path):
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def circ(a: float, b: float) -> float:
    d = abs(a - b) % K.ROAD_LENGTH_M
    return min(d, K.ROAD_LENGTH_M - d)


def joined(run: Path):
    disp = _rows(run / "dispatch_log.csv")
    dec = _rows(run / "wen_decisions.csv")
    if len(dec) != len(disp) or any(int(a["pick"]) != int(b["client_id"])
                                    for a, b in zip(dec, disp)):
        raise SystemExit(f"{run}: decisions do not line up with dispatches")
    return disp, dec


def gini(counts) -> float:
    x = sorted(counts)
    n, total = len(x), sum(x)
    if total == 0:
        return 0.0
    return sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(x)) / (n * total)


def smoke_row(run: Path) -> dict:
    disp, dec = joined(run)
    rounds = max(int(r["round"]) for r in disp)
    per_vehicle = Counter(int(r["client_id"]) for r in disp)
    cand = sum(int(r["n_candidates"]) for r in dec)
    return {
        "run": run.name,
        "s_per_round": max(float(r["t_finish_clean"]) for r in disp) / rounds,
        "share": {t: sum(r["tier"] == t for r in disp) / len(disp) for t in TIERS},
        "fallback": st.fmean(int(r["fallback"]) for r in dec),
        "cut": {(k, t): sum(int(r[f"n_cut_{k}_{t}"]) for r in dec) /
                max(1, sum(int(r[f"n_candidates_{t}"]) for r in dec))
                for k in ("T_only", "E_only", "both") for t in TIERS},
        "cut_total": sum(int(r["n_candidates"]) - int(r["n_feasible"]) for r in dec) / cand,
        "pareto": Counter(int(r["n_pareto"]) for r in dec),
        "distinct": len(per_vehicle),
        "gini": gini([per_vehicle.get(c, 0) for c in range(1, K.N_CLIENTS + 1)]),
        "xhat_err": st.fmean(circ(float(a["x_hat_upload"]), float(b["x_finish"]))
                             for a, b in zip(dec, disp)),
        "upload_dist": st.fmean(abs(float(r["x_finish"]) - K.RSU_X_M) for r in disp),
        "d_finish": st.fmean(float(r["d_finish"]) for r in disp),
        "waste": sum(r["accepted"] == "0" for r in disp) / len(disp),
    }


def render_smoke(rows) -> str:
    L = ["# Prompt E, Step 4: Wen smoke (200 rounds, seed 4126; accuracy not read)", "",
         "| run | s/round | share X / O / T | fallback | candidates cut (all) | "
         "distinct vehicles | Gini | mean \\|x̂ − x\\| at upload (m) | "
         "mean \\|x_upload − 1000\\| (m) | mean d (m) | waste |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['run']} | {r['s_per_round']:.3f} | "
                 + " / ".join(f"{r['share'][t]:.2f}" for t in TIERS)
                 + f" | {r['fallback']:.3f} | {r['cut_total']:.3f} | {r['distinct']} | "
                 f"{r['gini']:.3f} | {r['xhat_err']:.0f} | {r['upload_dist']:.0f} | "
                 f"{r['d_finish']:.0f} | {r['waste']:.3f} |")
    L += ["", "Constraint cuts per tier (share of that tier's candidates):", "",
          "| run | tier | T only | E only | both |", "|---|---|---|---|---|"]
    for r in rows:
        for t in TIERS:
            L.append(f"| {r['run']} | {t} | {r['cut'][('T_only', t)]:.3f} | "
                     f"{r['cut'][('E_only', t)]:.3f} | {r['cut'][('both', t)]:.3f} |")
    L += ["", "n_pareto distribution (decisions):", ""]
    for r in rows:
        L.append(f"- {r['run']}: " + ", ".join(f"{k}: {v}" for k, v in sorted(r["pareto"].items())))
    return "\n".join(L) + "\n"


def tta_row(run: Path) -> dict:
    curve = R.accuracy_curve(run)
    out = {"run": run.name, "last_round": curve[-1][0], "last_acc": curve[-1][2]}
    for thr in THRESHOLDS:
        hit = next(((t, rnd) for rnd, t, a in curve if a >= thr), None)
        out[thr] = hit
    return out


def render_tta(rows) -> str:
    L = ["| run | TTA@80% | TTA@85% | TTA@90% | last round | last accuracy |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        cells = []
        for thr in THRESHOLDS:
            h = r[thr]
            cells.append(f"{h[0]:.0f} s (r{h[1]})" if h else
                         f"not reached (last {r['last_acc']:.4f})")
        L.append(f"| {r['run']} | " + " | ".join(cells) +
                 f" | {r['last_round']} | {r['last_acc']:.4f} |")
    return "\n".join(L) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "tta"))
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.mode == "smoke":
        text = render_smoke([smoke_row(r) for r in args.runs])
    else:
        text = render_tta([tta_row(r) for r in args.runs])
    print(text)
    if args.out:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
