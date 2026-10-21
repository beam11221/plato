"""
Prompt J (Polaris in S1, seed 4126): mechanical report.

    --smoke     Step 4 table and per-50-round q statistics (accuracy not read)

Usage:
    python wen_env/polaris_report.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import Counter
from pathlib import Path

import wen_constants as K
from wen_faithful_report import gini

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs" / "wen_env"
TIERS = ("xavier", "orin", "thor")


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def dispatch_summary(disp: list[dict]) -> dict:
    rounds = max(int(r["round"]) for r in disp)
    counts = Counter(int(r["client_id"]) for r in disp)
    return {
        "s_per_round": max(float(r["t_finish_clean"]) for r in disp) / rounds,
        "share": {t: sum(r["tier"] == t for r in disp) / len(disp) for t in TIERS},
        "distinct": len(counts),
        "gini": gini([counts.get(c, 0) for c in range(1, K.N_CLIENTS + 1)]),
        "upload_dist": st.fmean(abs(float(r["x_finish"]) - K.RSU_X_M) for r in disp),
    }


def q_windows(dec: list[dict], width: int) -> list[dict]:
    out = []
    last = max(int(d["round"]) for d in dec)
    for lo in range(1, last + 1, width):
        w = [d for d in dec if lo <= int(d["round"]) < lo + width]
        if not w:
            continue
        G = [g for d in w for g in json.loads(d["G"])]
        tau = [t for d in w for t in json.loads(d["tau"])]
        out.append({"rounds": f"{lo}–{lo + width - 1}", "calls": len(w),
                    "max_q": st.median(float(d["max_q"]) for d in w),
                    "eff_pool": st.median(float(d["eff_pool"]) for d in w),
                    "G": (min(G), max(G)), "tau": (min(tau), max(tau))})
    return out


def render_smoke(run: Path) -> list[str]:
    disp = rows(run / "dispatch_log.csv")
    dec = rows(run / "polaris_decisions.csv")
    m = dispatch_summary(disp)
    share = " / ".join(f"{m['share'][t]:.2f}" for t in TIERS)
    out = [f"## Step 4 — smoke ({run.name}; accuracy not read)", "",
           "| s/round | share X / O / T | distinct vehicles | Gini | "
           "mean \\|x_upload − 1000\\| (m) | dispatches | selection calls |",
           "|---|---|---|---|---|---|---|",
           f"| {m['s_per_round']:.3f} | {share} | {m['distinct']} | {m['gini']:.3f} | "
           f"{m['upload_dist']:.0f} | {len(disp)} | {len(dec)} |", "",
           "| rounds | calls | median max q | median effective pool 1/Σq² | G range | τ range |",
           "|---|---|---|---|---|---|"]
    for w in q_windows(dec, 50):
        out.append(f"| {w['rounds']} | {w['calls']} | {w['max_q']:.4f} | {w['eff_pool']:.2f} | "
                   f"{w['G'][0]:.3f}–{w['G'][1]:.3f} | {w['tau'][0]:.2f}–{w['tau'][1]:.2f} |")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", default="S1_polaris_s4126_smoke")
    args = parser.parse_args()
    if args.smoke:
        print("\n".join(render_smoke(RUNS / args.run)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
