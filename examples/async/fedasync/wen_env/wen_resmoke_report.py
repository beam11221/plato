"""
Prompt C, Step 3: re-smoke numbers for the selection rule. Plumbing only --
accuracy is not read.

Per cell: s/round, dispatch share per tier, share of explored picks, mean
rank_of_pick among exploit picks, the V_max - V_min distribution, and (cells
with Stage 0) the rejection rate from the server log.

Usage:
    python wen_env/wen_resmoke_report.py runs/wen_env/*_smoke2 --out wen_env/evidence/envmodel_resmoke.md
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from pathlib import Path

import wen_clock_replay

TIERS = ("xavier", "orin", "thor")
STAGE0 = re.compile(r"\[WenStage0\] summary .*candidates=(\d+) rejected=(\d+)")


def _rows(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def stage0_rejection(run: Path) -> str:
    rate = "—"
    for log in run.glob("server_*.log"):
        for line in log.read_text(errors="replace").splitlines():
            m = STAGE0.search(line)
            if m:
                rate = f"{int(m.group(2)) / max(int(m.group(1)), 1):.1%}"
    return rate


def summarise(run: Path) -> dict:
    d = _rows(run / "dispatch_log.csv")
    rounds = max(int(c["round"]) for c in _rows(run / "cycle_log.csv"))
    explored = [r for r in d if r["explored"] != ""]
    exploit = [r for r in explored if r["explored"] == "0"]
    spreads = [float(r["V_max"]) - float(r["V_min"]) for r in d if r["V_max"] != ""]
    q = st.quantiles(spreads, n=4) if len(spreads) >= 4 else None
    return {
        "run": run.name,
        "rule": d[0]["selection_rule"] + (f" ε={d[0]['epsilon']}" if d[0]["epsilon"] else ""),
        "s_per_round": max(float(r["t_finish_clean"]) for r in d) / rounds,
        "share": {t: sum(r["tier"] == t for r in d) / len(d) for t in TIERS},
        "explored": (sum(r["explored"] == "1" for r in explored) / len(explored)
                     if explored else None),
        "rank_exploit": (st.fmean(int(r["rank_of_pick"]) for r in exploit
                                  if r["rank_of_pick"] != "") if exploit else None),
        "spread": q and (q[0], q[1], q[2], max(spreads)),
        "stage0_rejection": stage0_rejection(run),
        "clock": wen_clock_replay.replay(run)["max_abs_diff_s"],
    }


def render(results):
    out = ["# Prompt C, Step 3: re-smoke (200 rounds, seed 4126; accuracy not read)", "",
           "| cell | rule | s / round | share X / O / T | explored | mean rank (exploit) "
           "| V_max − V_min: q1 / median / q3 / max | Stage 0 rejected | clock diff |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        sh = " / ".join(f"{r['share'][t]:.2f}" for t in TIERS)
        exp = "—" if r["explored"] is None else f"{r['explored']:.3f}"
        rank = "—" if r["rank_exploit"] is None else f"{r['rank_exploit']:.2f}"
        spread = "—" if not r["spread"] else " / ".join(f"{v:.4f}" for v in r["spread"])
        out.append(f"| {r['run']} | {r['rule']} | {r['s_per_round']:.3f} | {sh} | {exp} | "
                   f"{rank} | {spread} | {r['stage0_rejection']} | {r['clock']:.1e} |")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = render([summarise(r) for r in args.runs])
    print(text)
    if args.out:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
