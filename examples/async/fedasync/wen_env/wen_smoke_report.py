"""
Prompt B, Step 9: the smoke-run numbers, from the run directories' own logs.

Plumbing only. Accuracy is deliberately not reported.

Usage:
    python wen_env/wen_smoke_report.py runs/wen_env/*_smoke [--out evidence/envmodel_smoke.md]
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import wen_clock_replay

TIERS = ("xavier", "orin", "thor")


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row, key):
    return float(row[key]) if row.get(key, "") != "" else None


def summarise(run: Path) -> dict:
    d = _rows(run / "dispatch_log.csv")
    pool = _rows(run / "pool_log.csv")
    first = d[0]
    out = {"run": run.name, "scenario": first["scenario"], "arm": first["arm"],
           "dispatches": len(d)}

    train = sum(_f(r, "T_train_true") for r in d)
    up = sum(_f(r, "T_up_true") for r in d)
    out["upload_share"] = up / (train + up)

    per = defaultdict(list)
    for r in d:
        per[r["tier"]].append(r)
    out["tiers"] = {}
    for tier in TIERS:
        rows = per.get(tier, [])
        if not rows:
            continue
        hat = [_f(r, "T_hat_total") for r in rows if r["T_hat_total"] != ""]
        hat_tr = [_f(r, "T_hat_train") for r in rows if r["T_hat_train"] != ""]
        s_tau = [_f(r, "s_tau_hat") for r in rows if r["s_tau_hat"] != ""]
        out["tiers"][tier] = {
            "n": len(rows),
            "true_total": st.fmean(_f(r, "duration") for r in rows),
            "true_train": st.fmean(_f(r, "T_train_true") for r in rows),
            "true_up": st.fmean(_f(r, "T_up_true") for r in rows),
            "hat_total": st.fmean(hat) if hat else None,
            "hat_train": st.fmean(hat_tr) if hat_tr else None,
            "s_tau": st.fmean(s_tau) if s_tau else None,
            "energy": st.fmean(_f(r, "E_train") + _f(r, "E_up") for r in rows),
            "waste": sum(r["accepted"] == "0" for r in rows) / len(rows),
        }
    out["waste_fraction"] = sum(r["accepted"] == "0" for r in d) / len(d)
    out["waste_gpu_s"] = sum(_f(r, "T_train_true") for r in d if r["accepted"] == "0")
    out["waste_up_s"] = sum(_f(r, "T_up_true") for r in d if r["accepted"] == "0")
    out["waste_energy_j"] = sum(_f(r, "E_train") + _f(r, "E_up")
                                for r in d if r["accepted"] == "0")
    if pool:
        out["median_feasible_idle"] = st.median(int(p["feasible_gpu"]) for p in pool)
        out["median_eligible_idle"] = st.median(int(p["eligible"]) for p in pool)
    out["sim_seconds"] = max(_f(r, "t_finish_clean") for r in d)
    cycles = _rows(run / "cycle_log.csv")
    rounds = max(int(c["round"]) for c in cycles)
    out["seconds_per_round"] = out["sim_seconds"] / max(rounds, 1)
    out["clock"] = wen_clock_replay.replay(run)
    return out


def render(results: list[dict]) -> str:
    lines = ["# Prompt B, Step 9: smoke-run numbers (plumbing only; accuracy not read)", ""]
    lines.append("| cell | dispatches | upload share | waste | median feasible-idle "
                 "(gpu T̂) | median in-coverage idle | s / round | clock max diff (s) "
                 "| order residuals |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['run']} | {r['dispatches']} | {r['upload_share']:.3f} | "
            f"{r['waste_fraction']:.3f} | {r.get('median_feasible_idle', '—')} | "
            f"{r.get('median_eligible_idle', '—')} | {r['seconds_per_round']:.3f} | "
            f"{r['clock']['max_abs_diff_s']:.2e} | {r['clock']['order_residuals']} |")
    lines.append("")
    lines.append("## Per tier: true vs estimated (means over dispatches)")
    lines.append("")
    lines.append("| cell | tier | n | true T_train | true T_up | true total | T̂_train | "
                 "T̂ total | s(τ̂) | energy (J) | waste |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        for tier, t in r["tiers"].items():
            fmt = lambda v, p=3: "—" if v is None else f"{v:.{p}f}"  # noqa: E731
            lines.append(
                f"| {r['run']} | {tier} | {t['n']} | {t['true_train']:.3f} | "
                f"{t['true_up']:.3f} | {t['true_total']:.3f} | {fmt(t['hat_train'])} | "
                f"{fmt(t['hat_total'])} | {fmt(t['s_tau'])} | {t['energy']:.1f} | "
                f"{t['waste']:.3f} |")
    lines.append("")
    lines.append("## Waste (S2)")
    lines.append("")
    for r in results:
        if r["scenario"] == "S2":
            lines.append(f"- {r['run']}: {r['waste_fraction']:.1%} of dispatches; "
                         f"{r['waste_gpu_s']:.1f} GPU-s, {r['waste_up_s']:.1f} upload-s, "
                         f"{r['waste_energy_j']:.1f} J wasted")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    results = [summarise(r) for r in args.runs]
    text = render(results)
    print(text)
    if args.out:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
