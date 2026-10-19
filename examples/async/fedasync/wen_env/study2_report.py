"""
Prompt I (Study 2, plug-in study, seed 4126): mechanical report.

    --step0     Step 0a eligibility table, 0b.2 replay table, 0c Stage 0 no-op check
    --smoke     Step 4 smoke table for the 5 Study 2 cells (accuracy not read)
    --study     Step 6 tables: 2a, 2b, cell 5, GPU limits and realised cuts,
                P26-P33 (seed 4126 directions, "single seed -- not evaluated"), curves

Usage:
    python wen_env/study2_report.py --step0 --smoke --study --out wen_env/evidence/study2_plugin_s4126.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

import wen_constants as K
import wen_main_report as R
import wen_study1 as S1m
from wen_faithful_report import circ, gini
from wen_ptable import run_dir

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs" / "wen_env"
SEED = 4126
TIERS = ("xavier", "orin", "thor")
THRESHOLDS = (0.80, 0.85, 0.90)
STAGE0_LINE = re.compile(r"\[WenStage0\] summary estimator=\w+ cycles=(\d+) candidates=(\d+) "
                         r"rejected=(\d+)")
STAGE0_CALL = re.compile(r"\[WenStage0\] round=\d+ .* pool=(\d+) feasible=(\d+) "
                         r"rejected=\[([\d, ]*)\] fallback=(\d)")
WEN_HOSTS = ("wen", "wen_fresh", "wen_fresh_s0", "wen_fresh_gpu", "wen_fresh_gpu_s0")
READING_B = {("S1_gpu", 4126), ("S1_cpu", 4126)}


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def fmt(v, f="{:.3f}"):
    return "—" if v is None else f.format(v)


def tier_of(cid: int) -> str:
    return "xavier" if cid <= 15 else "orin" if cid <= 25 else "thor"


# ---------------------------------------------------------------------------
# per-run metrics
# ---------------------------------------------------------------------------

def stage0_stats(run: Path) -> dict | None:
    """Rejection rate overall, by reason (share of eligible) and by tier
    (rejected / eligible of that tier), fallback rate, from stage0_log.csv; for
    the existing gpu + Stage 0 run (no stage0_log) from its server log."""
    log = run / "stage0_log.csv"
    if log.exists():
        r = rows(log)
        elig = sum(int(x["n_eligible"]) for x in r)
        out = {"calls": len(r), "rate": sum(int(x["n_rejected"]) for x in r) / elig,
               "fallback": st.fmean(int(x["fallback"]) for x in r)}
        for k in ("time", "energy", "both"):
            out[k] = sum(int(x[f"n_rej_{k}"]) for x in r) / elig
        for t in TIERS:
            e = sum(int(x[f"n_elig_{t}"]) for x in r)
            out[t] = sum(int(x[f"n_rej_{t}"]) for x in r) / e if e else None
        return out
    for lg in run.glob("server_*.log"):
        calls = [m for line in lg.read_text(errors="replace").splitlines()
                 if (m := STAGE0_CALL.search(line))]
        if not calls:
            continue
        elig = sum(int(m.group(1)) for m in calls)
        rej = Counter(tier_of(int(c)) for m in calls for c in m.group(3).split(",") if c.strip())
        out = {"calls": len(calls), "rate": sum(rej.values()) / elig,
               "fallback": st.fmean(int(m.group(4)) for m in calls),
               "time": None, "energy": None, "both": None}
        out.update({t: None for t in TIERS})
        out["rej_by_tier_count"] = dict(rej)
        return out
    return None


def metrics(cell: str) -> dict:
    run = run_dir(cell, SEED)
    m = S1m.one(cell, SEED)
    arm = cell.split("_", 1)[1]
    disp = rows(run / "dispatch_log.csv")
    if arm in WEN_HOSTS and arm != "wen":
        dec = rows(run / "wen_decisions.csv")
        if len(dec) != len(disp) or any(a["pick"] != b["client_id"] for a, b in zip(dec, disp)):
            raise SystemExit(f"{run}: decisions do not line up with dispatches")
        m["xhat_err"] = st.fmean(circ(float(a["x_hat_upload"]), float(b["x_finish"]))
                                 for a, b in zip(dec, disp))
        m["dec"] = dec
    if (cell, SEED) in READING_B and m["tta"][0.90] is None:
        curve = R.accuracy_curve(run)
        best = max(a for _, _, a in curve)
        m["tta"][0.90] = next(t for _, t, a in curve if a == best)
    m["stage0"] = stage0_stats(run)
    m["thor_whole"] = sum(r["tier"] == "thor" for r in disp) / len(disp)
    m["share"] = {t: sum(r["tier"] == t for r in disp) / len(disp) for t in TIERS}
    m["energy_total"] = sum(float(r["E_train"]) + float(r["E_up"]) for r in disp)
    m["rounds"] = max(int(r["round"]) for r in disp)
    m["distinct"] = len({r["client_id"] for r in disp})
    if m.get("dec"):
        decs = m["dec"]
        m["cuts"] = {t: tuple(sum(int(d[f"n_cut_{k}_{t}"]) for d in decs) /
                              max(1, sum(int(d[f"n_candidates_{t}"]) for d in decs))
                              for k in ("T_only", "E_only", "both")) for t in TIERS}
        m["fallback_wen"] = st.fmean(int(d["fallback"]) for d in decs)
    return m


def tta(m, thr):
    return fmt(m["tta"][thr], "{:.0f}")


def s0_cells(m):
    s = m["stage0"]
    if s is None:
        return "—", "—", "—", "—"
    reason = " / ".join(fmt(s[k], "{:.3f}") for k in ("time", "energy", "both"))
    tier = " / ".join(fmt(s[t], "{:.3f}") for t in TIERS)
    if "rej_by_tier_count" in s:
        tier = "counts " + " / ".join(str(s["rej_by_tier_count"].get(t, 0)) for t in TIERS)
    return f"{s['rate']:.3f}", reason, tier, f"{s['fallback']:.4f}"


def waste(m):
    return (f"{m['waste']:.3f} / {m['waste_gpu_s']:.0f} / {m['waste_j'] / 1e3:.1f}")


# ---------------------------------------------------------------------------
# Step 0
# ---------------------------------------------------------------------------

def render_step0() -> list[str]:
    out = ["## 7. Step 0 results", "", "### 0a — eligibility in every existing S2 run", "",
           "Per run: dispatches, dispatches with x_dispatch outside [400, 1600], dispatches to a "
           "vehicle still in flight (dispatched before its previous task's t_finish_clean).", "",
           "| run | dispatches | outside coverage | not idle |", "|---|---|---|---|"]
    dirs = sorted(p for p in list(RUNS.glob("S2_*")) + list(RUNS.glob("_failed/S2_*"))
                  if (p / "dispatch_log.csv").exists() and not p.is_symlink()
                  and not p.name.startswith(("S2_random_s0", "S2_wen_fresh_s0", "S2_wen_fresh_gpu")))
    for d in dirs:
        r = rows(d / "dispatch_log.csv")
        out_cov = sum(not (400.0 <= float(x["x_dispatch"]) <= 1600.0) for x in r)
        busy, notidle = {}, 0
        for x in r:
            c, t = int(x["client_id"]), float(x["t_dispatch"])
            notidle += c in busy and t < busy[c]
            busy[c] = float(x["t_finish_clean"])
        out.append(f"| {d.relative_to(RUNS)} | {len(r)} | {out_cov} | {notidle} |")
    out += ["", "### 0b.2 — 200-round replay with the current code (batch B)", "",
            "| arm family | replay run | source commit (dirty) | dispatch rows (r ≤ 200) | "
            "identical to the existing run |", "|---|---|---|---|---|"]
    for cell in ("S1_gpu", "S1_cpu", "S2_gpu_stage0", "S1_cpu_cal", "S1_wen", "S1_wen_fresh"):
        new = RUNS / f"{cell}_s4126_replay200"
        if not (new / "dispatch_log.csv").exists():
            out.append(f"| {cell} | not run | | | |")
            continue
        man = (new / "results" / "manifest.yaml").read_text()
        commit = re.search(r"source_commit: (\w+)", man).group(1)[:8]
        dirty = re.search(r"source_dirty: (\w+)", man).group(1)
        a = [_key(x) for x in rows(new / "dispatch_log.csv") if int(x["round"]) <= 200]
        b = [_key(x) for x in rows(run_dir(cell, SEED) / "dispatch_log.csv")
             if int(x["round"]) <= 200]
        out.append(f"| {cell} | {new.name} | {commit} ({dirty}) | {len(a)} | "
                   f"{'yes' if a == b else '**NO**'} |")
    new = RUNS / "S1_gpu_stage0_s4126_noop200"
    out += ["", "### 0c — Stage 0 in S1 (S1 gpu + Stage 0, 200 rounds)", ""]
    if (new / "stage0_log.csv").exists():
        s0 = rows(new / "stage0_log.csv")
        a = [_key(x) for x in rows(new / "dispatch_log.csv") if int(x["round"]) <= 200]
        b = [_key(x) for x in rows(run_dir("S1_gpu", SEED) / "dispatch_log.csv")
             if int(x["round"]) <= 200]
        out += [f"- Stage 0 calls: {len(s0)}; vehicles checked: "
                f"{sum(int(x['n_eligible']) for x in s0)}; rejected: "
                f"{sum(int(x['n_rejected']) for x in s0)}; fallbacks: "
                f"{sum(int(x['fallback']) for x in s0)}.",
                f"- Dispatch log (rounds ≤ 200, {len(a)} rows) identical to S1 gpu (Stage 0 off): "
                f"**{'yes' if a == b else 'NO'}**."]
    else:
        out.append("- not run")
    return out


def _key(r):
    return tuple(r[c] for c in ("cycle", "round", "client_id", "t_dispatch", "duration",
                                "t_finish_clean", "accepted"))


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

SMOKE = ("S2_random_s0", "S2_wen_fresh_s0", "S1_wen_fresh_gpu", "S2_wen_fresh_gpu",
         "S2_wen_fresh_gpu_s0")


def render_smoke() -> list[str]:
    out = ["## Step 4 — smoke (200 rounds, seed 4126; accuracy not read)", "",
           "| cell | s/round | share X / O / T | Stage 0 rejected (of eligible) | by reason "
           "time / energy / both | by tier X / O / T (rejected / eligible) | Stage 0 fallback | "
           "Wen fallback | waste (S2) | distinct vehicles | mean \\|x̂ − x\\| (m) | "
           "mean \\|x_upload − 1000\\| (m) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    cuts = []
    for cell in SMOKE:
        run = RUNS / f"{cell}_s4126_smoke"
        disp = rows(run / "dispatch_log.csv")
        rounds = max(int(r["round"]) for r in disp)
        share = " / ".join(f"{sum(r['tier'] == t for r in disp) / len(disp):.2f}" for t in TIERS)
        s0 = stage0_stats(run)
        rate, reason, tier, fb = s0_cells({"stage0": s0})
        dec = rows(run / "wen_decisions.csv") if (run / "wen_decisions.csv").exists() else None
        xerr = (st.fmean(circ(float(a["x_hat_upload"]), float(b["x_finish"]))
                         for a, b in zip(dec, disp)) if dec else None)
        wfb = st.fmean(int(d["fallback"]) for d in dec) if dec else None
        w = (sum(r["accepted"] == "0" for r in disp) / len(disp)) if cell.startswith("S2") else None
        out.append(f"| {cell} | {max(float(r['t_finish_clean']) for r in disp) / rounds:.3f} | "
                   f"{share} | {rate} | {reason} | {tier} | {fb} | {fmt(wfb, '{:.4f}')} | "
                   f"{fmt(w)} | {len({r['client_id'] for r in disp})} | {fmt(xerr, '{:.0f}')} | "
                   f"{st.fmean(abs(float(r['x_finish']) - 1000) for r in disp):.0f} |")
        if dec:
            for t in TIERS:
                n = max(1, sum(int(d[f"n_candidates_{t}"]) for d in dec))
                cuts.append(f"| {cell} | {t} | " + " | ".join(
                    f"{sum(int(d[f'n_cut_{k}_{t}']) for d in dec) / n:.3f}"
                    for k in ("T_only", "E_only", "both")) + " |")
    out += ["", "Constraint cuts per tier (share of that tier's candidates, Wen hosts):", "",
            "| cell | tier | T only | E only | both |", "|---|---|---|---|---|", *cuts]
    return out


# ---------------------------------------------------------------------------
# Study tables
# ---------------------------------------------------------------------------

def render_study(evidence: Path) -> list[str]:
    cells_2a = ("S2_random", "S2_random_s0", "S2_wen_fresh", "S2_wen_fresh_s0", "S2_gpu",
                "S2_gpu_stage0")
    cells_2b = ("S1_wen_fresh", "S1_wen_fresh_gpu", "S1_gpu", "S1_random",
                "S2_wen_fresh", "S2_wen_fresh_gpu", "S2_gpu", "S2_random")
    cells_5 = ("S2_wen_fresh", "S2_wen_fresh_s0", "S2_wen_fresh_gpu", "S2_wen_fresh_gpu_s0")
    data = {c: metrics(c) for c in dict.fromkeys(cells_2a + cells_2b + cells_5)}
    out = ["## 1. Study 2a — Stage 0 plug-in (S2, seed 4126)", "",
           "| cell | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | s/round | energy to 80% (kJ) | "
           "waste (disp / GPU-s / kJ) | Stage 0 rejected | by reason time / energy / both | "
           "by tier X / O / T | Stage 0 fallback | share X / O / T | "
           "mean \\|x_upload − 1000\\| (m) | Gini |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells_2a:
        m = data[c]
        rate, reason, tier, fb = s0_cells(m)
        out.append(f"| {c} | {tta(m, .80)} | {tta(m, .85)} | {tta(m, .90)} | {m['spr']:.3f} | "
                   f"{fmt(m['e80'] / 1e3 if m['e80'] else None, '{:.1f}')} | {waste(m)} | {rate} | "
                   f"{reason} | {tier} | {fb} | "
                   + " / ".join(f"{m['share'][t]:.2f}" for t in TIERS) +
                   f" | {m['upload_dist']:.0f} | {m['gini']:.3f} |")
    out += ["", "S2_gpu_stage0 is the existing run: its Stage 0 is reconstructed from the server "
            "log (rejection rate and per-tier rejection counts; reasons were not logged).", "",
            "## 2. Study 2b — GPU plug-in (seed 4126)", "",
            "| cell | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | s/round | energy to 80% (kJ) | "
            "waste (disp / GPU-s / kJ) | share X / O / T | Thor share | "
            "mean \\|x_upload − 1000\\| (m) | mean \\|x̂ − x\\| (m) | Gini | Wen cuts per tier "
            "X / O / T (T only, E only, both) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells_2b:
        m = data[c]
        cuts = ("; ".join(f"{t[0].upper()} {'/'.join(f'{v:.3f}' for v in m['cuts'][t])}"
                          for t in TIERS) if m.get("cuts") else "—")
        w = "—" if c.startswith("S1") else waste(m)
        out.append(f"| {c} | {tta(m, .80)} | {tta(m, .85)} | {tta(m, .90)} | {m['spr']:.3f} | "
                   f"{fmt(m['e80'] / 1e3 if m['e80'] else None, '{:.1f}')} | {w} | "
                   + " / ".join(f"{m['share'][t]:.2f}" for t in TIERS) +
                   f" | {m['thor_whole']:.3f} | {m['upload_dist']:.0f} | "
                   f"{fmt(m['xhat_err'], '{:.0f}')} | {m['gini']:.3f} | {cuts} |")
    out += ["", "## 3. Cell 5 — Wen 2 × 2 (S2): {CPU, GPU} F1/F2 × {Stage 0 off, on}", "",
            "| cell | F1/F2 | Stage 0 | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | waste (disp / "
            "GPU-s / kJ) | s/round | energy to 80% (kJ) | total energy (kJ) | rounds |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells_5:
        m = data[c]
        out.append(f"| {c} | {'GPU' if 'gpu' in c else 'CPU'} | {'on' if c.endswith('s0') else 'off'} | "
                   f"{tta(m, .80)} | {tta(m, .85)} | {tta(m, .90)} | {waste(m)} | {m['spr']:.3f} | "
                   f"{fmt(m['e80'] / 1e3 if m['e80'] else None, '{:.1f}')} | "
                   f"{m['energy_total'] / 1e3:.1f} | {m['rounds']} |")
    lim = json.loads((HERE / "evidence" / "wen_limits_gpu.json").read_text())
    out += ["", "## 4. GPU limits and realised cuts", "",
            f"T_max = **{lim['t_max']!r} s**, E_max = **{lim['e_max']!r} J** (P90 of the gpu-family "
            f"F1 / F2 over {lim['population']:,} (vehicle, sample) pairs; qualitative check "
            f"{'PASS' if not lim['qualitative_problems'] else 'FAIL'}). Per-tier table: "
            "`wen_limits_gpu.md` (copied below).", "",
            *(HERE / "evidence" / "wen_limits_gpu.md").read_text().split("\n")[5:], "",
            "Realised cuts in the runs (share of that tier's candidates; T only / E only / both) "
            "and Wen fallback rate:", "",
            "| cell | Xavier | Orin | Thor | fallback |", "|---|---|---|---|---|"]
    for c in ("S1_wen_fresh", "S1_wen_fresh_gpu", "S2_wen_fresh", "S2_wen_fresh_s0",
              "S2_wen_fresh_gpu", "S2_wen_fresh_gpu_s0"):
        m = data.get(c) or metrics(c)
        out.append(f"| {c} | " + " | ".join("/".join(f"{v:.3f}" for v in m["cuts"][t])
                                           for t in TIERS) + f" | {m['fallback_wen']:.4f} |")
    out += ["", "## 5. P26–P33 (seed 4126 direction; status: single seed — not evaluated)", "",
            *p_table(data), "", "## 6. Curves", "",
            "`study2_plugin_s4126_curves.png`: accuracy vs online clock, one panel per "
            "comparison group (2a random, 2a wen_fresh, 2b S1, 2b S2, cell 5).", ""]
    plot(evidence / "study2_plugin_s4126_curves.png")
    return out


def p_table(d) -> list[str]:
    def row(p, txt, ok, a, b):
        return (f"| {p} | {txt} | {a} vs {b} | {'as predicted' if ok else 'not as predicted'} | "
                "single seed — not evaluated |")
    lt = S1m.lower
    out = ["| P | comparison | seed 4126 | direction | status |", "|---|---|---|---|---|"]
    a, b = d["S2_random_s0"], d["S2_random"]
    out.append(row("P26", "random + Stage 0 waste fraction < random (S2)",
                   a["waste"] < b["waste"], f"{a['waste']:.3f}", f"{b['waste']:.3f}"))
    out.append(row("P27", "random + Stage 0 TTA@80% < random (S2)",
                   lt(a["tta"][0.80], b["tta"][0.80]), tta(a, .80), tta(b, .80)))
    a, b = d["S2_wen_fresh_s0"], d["S2_wen_fresh"]
    out.append(row("P28", "wen_fresh + Stage 0 waste fraction < wen_fresh (S2)",
                   a["waste"] < b["waste"], f"{a['waste']:.3f}", f"{b['waste']:.3f}"))
    a, b = d["S1_wen_fresh_gpu"], d["S1_wen_fresh"]
    out.append(row("P29", "wen_fresh + GPU mean |x̂ − x| < wen_fresh (S1)",
                   a["xhat_err"] < b["xhat_err"], f"{a['xhat_err']:.0f} m", f"{b['xhat_err']:.0f} m"))
    out.append(row("P30", "wen_fresh + GPU mean |x_upload − 1000| < wen_fresh (S1)",
                   a["upload_dist"] < b["upload_dist"], f"{a['upload_dist']:.0f} m",
                   f"{b['upload_dist']:.0f} m"))
    out.append(row("P31", "wen_fresh + GPU s/round < wen_fresh (S1)", a["spr"] < b["spr"],
                   f"{a['spr']:.3f}", f"{b['spr']:.3f}"))
    out.append(row("P32", "wen_fresh + GPU Thor share > wen_fresh (S1)",
                   a["thor_whole"] > b["thor_whole"], f"{a['thor_whole']:.3f}",
                   f"{b['thor_whole']:.3f}"))
    a, b = d["S2_wen_fresh_gpu_s0"], d["S2_wen_fresh_gpu"]
    out.append(row("P33", "wen_fresh + GPU + Stage 0 waste fraction ≤ wen_fresh + GPU (S2)",
                   a["waste"] <= b["waste"], f"{a['waste']:.3f}", f"{b['waste']:.3f}"))
    return out


def plot(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("2a: random (S2)", ("S2_random", "S2_random_s0")),
              ("2a: wen_fresh (S2)", ("S2_wen_fresh", "S2_wen_fresh_s0", "S2_gpu",
                                      "S2_gpu_stage0")),
              ("2b: S1", ("S1_wen_fresh", "S1_wen_fresh_gpu", "S1_gpu", "S1_random")),
              ("2b: S2", ("S2_wen_fresh", "S2_wen_fresh_gpu", "S2_gpu", "S2_random")),
              ("cell 5 (S2)", ("S2_wen_fresh", "S2_wen_fresh_s0", "S2_wen_fresh_gpu",
                               "S2_wen_fresh_gpu_s0"))]
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 4.4), sharey=True)
    for ax, (title, cells) in zip(axes, groups):
        for c in cells:
            curve = R.accuracy_curve(run_dir(c, SEED))
            ax.plot([p[1] for p in curve], [p[2] for p in curve], lw=1, label=c)
        for thr in THRESHOLDS:
            ax.axhline(thr, color="grey", lw=0.6, ls="--")
        ax.set_title(title)
        ax.set_xlabel("online clock (s)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("test accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step0", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--study", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lines = []
    if args.study:
        lines += render_study(args.out.parent) + [""]
    if args.step0:
        lines += render_step0() + [""]
    if args.smoke:
        lines += render_smoke() + [""]
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
