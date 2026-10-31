"""
Prompt M B4: tier-mix sweep report, seed 4126 (mechanical only).

M0 = the Part A runs (S1_<arm>_s4126_pm); mixes = S1_<mix>_<arm>_s4126.

    PYTHONPATH=<repo>:sumo_close_loop python wen_env/tier_sweep_report.py \
        --out wen_env/evidence/tier_sweep_s4126.md

Definitions as s1_final.md: TTA@X = online-clock time of the first evaluated
round with accuracy >= X; energy to X = ground-truth J of every dispatch that
finished by then; s/round to X = TTA / rounds to X. Missing runs are listed as
"not run".
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import wen_constants as K  # noqa: E402
import wen_formulas as F  # noqa: E402
import wen_main_report as R  # noqa: E402
import wen_partition  # noqa: E402
from mechanism_report import gini  # noqa: E402
from wen_faithful import cifar10_train_targets  # noqa: E402

HERE = Path(__file__).resolve().parent
FEDASYNC = HERE.parent
RUNS = Path(os.environ.get("PROMPTM_RUNS", FEDASYNC / "runs" / "wen_env"))
MIXES = list(K.FLEET_MIXES)
ARMS = [("Random", "random"), ("Ours (gpu)", "gpu"), ("Polaris", "polaris"),
        ("Wen et al. (wen_fresh)", "wen_fresh")]
THR = (0.80, 0.85, 0.90)
TIERS = ("xavier", "orin", "thor")
_PART = None


def run_of(mix: str, arm: str) -> Path:
    return RUNS / (f"S1_{arm}_s4126_pm" if mix == "M0" else f"S1_{mix}_{arm}_s4126")


def done(run: Path) -> bool:
    return (run / "dispatch_log.csv").exists() and any((run / "results").glob("*.csv"))


def part():
    global _PART
    if _PART is None:
        _PART = wen_partition.build(cifar10_train_targets(FEDASYNC / "data"), 4126)
    return _PART


def coverage(run: Path) -> dict:
    merged = [r for r in R._rows(run / "merge_log.csv") if r["accepted"] == "1"]
    out = {}
    for check in (500, 1000, None):
        seen = {int(r["client_id"]) for r in merged
                if check is None or int(r["round"]) <= check}
        out[check or "end"] = sum(len(part().indices[c]) for c in seen) / K.TRAIN_IMAGES
    return out


def metrics(run: Path) -> dict:
    curve = R.accuracy_curve(run)
    d = R._rows(run / "dispatch_log.csv")
    m = {"max_acc": max(a for _, _, a in curve), "rounds": curve[-1][0],
         "spr": curve[-1][1] / curve[-1][0], "hit": {}}
    for thr in THR:
        hit = next(((rnd, t) for rnd, t, a in curve if a >= thr), None)
        if hit is None:
            m["hit"][thr] = None
            continue
        e = sum(float(r["E_train"]) + float(r["E_up"]) for r in d
                if float(r["t_finish_clean"]) <= hit[1]) / 1000
        m["hit"][thr] = {"tta": hit[1], "rounds": hit[0], "energy": e, "spr": hit[1] / hit[0]}
    last = max(int(r["round"]) for r in d)
    first = [r for r in d if int(r["round"]) <= 500]
    tail = [r for r in d if int(r["round"]) > last - 500]
    m["first"] = {t: sum(r["tier"] == t for r in first) / len(first) for t in TIERS}
    m["last"] = {t: sum(r["tier"] == t for r in tail) / len(tail) for t in TIERS}
    m["whole"] = {t: sum(r["tier"] == t for r in d) / len(d) for t in TIERS}
    per = Counter(int(r["client_id"]) for r in d)
    m["gini"] = gini([per.get(c, 0) for c in range(1, 31)])
    m["distinct"] = len(per)
    m["cov"] = coverage(run)
    m["upload"] = st.fmean(abs(float(r["x_finish"]) - K.RSU_X_M) for r in d)
    m["curve"] = curve
    m["tiers_seen"] = {r["tier"] for r in d}
    if (run / "wen_decisions.csv").exists():
        dec = R._rows(run / "wen_decisions.csv")
        cuts = {}
        for t in TIERS:
            n = sum(int(r[f"n_candidates_{t}"]) for r in dec)
            c = sum(int(r[f"n_cut_{k}_{t}"]) for r in dec for k in ("T_only", "E_only", "both"))
            cuts[t] = c / n if n else None
        m["wen_cuts"] = cuts
        m["fallback"] = sum(int(r["fallback"]) for r in dec)
    m["wall_h"] = wall_hours(run)
    return m


def wall_hours(run: Path) -> float | None:
    logs = sorted(run.glob("server_*.log"))
    if not logs:
        return None
    return (logs[0].stat().st_mtime - (run / "config_used.toml").stat().st_mtime) / 3600


def fmt(v, f="{:,.0f}"):
    return "—" if v is None else f.format(v)


def hetero_index(mix: str, d_bar: float) -> float:
    """CV over the 30 vehicles of datasheet T_train + T_up at distance d_bar."""
    t_up = F.upload_seconds(F.distance_to_rsu(K.RSU_X_M + d_bar), 1.0, 357_566_784)
    tiers = [t for t, n in zip(TIERS, K.FLEET_MIXES[mix]) for _ in range(n)]
    tot = np.array([F.train_seconds(K.FLOPS_FWD_PER_IMAGE, K.partition_size(c + 1),
                                    K.LOCAL_EPOCHS, K.TIER_TFLOPS_FP16[t]) + t_up
                    for c, t in enumerate(tiers)])
    return float(tot.std() / tot.mean())


def gain(a, b):
    return None if a is None or b is None else 100 * (b - a) / b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    figdir = a.out.parent
    data = {(mix, arm): metrics(run_of(mix, arm)) for mix in MIXES for _, arm in ARMS
            if done(run_of(mix, arm))}
    L = ["# Tier-mix sweep, Scenario 1, seed 4126 (Prompt M B4; mechanical)", "",
         "M0 = the Part A runs (they replace the earlier seed-4126 runs, author's "
         "decision of 31 Oct). Single seed: every direction below is reported, not "
         "evaluated.", ""]

    # 1. per mix
    L += ["## 1. Per mix: time, energy, rounds and s/round to 80 / 85 / 90 %", ""]
    for mix in MIXES:
        x, o, t = K.FLEET_MIXES[mix]
        L += [f"### {mix} — Xavier / Orin / Thor = {x} / {o} / {t}", "",
              "| method | " + " | ".join(f"TTA@{int(h * 100)}% (s)" for h in THR) + " | "
              + " | ".join(f"energy to {int(h * 100)}% (kJ)" for h in THR) + " | "
              + " | ".join(f"rounds to {int(h * 100)}%" for h in THR) + " | "
              + " | ".join(f"s/round to {int(h * 100)}%" for h in THR) + " | max accuracy |",
              "|---|" + "---|" * 13]
        for label, arm in ARMS:
            m = data.get((mix, arm))
            if m is None:
                L.append(f"| {label} | not run |" + " |" * 12)
                continue
            cells = []
            for key, f_ in (("tta", "{:,.0f}"), ("energy", "{:.1f}"), ("rounds", "{:,}"),
                            ("spr", "{:.3f}")):
                for h in THR:
                    hit = m["hit"][h]
                    cells.append(fmt(hit[key], f_) if hit else
                                 f"not reached (max {m['max_acc']:.4f})")
            L.append(f"| {label} | " + " | ".join(cells) + f" | {m['max_acc']:.4f} |")
        L.append("")

    # 2. selection behaviour
    L += ["## 2. Selection behaviour per mix", "",
          "| mix | method | share X / O / T r1–500 | share X / O / T last 500 | Gini | "
          "distinct | image coverage r500 / r1000 / end | mean \\|x_upload − 1000\\| (m) |",
          "|---|---|---|---|---|---|---|---|"]
    for mix in MIXES:
        for label, arm in ARMS:
            m = data.get((mix, arm))
            if m is None:
                continue
            L.append(f"| {mix} | {label} | "
                     + " / ".join(f"{m['first'][t]:.2f}" for t in TIERS) + " | "
                     + " / ".join(f"{m['last'][t]:.2f}" for t in TIERS) + " | "
                     f"{m['gini']:.3f} | {m['distinct']} | "
                     + " / ".join(f"{m['cov'][k]:.3f}" for k in (500, 1000, "end"))
                     + f" | {m['upload']:.0f} |")
    L += ["", "**Wen et al.: limits (W5 rule per mix), realised cuts and tier shares**", "",
          "| mix | T_max (s) | E_max (J) | cut X / O / T (share of candidate rows) | "
          "fallback decisions | share X / O / T (whole run) |", "|---|---|---|---|---|---|"]
    for mix in MIXES:
        lim = json.loads((HERE / "evidence" / f"wen_limits_{mix}.json").read_text())
        m = data.get((mix, "wen_fresh"))
        cuts = (" / ".join(fmt(m["wen_cuts"][t], "{:.3f}") for t in TIERS) if m else "not run")
        L.append(f"| {mix} | {lim['t_max']:.3f} | {lim['e_max']:.3f} | {cuts} | "
                 f"{m['fallback'] if m else '—'} | "
                 + (" / ".join(f"{m['whole'][t]:.2f}" for t in TIERS) if m else "—") + " |")
    L.append("")

    # 3. cross-mix summary
    d_bar = data[("M0", "random")]["upload"] if ("M0", "random") in data else 495.0
    L += ["## 3. Cross-mix summary", "",
          f"Gain = (other − Ours) / other, %; positive = Ours lower. Heterogeneity index = "
          f"CV over the 30 vehicles of datasheet T_train + T_up at Random M0's mean upload "
          f"distance ({d_bar:.0f} m, |χ|² = 1).", "",
          "| mix | heterogeneity index | Ours vs Random TTA@80 / @85 | Ours vs Random "
          "energy @80 / @85 | Ours vs Polaris TTA@80 / @85 | Ours vs Polaris energy "
          "@80 / @85 | Wen vs Random TTA@80 / @85 (positive = Wen lower) |",
          "|---|---|---|---|---|---|---|"]

    def pick(mix, arm, h, key):
        m = data.get((mix, arm))
        return m["hit"][h][key] if m and m["hit"][h] else None

    for mix in MIXES:
        cells = []
        for other in ("random", "polaris"):
            for key in ("tta", "energy"):
                cells.append(" / ".join(fmt(gain(pick(mix, "gpu", h, key),
                                                  pick(mix, other, h, key)), "{:+.1f}")
                                        for h in (0.80, 0.85)))
        wen = " / ".join(fmt(gain(pick(mix, "wen_fresh", h, "tta"), pick(mix, "random", h, "tta")),
                             "{:+.1f}") for h in (0.80, 0.85))
        L.append(f"| {mix} | {hetero_index(mix, d_bar):.3f} | " + " | ".join(cells) +
                 f" | {wen} |")
    L.append("")

    # 4. mechanism maps for M3 and T1
    L += ["## 4. Mechanism maps (A4 plots 1, 6, 9) for M3 and T1", ""]
    for mix in ("M3", "T1"):
        runs = {arm: run_of(mix, arm) for arm in ("gpu", "polaris", "wen_fresh")}
        if not all(done(r) for r in runs.values()):
            L += [f"{mix}: not all runs finished.", ""]
            continue
        out = figdir / f"{a.out.stem}_maps_{mix}.md"
        subprocess.run([sys.executable, str(HERE / "mechanism_report.py"), "--maps-only",
                        "--tag", mix, "--gpu", str(runs["gpu"]),
                        "--polaris", str(runs["polaris"]), "--wen", str(runs["wen_fresh"]),
                        "--out", str(out)], check=True)
        for line in out.read_text().splitlines():
            if line.startswith("**") or line.startswith("## "):
                L.append(line.replace("## ", f"### {mix} — ", 1))
        L.append("")

    # 5. P-table
    L += ["## 5. Predictions v3.7 (seed 4126 directions; single seed — not evaluated)", "",
          "| P | mix | comparison | seed 4126 | direction | status |", "|---|---|---|---|---|---|"]
    preds = [("P44", "random", "tta", 0.80, "Ours TTA@80 < Random"),
             ("P45", "polaris", "tta", 0.80, "Ours TTA@80 < Polaris"),
             ("P46", "wen_fresh", "tta", 0.80, "Ours TTA@80 < Wen et al."),
             ("P47", "random", "energy", 0.85, "Ours energy to 85 % < Random")]
    for p, other, key, h, text in preds:
        for mix in MIXES[1:]:
            ours, oth = pick(mix, "gpu", h, key), pick(mix, other, h, key)
            if data.get((mix, "gpu")) is None or data.get((mix, other)) is None:
                L.append(f"| {p} | {mix} | {text} | not run | — | — |")
                continue
            if ours is None and oth is None:
                cmp_, dirn = "neither reached", "—"
            else:
                o = f"{ours:,.1f}" if ours is not None else "n/r"
                t = f"{oth:,.1f}" if oth is not None else "n/r"
                cmp_ = f"{o} vs {t}"
                lower = oth is None or (ours is not None and ours < oth)
                dirn = "as predicted" if lower else "not as predicted"
            L.append(f"| {p} | {mix} | {text} | {cmp_} | {dirn} | single seed — not evaluated |")
    g = {mix: gain(pick(mix, "gpu", 0.80, "tta"), pick(mix, "random", 0.80, "tta"))
         for mix in ("T1", "M3")}
    dirn = ("—" if None in g.values() else
            "as predicted" if g["T1"] > g["M3"] else "not as predicted")
    L += [f"| P48 | T1 vs M3 | Ours' TTA@80 gain over Random larger in T1 than M3 | "
          f"{fmt(g['T1'], '{:+.1f}')} % vs {fmt(g['M3'], '{:+.1f}')} % | {dirn} | "
          "single seed — not evaluated |", ""]

    # 6. curves
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), sharey=True)
    for ax, mix in zip(axes.flat, MIXES):
        for label, arm in ARMS:
            m = data.get((mix, arm))
            if m:
                ax.plot([t for _, t, _ in m["curve"]], [acc for *_, acc in m["curve"]],
                        lw=0.8, label=label)
        ax.axhline(0.8, color="#999", lw=0.5, ls=":")
        ax.axhline(0.9, color="#999", lw=0.5, ls=":")
        ax.set_title(f"{mix} {K.FLEET_MIXES[mix]}", fontsize=9)
        ax.set_xlabel("online clock (s)")
        ax.set_ylim(0.3, 0.95)
    axes.flat[0].legend(fontsize=7)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    pc = figdir / f"{a.out.stem}_curves.png"
    fig.savefig(pc, dpi=120)
    plt.close(fig)
    L += [f"## 6. Accuracy vs online clock ![]({pc.name})", ""]

    # 7. wall times and health
    L += ["## 7. Run health", "", "| run | rounds | wall time (h) | online = replayed clock | "
          "absent tiers seen in log |", "|---|---|---|---|---|"]
    import wen_clock_replay
    for (mix, arm), m in sorted(data.items()):
        run = run_of(mix, arm)
        absent = {t for t, n in zip(TIERS, K.FLEET_MIXES[mix]) if n == 0}
        diff = wen_clock_replay.replay(run)["max_abs_diff_s"]
        L.append(f"| {run.name} | {m['rounds']} | {fmt(m['wall_h'], '{:.1f}')} | "
                 f"{'yes' if diff < 1e-6 else f'NO ({diff})'} | "
                 f"{sorted(absent & m['tiers_seen']) or 'none'} |")
    L.append("")
    a.out.write_text("\n".join(L))
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
