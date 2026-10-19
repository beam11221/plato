"""
TTA estimation on CIFAR-100 (Stage 13 and 13B).  Log-only; no training.
Extends tta_estimation.py (spec prompts/TTA_smoothing_analysis.md) to the
CIFAR-100 runs, at the user's request (2026-10-05).

    python3 tta_estimation_c100.py

The estimators E0-E5 and their parameters are tta_estimation.py's, unchanged
(median 11, sustained 3, MA 5, isotonic, saturating fit with RMSE <= 1 pt,
AUC over 30 min).  Only the targets change: 75 / 78 / 80 % are out of reach on
CIFAR-100 (plateau ~46-51 %), so 30 / 40 / 50 % are used, as in the Stage 13
reports.

The headline estimator stays E2, fixed on CIFAR-10 (tta_estimator_choice.json)
and NOT re-chosen here.  What the same rule would pick on CIFAR-100 is reported
for information only.

Replicate pairs (true difference zero):
  within-stage   13 #6 vs 13 #2, 13B #6 vs 13B #2
  cross-stage    13B rounds 1-4,000 vs Stage 13, for each of the six runs:
                 nothing in the config depends on the total number of rounds,
                 so the first 4,000 rounds of 13B are a re-run of Stage 13.
                 The 13B series is cut at round 4,000 for these pairs, so E3 /
                 E4 (which use the whole series) see the same span on both sides.
50 % is reached only after round 4,000 (13B), so it has one replicate pair.
Sanity: baseline+F faster than baseline / none in all four (stage, run) pairs.
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_stage7 as a7  # noqa: E402
import tta_estimation as te  # noqa: E402

te.TARGETS = TARGETS = (30, 40, 50)
EST = te.EST
OUT = te.OUT
RUNS = {1: ("nf", "baseline_open", "baseline / none"), 2: ("f", "baseline_open", "baseline + F"),
        3: ("f", "gpu_open", "gpu + F"), 4: ("f", "observed_open", "observed + F"),
        5: ("f", "cpu_open", "cpu + F"), 6: ("rep", "baseline_open", "baseline + F (replicate)")}
STAGES = {"13": "stage13", "13B": "stage13b"}


def series(prefix, cell, max_round=None):
    a7.PREFIX = prefix
    run = a7.parse(cell)
    rows = [(c["round"], c["sim_time_reconstructed"], c["accuracy"]) for c in run.curve
            if c["round"] % a7.EVAL_INTERVAL == 0 and (max_round is None or c["round"] <= max_round)]
    return rows, a7.waste(run)["overall"]["wasted_frac"]


def main():
    E, W, S4 = {}, {}, {}
    for sk, stem in STAGES.items():
        for n, (kind, cell, _) in RUNS.items():
            rows, w = series(f"{stem}{kind}_s4126", cell)
            E[(sk, n)] = te.estimate(rows)
            W[(sk, n)] = w
            if sk == "13B":
                S4[n] = te.estimate([r for r in rows if r[0] <= 4000])

    def val(e, est, t):
        return e["E5"] if est == "E5" else e[(est, t)]["t"]

    pairs = {30: [], 40: [], 50: []}
    labels = {30: [], 40: [], 50: []}
    for t in TARGETS:
        for sk in STAGES:
            pairs[t].append((E[(sk, 2)], E[(sk, 6)]))
            labels[t].append(f"Stage {sk}: #6 vs #2")
        if t < 50:
            for n in RUNS:
                pairs[t].append((E[("13", n)], S4[n]))
                labels[t].append(f"13B r≤4k vs 13: #{n} {RUNS[n][2]}")
    S = {}
    for est in EST + ("E5",):
        for t in (TARGETS if est != "E5" else (30,)):
            d = [te.pct(val(b, est, t), val(a, est, t)) for a, b in pairs[t if est != "E5" else 40]]
            absd = [abs(x) for x in d if x is not None]
            san = [te.pct(val(E[(sk, rep)], est, t), val(E[(sk, 1)], est, t)) for sk in STAGES for rep in (2, 6)]
            dfn = [x for x in san if x is not None]
            # a pair is undefined when a run never reaches the target (Stage 13 never reaches 50 %)
            ok = bool(dfn) and all(((x > 0) if est == "E5" else (x < 0)) for x in dfn)
            cvs = []
            for n in RUNS:  # spread across the two runs of each config (13 vs 13B<=4k), at 30/40 only
                if t < 50 or est == "E5":
                    v = [val(E[("13", n)], est, t), val(S4[n], est, t)]
                    if all(x is not None for x in v):
                        cvs.append(100 * st.stdev(v) / st.mean(v))
            S[(est, t)] = {"pairs": d, "defined": len(absd), "n": len(d), "median": st.median(absd) if absd else None,
                           "max": max(absd) if absd else None, "sanity": san, "ok": ok,
                           "cv": st.mean(cvs) if cvs else None}

    # what the CIFAR-10 rule would pick here (information only), at 40 % (8 pairs)
    elig = [e for e in EST if S[(e, 40)]["ok"] and S[(e, 40)]["defined"] == S[(e, 40)]["n"]]
    best = min(S[(e, 40)]["median"] for e in elig) if elig else None
    tied = [e for e in elig if S[(e, 40)]["median"] - best <= 1.0] if elig else []
    pick = min(tied, key=lambda e: S[(e, 40)]["cv"]) if tied else None
    fixed = json.loads(te.CHOICE.read_text())["choice"]

    L = ["# TTA estimation on CIFAR-100 (Stage 13 and 13B)", "",
         "Log-only. Extends `TTA_estimation_report.md` (CIFAR-10) to the 12 CIFAR-100 runs, at the user's request "
         "(2026-10-05). Generated by `sumo_open_loop/tta_estimation_c100.py`; per-run values in "
         "`tta_estimates_cifar100.csv`.", "",
         "## What was kept and what was changed", "",
         "- **Estimators and their parameters are unchanged** (E0 median-11, E1 sustained ×3, E2 MA-5 interpolated, "
         "E3 isotonic, E4 saturating fit with RMSE ≤ 1 pt, E5 AUC over 30 min).",
         "- **Targets: 30 / 40 / 50 %**, because 75–80 % is never reached on CIFAR-100 (plateau about 46–51 %). "
         "These are the same targets as the Stage 13 reports.",
         f"- **The headline estimator stays {fixed}**, fixed on CIFAR-10 before any arm comparison. It is not "
         "re-chosen here.",
         "- **Measurement.** One evaluation every 10 rounds on the 10,000-image test set. The binomial SE of one "
         "reading is 0.46 / 0.49 / 0.50 points at 30 / 40 / 50 %.", "",
         "## Replicate noise on CIFAR-100", "",
         "**Replicate pairs (true difference zero):**", "",
         "- Stage 13 #6 vs #2, and Stage 13B #6 vs #2.",
         "- The first 4,000 rounds of each 13B run vs its Stage 13 partner, for all six runs. Nothing in the config "
         "depends on the total number of rounds (no LR scheduler), so these are re-runs. The 13B series is cut at "
         "round 4,000 for these pairs.",
         "- **50 % is reached only after round 4,000**, so it has a single pair: 13B #6 vs #2.", "",
         "Δ % = (second − first) / first. E5 is in % of mean accuracy.", "",
         "| pair | " + " | ".join(f"{e}@40" for e in EST) + " | E5 |", "|---|" + "---|" * (len(EST) + 1)]
    for i, lab in enumerate(labels[40]):
        L.append(f"| {lab} | " + " | ".join(te.sg(S[(e, 40)]['pairs'][i]) for e in EST)
                 + f" | {te.sg(S[('E5', 30)]['pairs'][i], 2)} |")
    L += ["", "| estimator | target | pairs defined | median |Δ| | max |Δ| | spread between the two runs of a config (mean CV) | sanity: baseline+F − none (4 pairs) | sanity |",
          "|---|---|---|---|---|---|---|---|"]
    for est in EST + ("E5",):
        for t in (TARGETS if est != "E5" else (30,)):
            s = S[(est, t)]
            n_def = sum(x is not None for x in s["sanity"])
            L.append(f"| {est} | {t if est != 'E5' else 'AUC 30 min'} | {s['defined']}/{s['n']} | {te.ab(s['median'])} | "
                     f"{te.ab(s['max'])} | {te.ab(s['cv'])}% | " + " / ".join(te.sg(x, 0) for x in s["sanity"])
                     + f" | {('pass' if s['ok'] else 'FAIL') + ('' if all(x is not None for x in s['sanity']) else f' ({n_def}/4 defined)')} |")
    L += ["", f"**For information only:** applied at 40 % (the target with 8 pairs), the CIFAR-10 rule would pick "
          f"**{pick or 'none'}** on CIFAR-100. Eligible: {', '.join(elig) or 'none'}; within 1 point of the best "
          f"median: {', '.join(tied) or '—'}. The headline remains **{fixed}**.", ""]

    noise = {t: S[(fixed, t)]["max"] for t in TARGETS}
    n5 = S[("E5", 30)]["max"]
    L += ["## Comparisons with the headline estimator", "",
          f"A comparison is **beyond** noise when its |Δ| exceeds {fixed}'s max replicate |Δ| on CIFAR-100 at that "
          f"target ({', '.join(f'{noise[t]:.1f} % at {t} %' for t in TARGETS if noise[t] is not None)}; at 50 % from a "
          f"single pair). For E5, the bound is {n5:.2f} %. Δ % of TTA: negative = first run faster. E5 Δ in points: "
          "positive = first run better.", ""]
    comps = [("filter: baseline+F vs none", 2, 1), ("gpu+F vs baseline+F", 3, 2), ("gpu+F vs replicate baseline+F", 3, 6),
             ("observed+F vs baseline+F", 4, 2), ("cpu+F vs gpu+F", 5, 3)]
    L += [f"| comparison | stage | E0 30 / 40 / 50 | {fixed} 30 / 40 / 50 | {fixed} vs noise (30 / 40 / 50) | E5 Δ pts | E5 vs noise |",
          "|---|---|---|---|---|---|---|"]
    for lab, a, b in comps:
        for sk in STAGES:
            ea, eb = E[(sk, a)], E[(sk, b)]
            d0 = [te.pct(ea[("E0", t)]["t"], eb[("E0", t)]["t"]) for t in TARGETS]
            d = [te.pct(ea[(fixed, t)]["t"], eb[(fixed, t)]["t"]) for t in TARGETS]
            v = ["—" if x is None or noise[t] is None else ("**beyond**" if abs(x) > noise[t] else "inside")
                 for x, t in zip(d, TARGETS)]
            d5 = ea["E5"] - eb["E5"]
            v5 = "**beyond**" if abs(te.pct(ea["E5"], eb["E5"])) > n5 else "inside"
            L.append(f"| {lab} | {sk} | " + " / ".join(te.sg(x) for x in d0) + " | " + " / ".join(te.sg(x) for x in d)
                     + f" | {' / '.join(v)} | {d5:+.2f} | {v5} |")
    L += ["", f"## Rounds-to-target × seconds-per-round: gpu+F vs baseline+F ({fixed})", "",
          "| stage | vs | target | rounds gpu / base | Δ rounds | s/round gpu / base | Δ s/round | Δ TTA |",
          "|---|---|---|---|---|---|---|---|"]
    for sk in STAGES:
        for b in (2, 6):
            for t in TARGETS:
                x, y = E[(sk, 3)][(fixed, t)], E[(sk, b)][(fixed, t)]
                if x["t"] is None or y["t"] is None:
                    L.append(f"| {sk} | #{b} | {t} | not reached | | | | |")
                    continue
                L.append(f"| {sk} | #{b} | {t} | {x['rounds']:.0f} / {y['rounds']:.0f} | {te.sg(te.pct(x['rounds'], y['rounds']))} | "
                         f"{x['spr']:.3f} / {y['spr']:.3f} | {te.sg(te.pct(x['spr'], y['spr']))} | {te.sg(te.pct(x['t'], y['t']))} |")
    fits = [E[k]["fit"]["rmse_pts"] for k in E if E[k]["fit"]]
    L += ["", f"E4 fit: RMSE median {st.median(fits):.2f}, max {max(fits):.2f} points over {len(fits)} runs. "
          f"Failed (RMSE > 1 pt) in {sum(E[k]['fit_failed'] for k in E)} of {len(E)}.", ""]
    (OUT / "TTA_estimation_cifar100.md").write_text("\n".join(L) + "\n")
    with (OUT / "tta_estimates_cifar100.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "run", "label", "estimator", "target", "tta_s", "rounds", "seconds_per_round", "auc30_pts", "waste"])
        for (sk, n), e in E.items():
            for est in EST:
                for t in TARGETS:
                    x = e[(est, t)]
                    w.writerow([sk, n, RUNS[n][2], est, t, x["t"], x["rounds"], x["spr"], "", W[(sk, n)]])
            w.writerow([sk, n, RUNS[n][2], "E5", "auc_30min", "", "", "", round(e["E5"], 4), W[(sk, n)]])
    print(f"wrote {OUT / 'TTA_estimation_cifar100.md'}; rule-on-CIFAR-100 pick (info): {pick}; headline {fixed}")


if __name__ == "__main__":
    main()
