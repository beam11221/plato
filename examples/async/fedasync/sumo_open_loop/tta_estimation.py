"""
TTA estimation: less noisy estimators of time-to-accuracy.  Log-only, no
training.  Spec: gpu_tiers_dataset/prompts/TTA_smoothing_analysis.md.

    python3 tta_estimation.py score   # Step 0 + Step 2: scores the estimators on
                                      # replicate data only and FIXES the choice in
                                      # tta_evidence/tta_estimator_choice.json
    python3 tta_estimation.py apply   # Step 3 with the fixed choice; writes
                                      # TTA_estimation_report.md + tta_estimates.csv

`apply` refuses to run without the choice file and never rewrites it; `score`
refuses to write a different choice over an existing one.  Re-run `apply` when
Stage 11 Phase 2 has finished (it picks up every finished run).

Every run is read through analyze_stage7.parse (read-only).  The per-evaluation
series is (round, reconstructed sim-time, accuracy) at rounds that are multiples
of eval_interval = 10 -- exactly the rows the existing analysis uses.

Estimators (parameters fixed by the spec, not tuned):
  E0  current method (analyze_stage7.curve_stats): centred rolling MEDIAN over 11
      evaluations, first evaluation whose smoothed value >= target, no
      interpolation
  E1  first evaluation >= target whose next 3 evaluations are also >= target
  E2  centred moving average over 5 evaluations (truncated at the ends), first
      crossing, linearly interpolated in sim-time between the two evaluations
      that bracket it
  E3  isotonic (non-decreasing) regression of accuracy on sim-time (PAVA), first
      crossing, interpolated as E2
  E4  least-squares fit acc(t) = a - b exp(-c t) on the whole run, inverted at the
      target; "fit failed" if RMSE > 1 point or target >= a
  E5  normalised area under the accuracy curve over sim-time [0, 1,800 s]
      (accuracy held from one evaluation to the next, 0 before the first)
Rounds-to-target is interpolated the same way as the time; seconds-per-round =
TTA / rounds-to-target.
"""

from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_stage7 as a7  # noqa: E402
import analyze_stage9 as a9  # noqa: E402

OUT = HERE.parent / "gpu_tiers_dataset" / "tta_evidence"
CACHE = OUT / "series_cache.json"
CHOICE = OUT / "tta_estimator_choice.json"
SEEDS = (4126, 4127, 4128)
REGIMES = ("open", "closed")
TARGETS = (75, 78, 80)
EST = ("E0", "E1", "E2", "E3", "E4")
T_REF = 1800.0
SUSTAIN, MA_WINDOW, MEDIAN_WINDOW = 3, 5, 11
FIT_RMSE_MAX = 1.0  # points

# --------------------------------------------------------------------------
# run inventory (Stages 7b-10 + Stage 11)
# --------------------------------------------------------------------------

def pfx(kind, seed):
    return {"a10": "stage7b" if seed == 4126 else f"stage8_s{seed}", "s9": f"stage9_s{seed}",
            "nf": f"stage10nf_s{seed}", "f1": f"stage10f1_s{seed}", "f10": f"stage10f10_s{seed}"}[kind]


KIND_ARMS = {"a10": ("baseline", "gpu", "cpu", "observed"), "s9": ("gpu", "cpu", "observed"),
             "nf": ("baseline",), "f1": ("baseline", "gpu", "cpu", "observed"), "f10": ("gpu", "cpu", "observed")}
KIND_LABEL = {"a10": "Stage 7b/8: no filter, α = 10", "s9": "Stage 9: no filter, α = 1",
              "nf": "Stage 10 re-run: baseline / none", "f1": "Stage 10 Wave A: filter, α = 1",
              "f10": "Stage 10 Wave B: filter, α = 10"}


def inventory():
    runs = []
    for kind, arms in KIND_ARMS.items():
        for s in SEEDS:
            for arm in arms:
                for reg in REGIMES:
                    runs.append({"key": f"{kind}/{s}/{arm}_{reg}", "kind": kind, "seed": s, "arm": arm,
                                 "regime": reg, "prefix": pfx(kind, s), "cell": f"{arm}_{reg}", "p": None})
    runs.append({"key": "s9inv/4126/baseline_open", "kind": "s9inv", "seed": 4126, "arm": "baseline",
                 "regime": "open", "prefix": "stage9_s4126", "cell": "baseline_open", "p": None})
    for s in SEEDS:
        for p in (0.25, 0.5, 0.75, 1.0):
            for arm in ("gpu", "baseline") + (("observed", "cpu") if p < 1 else ()):
                runs.append({"key": f"s11/{s}/{arm}_p{p}", "kind": "s11", "seed": s, "arm": arm, "regime": "ret",
                             "prefix": f"stage11f_p{round(p * 100):03d}_s{s}", "cell": f"{arm}_ret", "p": p})
            if p < 1:
                runs.append({"key": f"s11/{s}/none_p{p}", "kind": "s11", "seed": s, "arm": "none", "regime": "ret",
                             "prefix": f"stage11nf_p{round(p * 100):03d}_s{s}", "cell": "baseline_ret", "p": p})
    return [r for r in runs if a9.available(r["prefix"], r["cell"])]


def load_series(runs):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    changed = False
    for r in runs:
        tag = f"{r['prefix']}/{r['cell']}"
        done = (a7.LOGS if hasattr(a7, "LOGS") else HERE / "logs") / f"{r['prefix']}_{r['cell']}.done"
        stamp = done.stat().st_mtime if done.exists() else 0
        if tag not in cache or cache[tag]["stamp"] != stamp:
            a7.PREFIX = r["prefix"]
            run = a7.parse(r["cell"])
            rows = [(c["round"], c["sim_time_reconstructed"], c["accuracy"])
                    for c in run.curve if c["round"] % a7.EVAL_INTERVAL == 0]
            cs = a7.curve_stats(run)
            cache[tag] = {"stamp": stamp, "series": rows, "waste": a7.waste(run)["overall"]["wasted_frac"],
                          "plateau": cs["plateau"],
                          "e0_check": {str(t): cs.get(f"tta_h_{t}") for t in TARGETS},
                          "T4000": a7.clock_at(run, 4000)}
            changed = True
            print(f"  parsed {tag}", flush=True)
        r.update(cache[tag])
    if changed:
        OUT.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache))
    return runs


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

def first_cross(ts, rs, ys, tgt, interpolate):
    for i, y in enumerate(ys):
        if y >= tgt:
            if interpolate and i > 0 and ys[i - 1] < tgt:
                f = (tgt - ys[i - 1]) / (ys[i] - ys[i - 1])
                return ts[i - 1] + f * (ts[i] - ts[i - 1]), rs[i - 1] + f * (rs[i] - rs[i - 1])
            return ts[i], float(rs[i])
    return None, None


def moving_average(ys, w):
    h = w // 2
    return [st.mean(ys[max(0, i - h): i + h + 1]) for i in range(len(ys))]


def isotonic(ys):
    """Pool-adjacent-violators, equal weights, non-decreasing."""
    blocks = []  # [sum, count]
    for y in ys:
        blocks.append([y, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
    out = []
    for s, n in blocks:
        out += [s / n] * n
    return out


def _sat(t, a, b, c):
    return a - b * np.exp(-c * t)


def saturating_fit(ts, ys):
    t = np.asarray(ts) / 3600.0
    y = np.asarray(ys)
    tail = float(np.mean(y[-max(1, len(y) // 10):]))
    half = y[0] + (tail - y[0]) / 2
    t_half = t[int(np.argmax(y >= half))] or t[1]
    try:
        (a, b, c), _ = curve_fit(_sat, t, y, p0=(tail, tail - y[0], math.log(2) / t_half),
                                 bounds=([0, 0, 1e-6], [1, 2, 1e4]), maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    rmse = 100 * float(np.sqrt(np.mean((_sat(t, a, b, c) - y) ** 2)))
    return {"a": float(a), "b": float(b), "c_per_h": float(c), "rmse_pts": rmse}


def auc(ts, ys, t_ref=T_REF):
    area = 0.0
    for i, (t, y) in enumerate(zip(ts, ys)):
        if t >= t_ref:
            break
        t_next = ts[i + 1] if i + 1 < len(ts) else t_ref
        area += y * (min(t_next, t_ref) - t)
    return 100 * area / t_ref


def estimate(series):
    """{(estimator, target): {"t", "rounds", "spr", ...}} plus E5 and the fit."""
    rs = [r for r, _, _ in series]
    ts = [t for _, t, _ in series]
    ys = [a for _, _, a in series]
    med = a7.rolling_median(ys, MEDIAN_WINDOW)
    ma = moving_average(ys, MA_WINDOW)
    order = sorted(range(len(ts)), key=lambda i: ts[i])
    ts_s, rs_s = [ts[i] for i in order], [rs[i] for i in order]
    iso = isotonic([ys[i] for i in order])
    fit = saturating_fit(ts_s, [ys[i] for i in order])
    out = {}
    for tgt in TARGETS:
        g = tgt / 100
        res = {"E0": first_cross(ts, rs, med, g, False), "E2": first_cross(ts, rs, ma, g, True),
               "E3": first_cross(ts_s, rs_s, iso, g, True)}
        e1 = next((i for i in range(len(ys) - SUSTAIN) if all(y >= g for y in ys[i: i + SUSTAIN + 1])), None)
        res["E1"] = (ts[e1], float(rs[e1])) if e1 is not None else (None, None)
        if fit and fit["rmse_pts"] <= FIT_RMSE_MAX and g < fit["a"]:
            t4 = -math.log((fit["a"] - g) / fit["b"]) / fit["c_per_h"] * 3600 if fit["b"] > 0 else None
            t4 = max(t4, 0.0) if t4 is not None else None
            res["E4"] = (t4, float(np.interp(t4, ts_s, rs_s))) if t4 is not None else (None, None)
        else:
            res["E4"] = (None, None)
        for e, (t, r) in res.items():
            out[(e, tgt)] = {"t": t, "rounds": r, "spr": (t / r) if t is not None and r else None}
    out["E5"] = auc(ts_s, [ys[i] for i in order])
    out["fit"] = fit
    out["fit_failed"] = fit is None or fit["rmse_pts"] > FIT_RMSE_MAX
    return out


def pct(a, b):
    return 100.0 * (a - b) / b if a is not None and b not in (None, 0) else None


# --------------------------------------------------------------------------
# Step 2: scores on replicate data only
# --------------------------------------------------------------------------

def replicate_pairs():
    pairs = [(f"a10/{s}/baseline_{r}", f"nf/{s}/baseline_{r}", f"baseline/none {r} s{s}: 7b/8 → Stage 10 re-run")
             for r in REGIMES for s in SEEDS]
    pairs.append(("a10/4126/baseline_open", "s9inv/4126/baseline_open", "Stage 9 invariance: 7b → Stage 9 baseline_open s4126"))
    return pairs


def value(E, key, est, tgt):
    e = E.get(key)
    if e is None:
        return None
    return e["E5"] if est == "E5" else e[(est, tgt)]["t"]


def scores(E):
    out = {}
    pairs = replicate_pairs()
    for est in EST + ("E5",):
        for tgt in (TARGETS if est != "E5" else (None,)):
            d = [pct(value(E, b, est, tgt), value(E, a, est, tgt)) for a, b, _ in pairs]
            absd = [abs(x) for x in d if x is not None]
            cvs = []
            for kind, arms in KIND_ARMS.items():
                for arm in arms:
                    for reg in REGIMES:
                        v = [value(E, f"{kind}/{s}/{arm}_{reg}", est, tgt) for s in SEEDS]
                        if all(x is not None for x in v):
                            cvs.append(100 * st.stdev(v) / st.mean(v))
            san = [pct(value(E, f"f1/{s}/baseline_{r}", est, tgt), value(E, f"nf/{s}/baseline_{r}", est, tgt))
                   for s in SEEDS for r in REGIMES]
            want_pos = est == "E5"
            ok = all(x is not None and ((x > 0) if want_pos else (x < 0)) for x in san)
            out[f"{est}@{tgt}" if tgt else est] = {
                "pairs": d, "pairs_defined": len(absd), "median_abs": st.median(absd) if absd else None,
                "max_abs": max(absd) if absd else None, "seed_cv_mean": st.mean(cvs) if cvs else None,
                "seed_cv_cells": len(cvs), "sanity": san, "sanity_pass": ok}
    return out


def choose(S):
    """The rule written in the spec (Step 2), applied at the 80 % target.
    Eligible: passes sanity (3) AND has a value on all 7 replicate pairs (an
    estimator undefined on a pair cannot be scored on it)."""
    elig = [e for e in EST if S[f"{e}@80"]["sanity_pass"] and S[f"{e}@80"]["pairs_defined"] == 7]
    if not elig:
        return None, elig, []
    best = min(S[f"{e}@80"]["median_abs"] for e in elig)
    tied = [e for e in elig if S[f"{e}@80"]["median_abs"] - best <= 1.0]
    pick = min(tied, key=lambda e: S[f"{e}@80"]["seed_cv_mean"])
    return pick, elig, tied


# --------------------------------------------------------------------------
# report pieces
# --------------------------------------------------------------------------

def h(t):
    return "—" if t is None else f"{t / 3600:.3f}"


def ab(v):
    return "—" if v is None else f"{v:.1f}"


def sg(v, nd=1):
    return "—" if v is None else f"{v:+.{nd}f}"


def step0(E, runs):
    rep = next(r for r in runs if r["key"] == "f1/4126/gpu_open")
    ser = rep["series"]
    e0 = E[rep["key"]][("E0", 80)]
    n_before = sum(1 for r, _, _ in ser if r <= (e0["rounds"] or 0))
    win = [(r, t, a) for r, t, a in ser if 600 <= r <= 1600]
    above = [a >= 0.80 for _, _, a in win]
    ups = sum(1 for x, y in zip(above, above[1:]) if not x and y)
    downs = sum(1 for x, y in zip(above, above[1:]) if x and not y)
    first_raw = next(((r, t) for r, t, a in ser if a >= 0.8), None)
    se = 100 * math.sqrt(0.8 * 0.2 / 10000)
    check = max(abs(E[r["key"]][("E0", t)]["t"] / 3600 - r["e0_check"][str(t)])
                for r in runs for t in TARGETS if r["e0_check"][str(t)] is not None and E[r["key"]][("E0", t)]["t"] is not None)
    L = ["## Step 0: the measurement", "",
         "Representative run: Stage 10 `gpu + F` open, s4126 (`stage10f1_s4126 / gpu_open`).", "",
         f"1. **Evaluation schedule.** The server evaluates the global model every **10 rounds** (`eval_interval = 10`); "
         f"the results CSV repeats the last reading on the rounds in between, and only the multiples of 10 are used. "
         f"A 4,000-round run has 400 evaluations. In the representative run E0 places the 80 % crossing at round "
         f"{e0['rounds']:.0f}, so **{n_before} evaluations** lie at or before it.",
         f"2. **Test set.** The full CIFAR-10 test set, **10,000 images** (no `testset_size` in the config, so "
         f"`plato/servers/fedavg.py` uses the whole set). The binomial standard error of one reading at 80 % is "
         f"√(0.8·0.2/10,000) = **{se:.2f} points**. Readings are also correlated with the model's real "
         f"round-to-round fluctuation, so the observed scatter is larger (below).",
         f"3. **Raw series, rounds 600–1,600** ({len(win)} evaluations): accuracy crosses 80 % **upward {ups} times "
         f"and downward {downs} times**. The first raw reading ≥ 80 % is at round {first_raw[0]} "
         f"(t = {first_raw[1] / 3600:.3f} h); E0 says round {e0['rounds']:.0f} (t = {h(e0['t'])} h).", "",
         "| round | sim-time (h) | acc | | round | sim-time (h) | acc | | round | sim-time (h) | acc | | round | sim-time (h) | acc |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = [f"{r} | {t / 3600:.3f} | {'**' if a >= 0.8 else ''}{100 * a:.2f}{'**' if a >= 0.8 else ''}" for r, t, a in win]
    q = math.ceil(len(rows) / 4)
    for i in range(q):
        cols = [rows[i + k * q] if i + k * q < len(rows) else " | | " for k in range(4)]
        L.append("| " + " | | ".join(cols) + " |")
    L += ["", "Bold = at or above 80 %.", "",
          "4. **How TTA is computed now** (`sumo_open_loop/analyze_stage7.py`, `curve_stats`, lines 380–395; "
          "`rolling_median` 375): the evaluations are smoothed with a **centred rolling median over 11 evaluations** "
          "(`SMOOTH_WINDOW = 11`, truncated at the ends), and TTA is the reconstructed sim-time of the **first "
          "evaluation whose smoothed value ≥ target**. No interpolation. So TTA is already smoothed, and it moves in "
          "10-round steps. That method is E0 here; the reimplementation matches the existing `tta_h_*` on every run "
          f"(max difference {check:.1e} h).", ""]
    return L


def step2_section(S, choice):
    pairs = replicate_pairs()
    L = ["## Step 2: scores on data where the true difference is zero or known", "",
         "Written before any arm comparison was computed. Δ % = (second − first) / first.", "",
         "### 1. Replicate noise (7 identical-config pairs)", "",
         "| pair | " + " | ".join(f"{e}@80" for e in EST) + " | E5 |", "|---|" + "---|" * (len(EST) + 1)]
    for i, (_, _, label) in enumerate(pairs):
        L.append(f"| {label} | " + " | ".join(sg(S[f'{e}@80']['pairs'][i]) for e in EST) + f" | {sg(S['E5']['pairs'][i], 2)} |")
    L += ["", "### Summary per estimator and target", "",
          "| estimator | target | median |Δ| | max |Δ| | pairs defined | seed CV (mean, cells) | sanity: baseline+F − none, 6 pairs | sanity |",
          "|---|---|---|---|---|---|---|---|"]
    for e in EST + ("E5",):
        for t in (TARGETS if e != "E5" else (None,)):
            s = S[f"{e}@{t}" if t else e]
            cv = "—" if s["seed_cv_mean"] is None else f"{s['seed_cv_mean']:.1f}"
            L.append(f"| {e} | {t or 'AUC 30 min'} | {ab(s['median_abs'])} | {ab(s['max_abs'])} | {s['pairs_defined']}/7 | "
                     f"{cv}% ({s['seed_cv_cells']}) | "
                     + " / ".join(sg(x, 0) for x in s["sanity"]) + f" | {'pass' if s['sanity_pass'] else 'FAIL'} |")
    pick, elig, tied = choice["choice"], choice["eligible"], choice["tied"]
    L += ["", "### Choice", "",
          "Rule (spec): among estimators passing the sanity check, the lowest median replicate noise at 80 %; "
          "ties within 1 point broken by seed spread. E5 is reported next to it, not chosen. An estimator undefined "
          "on any of the 7 pairs (no crossing, or fit failed) cannot be scored and is not eligible.", "",
          f"- eligible: {', '.join(elig) or 'none'}",
          f"- within 1 point of the lowest median: {', '.join(tied) or '—'}",
          f"- **headline estimator: {pick}**, fixed {choice['fixed_at']} (`tta_evidence/tta_estimator_choice.json`), "
          "before any arm comparison.", ""]
    return L


# --------------------------------------------------------------------------
# Step 3
# --------------------------------------------------------------------------

COMPARISONS = []
for reg in REGIMES:
    for arm in ("gpu", "observed", "cpu"):
        COMPARISONS.append((f"no filter α=10: {arm} vs baseline, {reg}", f"a10/{{s}}/{arm}_{reg}", f"a10/{{s}}/baseline_{reg}"))
    COMPARISONS.append((f"no filter α=10: gpu vs observed, {reg}", f"a10/{{s}}/gpu_{reg}", f"a10/{{s}}/observed_{reg}"))
for reg in REGIMES:
    for arm in ("gpu", "observed", "cpu"):
        COMPARISONS.append((f"no filter α=1: {arm} vs baseline, {reg}", f"s9/{{s}}/{arm}_{reg}", f"a10/{{s}}/baseline_{reg}"))
    COMPARISONS.append((f"no filter α=1: gpu vs observed, {reg}", f"s9/{{s}}/gpu_{reg}", f"s9/{{s}}/observed_{reg}"))
    for arm in ("gpu", "observed", "cpu"):
        COMPARISONS.append((f"no filter: {arm} α=1 vs α=10, {reg}", f"s9/{{s}}/{arm}_{reg}", f"a10/{{s}}/{arm}_{reg}"))
for reg in REGIMES:
    COMPARISONS.append((f"Stage 10 {'A' if reg == 'open' else 'B'}: gpu+F vs baseline+F, {reg}", f"f1/{{s}}/gpu_{reg}", f"f1/{{s}}/baseline_{reg}"))
    COMPARISONS.append((f"Stage 10 C: gpu+F vs observed+F, {reg}", f"f1/{{s}}/gpu_{reg}", f"f1/{{s}}/observed_{reg}"))
    for arm in ("baseline", "gpu", "observed", "cpu"):
        COMPARISONS.append((f"Stage 10 D: {arm}+F vs baseline/none, {reg}", f"f1/{{s}}/{arm}_{reg}", f"nf/{{s}}/baseline_{reg}"))
    COMPARISONS.append((f"Stage 10 G: cpu+F vs gpu+F, {reg}", f"f1/{{s}}/cpu_{reg}", f"f1/{{s}}/gpu_{reg}"))
    for arm in ("gpu", "observed", "cpu"):
        COMPARISONS.append((f"Stage 10 I: {arm}+F α=1 vs α=10, {reg}", f"f1/{{s}}/{arm}_{reg}", f"f10/{{s}}/{arm}_{reg}"))
    COMPARISONS.append((f"Stage 10 F10: gpu+F α=10 vs baseline+F, {reg}", f"f10/{{s}}/gpu_{reg}", f"f1/{{s}}/baseline_{reg}"))


def step3(E, runs, S, pick):
    noise = {t: S[f"{pick}@{t}"]["max_abs"] for t in TARGETS}
    noise5 = S["E5"]["max_abs"]
    noise0 = {t: S[f"E0@{t}"]["max_abs"] for t in TARGETS}
    W = {r["key"]: r["waste"] for r in runs}
    L = ["## Step 3: the chosen estimator everywhere", "",
         f"Headline: **{pick}**. Its replicate noise (max |Δ| over the 7 pairs) is "
         + ", ".join(f"{noise[t]:.1f} % at {t} %" for t in TARGETS)
         + f"; E5's is {noise5:.2f} %; E0's at 80 % was {noise0[80]:.1f} %. A comparison is **beyond** noise when its "
         "|Δ| exceeds that bound, **inside** otherwise. TTA in simulated hours; E5 in accuracy points "
         "(mean accuracy over the first 30 simulated minutes, higher = better). Waste is unchanged from the existing analysis.", "",
         "> The project doc `tta80-waste-by-stage` could not be opened from this session (it is not in the workspace or "
         "the artifact list). The tables below regenerate its scope as the spec defines it: Stages 7b–10, both regimes, "
         "both α, with and without the filter, per seed. The comparisons are the paired comparisons of the Stage 7b–10 "
         "reports. Check them against the doc's list.", "",
         "### 3.1 Tables by stage", ""]
    for kind, label in KIND_LABEL.items():
        L += [f"#### {label}", "",
              f"| cell | seed | E0 75 | E0 78 | E0 80 | {pick} 75 | {pick} 78 | {pick} 80 | E5 | waste |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for arm in KIND_ARMS[kind]:
            for reg in REGIMES:
                for s in SEEDS:
                    k = f"{kind}/{s}/{arm}_{reg}"
                    if k not in E:
                        continue
                    e = E[k]
                    L.append(f"| {arm}_{reg} | {s} | " + " | ".join(h(e[('E0', t)]['t']) for t in TARGETS) + " | "
                             + " | ".join(h(e[(pick, t)]['t']) for t in TARGETS) + f" | {e['E5']:.2f} | {100 * W[k]:.2f}% |")
        L.append("")

    L += ["### 3.2 Paired comparisons (Δ % of TTA, negative = first arm faster; E5 Δ in points, positive = first arm better)", "",
          f"| comparison | seed | E0 80 | {pick} 75 | {pick} 78 | {pick} 80 | {pick} 80 vs noise | E5 Δ pts | E5 vs noise |",
          "|---|---|---|---|---|---|---|---|---|"]
    tally = {"beyond": 0, "inside": 0, "beyond_E0": 0}
    for label, a, b in COMPARISONS:
        for s in SEEDS:
            ka, kb = a.format(s=s), b.format(s=s)
            if ka not in E or kb not in E:
                continue
            d = {t: pct(E[ka][(pick, t)]["t"], E[kb][(pick, t)]["t"]) for t in TARGETS}
            d0 = pct(E[ka][("E0", 80)]["t"], E[kb][("E0", 80)]["t"])
            d5 = E[ka]["E5"] - E[kb]["E5"]
            d5p = pct(E[ka]["E5"], E[kb]["E5"])
            v = "—" if d[80] is None else ("**beyond**" if abs(d[80]) > noise[80] else "inside")
            v5 = "**beyond**" if abs(d5p) > noise5 else "inside"
            if d[80] is not None:
                tally["beyond" if abs(d[80]) > noise[80] else "inside"] += 1
            if d0 is not None and abs(d0) > noise0[80]:
                tally["beyond_E0"] += 1
            L.append(f"| {label} | {s} | {sg(d0)} | {sg(d[75])} | {sg(d[78])} | {sg(d[80])} | {v} | {d5:+.2f} | {v5} |")
    L += ["", f"At 80 %: {tally['beyond']} comparisons beyond {pick}'s noise, {tally['inside']} inside "
          f"(E0 with its own noise bound: {tally['beyond_E0']} beyond).", ""]

    # Stage 11 curve
    L += ["### 3.3 Stage 11 curve over p_return", "",
          "`p = 0` is Stage 10 open, `p = 1` is Stage 10 closed (the anchor runs are shown separately). Seeds whose "
          "Stage 11 runs are not finished show —. Re-run `apply` when Phase 2 has finished.", ""]
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)

    def k11(arm, s, p):
        a = "baseline" if arm == "none" else arm
        if p == 0.0:
            return f"{'nf' if arm == 'none' else 'f1'}/{s}/{a}_open"
        if p == 1.0:
            return f"{'nf' if arm == 'none' else 'f1'}/{s}/{a}_closed"
        return f"s11/{s}/{arm}_p{p}"

    for s in SEEDS:
        if not any(r["kind"] == "s11" and r["seed"] == s for r in runs):
            continue
        L += [f"#### seed {s}", "",
              f"| arm | metric | p = 0 | 0.25 | 0.5 | 0.75 | 1.0 (Stage 10 closed) | 1.0 anchor |", "|---|---|---|---|---|---|---|---|"]
        for arm in ("gpu", "baseline", "observed", "cpu", "none"):
            for metric in (f"{pick} 80 (h)", "E0 80 (h)", "E5 (pts)", "waste"):
                cells = []
                for p in grid + ("anchor",):
                    k = f"s11/{s}/{arm}_p1.0" if p == "anchor" else k11(arm, s, p)
                    if k not in E:
                        cells.append("—")
                    elif metric.startswith(pick):
                        cells.append(h(E[k][(pick, 80)]["t"]))
                    elif metric.startswith("E0"):
                        cells.append(h(E[k][("E0", 80)]["t"]))
                    elif metric.startswith("E5"):
                        cells.append(f"{E[k]['E5']:.2f}")
                    else:
                        cells.append(f"{100 * W[k]:.2f}%")
                L.append(f"| {arm}{'' if arm == 'none' else '+F'} | {metric} | " + " | ".join(cells) + " |")
        gapA = []
        for p in grid:
            ka, kb = k11("gpu", s, p), k11("baseline", s, p)
            gapA.append(pct(E[ka][(pick, 80)]["t"], E[kb][(pick, 80)]["t"]) if ka in E and kb in E else None)
        gapA0 = []
        for p in grid:
            ka, kb = k11("gpu", s, p), k11("baseline", s, p)
            gapA0.append(pct(E[ka][("E0", 80)]["t"], E[kb][("E0", 80)]["t"]) if ka in E and kb in E else None)
        L += ["", f"Gap A (gpu+F − baseline+F, TTA@80, %): {pick} " + " / ".join(sg(x) for x in gapA)
              + f"; E0 " + " / ".join(sg(x) for x in gapA0) + f". {pick}'s noise bound: ±{noise[80]:.1f} %.", ""]

    # rounds decomposition
    L += ["### 3.4 Rounds-to-target and seconds-per-round: gpu+F vs baseline+F", "",
          "TTA = rounds-to-target × seconds-per-round. Δ % is gpu+F relative to baseline+F.", "",
          "| regime | seed | estimator | target | rounds gpu / base | Δ rounds | s/round gpu / base | Δ s/round | Δ TTA |",
          "|---|---|---|---|---|---|---|---|---|"]
    for reg in REGIMES:
        for s in SEEDS:
            g, b = E.get(f"f1/{s}/gpu_{reg}"), E.get(f"f1/{s}/baseline_{reg}")
            if not g or not b:
                continue
            for est in dict.fromkeys(("E0", pick)):
                for t in TARGETS:
                    x, y = g[(est, t)], b[(est, t)]
                    if x["t"] is None or y["t"] is None:
                        L.append(f"| {reg} | {s} | {est} | {t} | — | | | | |")
                        continue
                    L.append(f"| {reg} | {s} | {est} | {t} | {x['rounds']:.0f} / {y['rounds']:.0f} | {sg(pct(x['rounds'], y['rounds']))} | "
                             f"{x['spr']:.3f} / {y['spr']:.3f} | {sg(pct(x['spr'], y['spr']))} | {sg(pct(x['t'], y['t']))} |")
    L.append("")
    return L


def fits_section(E, runs):
    failed = [r["key"] for r in runs if E[r["key"]]["fit_failed"]]
    above = [(r["key"], t) for r in runs for t in TARGETS
             if not E[r["key"]]["fit_failed"] and E[r["key"]][("E4", t)]["t"] is None]
    rm = [E[r["key"]]["fit"]["rmse_pts"] for r in runs if E[r["key"]]["fit"]]
    return ["### E4 fit quality", "",
            f"RMSE over {len(rm)} fitted runs: median {st.median(rm):.2f}, max {max(rm):.2f} points. "
            f"**Fit failed (RMSE > 1 point): {len(failed)} of {len(runs)} runs**"
            + (f" ({', '.join(failed[:12])}{', …' if len(failed) > 12 else ''})" if failed else "")
            + f". Target above the fitted asymptote a: {len(above)} run × target cases.", ""]


def write_csv(E, runs):
    with (OUT / "tta_estimates.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_key", "stage_kind", "prefix", "cell", "seed", "arm", "regime", "p_return", "estimator", "target",
                    "tta_s", "tta_h", "rounds_to_target", "seconds_per_round", "auc30_pts", "fit_rmse_pts", "fit_failed",
                    "waste", "plateau", "T4000"])
        for r in runs:
            e = E[r["key"]]
            base = [r["key"], r["kind"], r["prefix"], r["cell"], r["seed"], r["arm"], r["regime"], r["p"]]
            tail = [round(e["fit"]["rmse_pts"], 4) if e["fit"] else "", int(e["fit_failed"]),
                    round(r["waste"], 6), r["plateau"], r["T4000"]]
            for est in EST:
                for t in TARGETS:
                    x = e[(est, t)]
                    w.writerow(base + [est, t, x["t"], x["t"] / 3600 if x["t"] is not None else None,
                                       x["rounds"], x["spr"], "", *tail])
            w.writerow(base + ["E5", "auc_30min", "", "", "", "", round(e["E5"], 4), *tail])


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "apply"
    runs = load_series(inventory())
    E = {r["key"]: estimate(r["series"]) for r in runs}
    S = scores(E)
    pick, elig, tied = choose(S)
    if CHOICE.exists():
        choice = json.loads(CHOICE.read_text())
        if mode == "score" and choice["choice"] != pick:
            raise SystemExit(f"recomputed choice {pick} differs from the fixed {choice['choice']}; not overwriting")
    elif mode == "score":
        from datetime import datetime, timezone
        choice = {"choice": pick, "eligible": elig, "tied": tied,
                  "fixed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                  "scores_at_80": {e: S[f"{e}@80"] for e in EST}, "E5": S["E5"]}
        OUT.mkdir(parents=True, exist_ok=True)
        CHOICE.write_text(json.dumps(choice, indent=2))
    else:
        raise SystemExit("no fixed choice: run `score` first")
    if choice["choice"] is None:
        raise SystemExit("no estimator is eligible")
    L = ["# TTA estimation: reducing the noise on time-to-accuracy", "",
         "Log-only; no training run. Spec: `prompts/TTA_smoothing_analysis.md`. Generated by "
         "`sumo_open_loop/tta_estimation.py`; per-run values in `tta_estimates.csv`.", "",
         f"Runs read: {len(runs)} (Stages 7b–10: {sum(r['kind'] != 's11' for r in runs)}; "
         f"Stage 11 finished so far: {sum(r['kind'] == 's11' for r in runs)}).", ""]
    L += step0(E, runs) + step2_section(S, choice) + fits_section(E, runs)
    if mode == "apply":
        L += step3(E, runs, S, choice["choice"])
        write_csv(E, runs)
    (OUT / "TTA_estimation_report.md").write_text("\n".join(L) + "\n")
    print(f"{mode}: {len(runs)} runs; choice {choice['choice']} (eligible {choice['eligible']}); wrote {OUT}")


if __name__ == "__main__":
    main()
