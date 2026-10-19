"""
Accuracy dips: where do they come from?  Log-only; no training run.
Spec: gpu_tiers_dataset/prompts/DIP_analysis.md.

    python3 dip_analysis.py

Runs: Stage 13 (CIFAR-100, s4126, all six) and Stage 10 open filtered alpha = 1
baseline+F / gpu+F, s4126 / 4127 / 4128 (CIFAR-10).  Run folders are only read.

Dip definition (fixed by the spec before any result):
  residual = accuracy - centred 21-evaluation rolling median (truncated at the ends)
  dip      = an evaluation at round >= 500 with residual < -5 points (CIFAR-100)
             or < -3 points (CIFAR-10)
  recovery = evaluations until the residual is back above -1 point

Per merge, from the server log: the applied mixing weight w and staleness tau
("FedAsync: step k - client #slot, effective_round=r, staleness=tau, mixing=w"),
joined to the dispatch it merged (tier, partition) exactly as
analyze_stage7.parse joins merges to dispatches.  Partition label skew is the
Dirichlet class distribution of each partition, rebuilt with Plato's sampler
for the run's data seed and checked against the partition fingerprints the
clients logged.

An evaluation at round r reflects the merges with effective_round in (r - 10, r]:
its "interval".
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics as st
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_stage7 as a7  # noqa: E402

OUT = HERE.parent / "gpu_tiers_dataset" / "dip_evidence"
REPO = HERE.parents[3]
WINDOW, MIN_ROUND, RECOVER = 21, 500, -1.0
THRESH = {"CIFAR100": -5.0, "CIFAR10": -3.0}
RUNS = [("CIFAR100", "stage13nf_s4126", "baseline_open", "baseline / none", 4126),
        ("CIFAR100", "stage13f_s4126", "baseline_open", "baseline + F", 4126),
        ("CIFAR100", "stage13f_s4126", "gpu_open", "gpu + F", 4126),
        ("CIFAR100", "stage13f_s4126", "observed_open", "observed + F", 4126),
        ("CIFAR100", "stage13f_s4126", "cpu_open", "cpu + F", 4126),
        ("CIFAR100", "stage13rep_s4126", "baseline_open", "baseline + F (replicate)", 4126)]
RUNS += [("CIFAR10", f"stage10f1_s{s}", f"{a}_open", f"{a} + F", s)
         for s in (4126, 4127, 4128) for a in ("baseline", "gpu")]
MERGE = re.compile(r"FedAsync: step \d+ - client #(\d+), effective_round=(\d+), staleness=(\d+), mixing=([\d.]+)")
PART = re.compile(r"\[Stage7\] dispatch round=(\d+) slot=(\d+) .*? partition=(\d+)")
FPR = re.compile(r"partition=(\d+) sampler_id=\d+ fingerprint=(\w+)")
QUANT = ("w_max", "w_sum", "tau_max", "tau_mean", "entropy_min", "maxshare_max",
         "n_xavier", "n_orin", "n_thor", "n_merges")

# --------------------------------------------------------------------------
# partitions: label distribution per (config) -- in a fresh process, from a
# COPY of the config whose base_path is a temp dir (loading a config makes
# Plato write a server log under base_path; never into a run folder)
# --------------------------------------------------------------------------

PART_SCRIPT = r"""
import sys, os, json, numpy as np
from plato.config import Config
Config()
from plato.datasources import registry as D
from plato.samplers import registry as S
sys.path.insert(0, os.environ["DIP_HERE"])
from occupant_identity import partition_fingerprint
ds = D.get(); t = np.asarray(ds.targets()); k = len(ds.classes())
out = {}
for p in range(20):
    s = S.get(ds, client_id=p + 1)
    w = np.asarray(s.sample_weights)
    per = np.array([w[t == c][0] for c in range(k)])
    q = per / per.sum()
    nz = q[q > 0]
    out[p] = {"entropy_bits": float(-(nz * np.log2(nz)).sum()), "max_share": float(q.max()),
              "fingerprint": partition_fingerprint(s, ds)}
json.dump(out, open(os.environ["DIP_OUT"], "w"))
"""


def partitions(prefix, cell):
    cache = OUT / f"partitions_{prefix}.json"
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    cfg = (HERE / "configs" / prefix / f"{cell}.toml").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = re.sub(r'(?m)^base_path = ".*"$', f'base_path = "{tmp}"', cfg, count=1)
        (Path(tmp) / "c.toml").write_text(cfg)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", DIP_OUT=str(cache), DIP_HERE=str(HERE),
                   PYTHONPATH=f"{REPO}:{os.environ.get('PYTHONPATH', '')}")
        r = subprocess.run([sys.executable, "-c", PART_SCRIPT, "-c", str(Path(tmp) / "c.toml")],
                           capture_output=True, text=True, cwd=HERE.parent, env=env, timeout=900)
        if r.returncode != 0:
            raise SystemExit(r.stderr[-1500:])
    return {int(k): v for k, v in json.loads(cache.read_text()).items()}


# --------------------------------------------------------------------------
# per run
# --------------------------------------------------------------------------

REPORT = re.compile(r"\[Stage7\] report round=(\d+) slot=(\d+) ")
TIER = re.compile(r"\[Stage7\] dispatch round=\d+ slot=\d+ veh_id=\S+ tier=(\S+) ")


def load(ds, prefix, cell):
    """Merges joined to their dispatch by analyze_stage7.parse's own rule, in log
    order: a merge belongs to the most recent dispatch on its slot that has been
    reported and not yet merged."""
    a7.PREFIX = prefix
    run = a7.parse(cell)
    log = a7.server_log(a7.RUNS / f"{prefix}_{cell}")
    order, pending, rows = [], {}, []
    for line in log.open(errors="replace"):
        m = PART.search(line)
        if m:
            d = {"round": int(m.group(1)), "slot": int(m.group(2)), "partition": int(m.group(3)),
                 "tier": TIER.search(line).group(1), "reported": False, "merged": False}
            order.append(d)
            pending[(d["round"], d["slot"])] = d
            continue
        m = REPORT.search(line)
        if m:
            d = pending.pop((int(m.group(1)), int(m.group(2))), None)
            if d is not None:
                d["reported"] = True
            continue
        m = MERGE.search(line)
        if m:
            slot, eff, tau, w = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
            d = next((d for d in reversed(order) if d["slot"] == slot and d["reported"] and not d["merged"]), None)
            if d is None:
                raise SystemExit(f"{prefix}/{cell}: merge r{eff} slot {slot} not joined")
            d["merged"] = True
            rows.append({"round": eff, "slot": slot, "tau": tau, "w": w, "tier": d["tier"], "partition": d["partition"]})
    # same join as analyze_stage7: the staleness sequence must agree
    a7_tau = sorted((d.merge_round, d.slot, d.staleness) for d in run.dispatches if d.merge_round is not None)
    assert sorted((r["round"], r["slot"], r["tau"]) for r in rows) == a7_tau, f"{prefix}/{cell}: join differs from analyze_stage7"
    evals = [(c["round"], c["accuracy"]) for c in run.curve if c["round"] % a7.EVAL_INTERVAL == 0]
    return rows, evals


def check_fingerprints(prefix, cell, P):
    """Rebuilt partitions == the fingerprints the clients logged."""
    seen, bad = {}, 0
    for f in (a7.RUNS / f"{prefix}_{cell}").glob("client_*.log"):
        for line in f.open(errors="replace"):
            m = FPR.search(line)
            if m:
                seen[int(m.group(1))] = m.group(2)
    for p, fp in seen.items():
        bad += P[p]["fingerprint"] != fp
    return len(seen), bad


def analyse_run(ds, prefix, cell, label, seed):
    rows, evals = load(ds, prefix, cell)
    # the partition depends only on dataset and data seed: one rebuild per (prefix family, seed)
    P = partitions(prefix if ds == "CIFAR10" else "stage13f_s4126", "gpu_open")
    n_fp, bad_fp = check_fingerprints(prefix, cell, P)
    acc = [100 * a for _, a in evals]
    med = a7.rolling_median(acc, WINDOW)
    res = [a - m for a, m in zip(acc, med)]
    thr = THRESH[ds]
    by_round = defaultdict(list)
    for r in rows:
        by_round[r["round"]].append(r)
    intervals = []
    for i, (rnd, _) in enumerate(evals):
        if rnd < MIN_ROUND:
            continue
        ms = [m for r in range(rnd - 9, rnd + 1) for m in by_round.get(r, [])]
        is_dip = res[i] < thr
        rec = None
        if is_dip:
            rec = next((k for k in range(1, len(res) - i) if res[i + k] > RECOVER), None)
        q = {"round": rnd, "acc": acc[i], "residual": res[i], "dip": is_dip, "recovery": rec}
        if ms:
            ws, ts = [m["w"] for m in ms], [m["tau"] for m in ms]
            c = Counter(m["tier"] for m in ms)
            top = max(ms, key=lambda m: m["w"])
            q.update(w_max=max(ws), w_sum=sum(ws), tau_max=max(ts), tau_mean=st.mean(ts),
                     entropy_min=min(P[m["partition"]]["entropy_bits"] for m in ms),
                     maxshare_max=max(P[m["partition"]]["max_share"] for m in ms),
                     n_xavier=c["xavier"], n_orin=c["orin"], n_thor=c["thor"], n_merges=len(ms),
                     top_partition=top["partition"], top_tier=top["tier"], top_tau=top["tau"],
                     top_w=top["w"], lag=rnd - top["round"],
                     lag_latest=rnd - max(m["round"] for m in ms if m["w"] == top["w"]),
                     n_w_top=sum(m["w"] == top["w"] for m in ms))
        intervals.append(q)
    return {"ds": ds, "prefix": prefix, "cell": cell, "label": label, "seed": seed,
            "intervals": intervals, "n_fp": n_fp, "bad_fp": bad_fp, "n_merges": len(rows),
            "w_values": sorted(Counter(round(r["w"], 4) for r in rows).items())[:6]}


# --------------------------------------------------------------------------
# statistics and report
# --------------------------------------------------------------------------

def compare(intervals):
    out = {}
    dip = [q for q in intervals if q["dip"] and "w_max" in q]
    oth = [q for q in intervals if not q["dip"] and "w_max" in q]
    for k in QUANT:
        a, b = [q[k] for q in dip], [q[k] for q in oth]
        if len(a) < 2 or len(b) < 2:
            out[k] = None
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        auc = u / (len(a) * len(b))
        out[k] = {"dip_median": st.median(a), "other_median": st.median(b), "auc": auc,
                  "rank_biserial": 2 * auc - 1, "p": p, "n_dip": len(a), "n_other": len(b)}
    return out


def f(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


READING = """**Which quantity separates dip intervals from the rest: the total weight merged, w_sum.**

**The separator.**
- Dip intervals carry more of the global model's mass handed to client
  updates in the preceding 10 rounds.
- AUC is 0.71 (CIFAR-100) and 0.73 (CIFAR-10), rank-biserial +0.42 / +0.46.
  It holds in every one of the 12 runs (per-run AUC 0.55–0.84).
- The dip rate rises steadily across the w_sum quartiles:
  - CIFAR-100: 1.1% → 2.1% → 4.4% → 7.8%;
  - CIFAR-10: 1.3% → 6.3% → 10.3% → 19.0%.
  That is a 7× to 15× difference between the lightest and heaviest quarter.

**What makes w_sum large.** It is not one merge.
- w depends only on staleness, and the number of merges per interval is
  constant (10). So a high w_sum means more fresh, full-weight (w = 0.5,
  τ = 0) merges in a row: median 5 vs 3 (CIFAR-100) and 4 vs 3 (CIFAR-10).
- The largest single weight (w_max) separates only weakly (AUC 0.56–0.58).
  Most intervals contain at least one w = 0.5 merge (82–87%).
- Using the latest of the tied full-weight merges, the dip evaluation more
  often has one in the evaluated round itself: 44% vs 34% (CIFAR-100) and 56%
  vs 29% (CIFAR-10). The pull is recent.

**What does not separate.**
- Staleness max: AUC ≈ 0.5.
- Tier: the top merge is orin or thor in dip and other intervals alike.
  xavier counts differ by at most one merge.
- Partition skew, in CIFAR-100: AUC 0.46–0.51. Its Dirichlet over 100 labels
  gives every partition a largest class of only about 7–8%.
- Partition skew in CIFAR-10 is a secondary signal. Pooled, the dips follow
  merges from more skewed partitions: largest-class share 0.51 vs 0.43,
  AUC 0.68. But it is not consistent across seeds: entropy AUC is 0.21–0.33
  in four runs and 0.54–0.56 in both seed-4127 runs.

**Arms.**
- Dip frequency is about the same in every arm:
  - CIFAR-10: baseline+F 9.1% vs gpu+F 9.3% mean over three seeds;
  - CIFAR-100: 3.1–4.6% across all six runs, including random selection
    without the filter.
- So **the dips come from the FedAsync training dynamics, not from any
  selector**.

**Strength and caveats.**
- This is a strong association, not proof of cause. No global-model
  checkpoint survives to test a merge directly.
- The mixing also averages the BatchNorm running statistics with the same w.
  The logs cannot separate weights from statistics.
- Report only; no fix is proposed or tested here."""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = [analyse_run(*r) for r in RUNS]
    L = ["# Accuracy dips: where do they come from?", "",
         "Log-only analysis. Spec: `prompts/DIP_analysis.md`. Generated by `sumo_open_loop/dip_analysis.py`; "
         "every dip interval is listed in `dip_intervals.csv`. Run folders were only read.", ""]
    w0 = R[0]["w_values"]
    L += ["## Step 0: what is available", "",
          "1. **Mixing and staleness.** Config: `mixing_hyperparameter = 0.5`, `adaptive_mixing = true`, "
          "`staleness_weighting_function = polynomial, a = 0.5`. Applied weight "
          "**w = 0.5 · (τ + 1)^−0.5** (`plato/servers/strategies/aggregation/fedasync.py:157–186`, "
          "`_staleness_function`; per update in arrival order with τ + idx in "
          "`examples/async/fedasync/fedasync_aggregation.py:66–90`). The merge is "
          "`x ← (1 − w)·x + w·x_client` on **every entry of the model state_dict** "
          "(`examples/async/fedasync/fedasync_algorithm.py:16–19`). That includes the BatchNorm running mean / variance buffers. "
          "Because `w` is a fixed function of τ, \"high w\" and \"low τ\" are the same thing: the largest weight, 0.5, "
          "goes to a fresh (τ = 0) update.",
          f"2. **Logged per merge.** `w`, τ and the client slot are logged directly on every merge "
          f"(`FedAsync: step k - client #slot, effective_round, staleness, mixing`), so `w` is **read, not recomputed**. "
          f"Tier and partition come from the dispatch the merge belongs to (`[Stage7] dispatch … tier … partition`), "
          f"joined as in `analyze_stage7.parse`; every merge in every run was joined (τ cross-checked). "
          f"Most frequent w values (first run): {', '.join(f'{w:g}' for w, _ in w0)}. "
          "Partition label skew comes from Plato's Dirichlet sampler, rebuilt for each run's data seed. "
          "It matches the partition fingerprints the clients logged: "
          + ", ".join(f"{r['prefix'].replace('_s', ' s')}/{r['cell'].split('_')[0]} {r['n_fp'] - r['bad_fp']}/{r['n_fp']}" for r in R) + ".",
          "3. **Checkpoints.** No global-model checkpoint survives for these runs: the queue deletes each run's "
          "`checkpoints/` when it ends. (`stage13f_s4126_gpu_open/checkpoints/` exists but is empty; it was created by "
          "a config-loading test on 2026-10-04.) So the dips can only be studied from the logs.", ""]

    L += ["## Step 1: dips (definition fixed before the results)", "",
          "Residual = evaluation − centred 21-evaluation rolling median. A dip is an evaluation at round ≥ 500 with "
          "residual **< −5 points (CIFAR-100)** or **< −3 points (CIFAR-10)**. Recovery is the number of evaluations "
          "until the residual is back above −1 point.", "",
          "| dataset | run | seed | evaluations ≥ r500 | dips | share | depth median / max (pts) | recovery median / max (evals) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in R:
        iv = r["intervals"]
        d = [q for q in iv if q["dip"]]
        rec = [q["recovery"] for q in d if q["recovery"] is not None]
        L.append(f"| {r['ds']} | {r['label']} | {r['seed']} | {len(iv)} | {len(d)} | {100 * len(d) / len(iv):.1f}% | "
                 + (f"{st.median(q['residual'] for q in d):.1f} / {min(q['residual'] for q in d):.1f}" if d else "—")
                 + " | " + (f"{st.median(rec):.0f} / {max(rec)}" if rec else "—") + " |")
    L.append("")

    L += ["## Step 2: what was merged just before each dip", "",
          "An interval is the 10 rounds before an evaluation (rounds ≥ 500). Dip intervals are compared with all "
          "others, pooled over the runs of each dataset. **AUC** = P(value in a dip interval > value in another "
          "interval), from Mann–Whitney; 0.5 = no separation. Rank-biserial r = 2·AUC − 1.", ""]
    for ds in ("CIFAR100", "CIFAR10"):
        iv = [q for r in R if r["ds"] == ds for q in r["intervals"]]
        C = compare(iv)
        nd = sum(q["dip"] for q in iv)
        L += [f"### {ds} ({nd} dip intervals, {len(iv) - nd} others)", "",
              "| quantity | dip median | other median | AUC | rank-biserial | p |", "|---|---|---|---|---|---|"]
        for k in QUANT:
            c = C[k]
            L.append(f"| {k} | " + ("— | — | — | — | — |" if c is None else
                     f"{f(c['dip_median'])} | {f(c['other_median'])} | **{c['auc']:.3f}** | {c['rank_biserial']:+.3f} | {c['p']:.2g} |"))
        L.append("")
        # per run AUC for the two w quantities and entropy
        L += ["Per run (AUC):", "", "| run | seed | dips | w_max | w_sum | tau_mean | entropy_min | n_xavier | n_merges |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in [r for r in R if r["ds"] == ds]:
            c = compare(r["intervals"])
            L.append(f"| {r['label']} | {r['seed']} | {sum(q['dip'] for q in r['intervals'])} | "
                     + " | ".join("—" if c[k] is None else f"{c[k]['auc']:.2f}" for k in
                                  ("w_max", "w_sum", "tau_mean", "entropy_min", "n_xavier", "n_merges")) + " |")
        L.append("")
        ws = sorted(q["w_sum"] for q in iv if "w_sum" in q)
        cuts = [ws[len(ws) * k // 4] for k in (1, 2, 3)]
        L += ["Dip rate by quartile of w_sum (total weight merged in the interval):", "",
              "| w_sum quartile | range | intervals | dip rate | full-weight (w = 0.5) merges, median |",
              "|---|---|---|---|---|"]
        for k, (lo, hi) in enumerate(zip([-1.0] + cuts, cuts + [99.0])):
            g = [q for q in iv if "w_sum" in q and lo < q["w_sum"] <= hi]
            L.append(f"| Q{k + 1} | {'' if lo < 0 else f'> {lo:.2f} '}{'' if hi > 90 else f'≤ {hi:.2f}'} | {len(g)} | "
                     f"{100 * sum(q['dip'] for q in g) / len(g):.1f}% | {st.median(q['n_w_top'] for q in g if q['w_max'] == 0.5) if any(q['w_max'] == 0.5 for q in g) else 0:.0f} |")
        L.append("")
        top = [q for q in iv if q["dip"] and "top_w" in q]
        L += ["The single largest-w merge in each dip interval:", "",
              f"- tier: {dict(Counter(q['top_tier'] for q in top))}; in other intervals: "
              f"{dict(Counter(q['top_tier'] for q in iv if not q['dip'] and 'top_w' in q))}",
              f"- τ: {dict(sorted(Counter(q['top_tau'] for q in top).items()))}",
              f"- partition (count): {dict(Counter(q['top_partition'] for q in top).most_common(8))}", ""]

    L += ["## Step 3: two direct checks", "", "### 1. Lag", "",
          "Rounds between the highest-w merge of the interval and the evaluation (0 = merged in the evaluated round). "
          "Almost every interval holds several merges tied at the top weight (w = 0.5, a τ = 0 update), so the lag "
          "is shown for both tie-breaks: the earliest and the latest of the tied merges.", "",
          "| dataset | group | n | tied top-w merges per interval (median) | tie-break | lag distribution (0 … 9) | median |",
          "|---|---|---|---|---|---|---|"]
    for ds in ("CIFAR100", "CIFAR10"):
        iv = [q for r in R if r["ds"] == ds for q in r["intervals"] if "lag" in q]
        for g, keep in (("dip", True), ("other", False)):
            sel = [q for q in iv if q["dip"] is keep]
            for tb, key in (("earliest", "lag"), ("latest", "lag_latest")):
                lags = [q[key] for q in sel]
                c = Counter(lags)
                L.append(f"| {ds} | {g} | {len(lags)} | {st.median(q['n_w_top'] for q in sel):.0f} | {tb} | "
                         + " ".join(f"{100 * c[k] / len(lags):.0f}%" for k in range(10))
                         + f" | {st.median(lags) if lags else '—'} |")
    L += ["", "### 2. Arms", "", "| dataset | arm | runs | dip share per run | mean |", "|---|---|---|---|---|"]
    arms = defaultdict(list)
    for r in R:
        arms[(r["ds"], r["label"].replace(" (replicate)", ""))].append(
            100 * sum(q["dip"] for q in r["intervals"]) / len(r["intervals"]))
    for (ds, arm), v in arms.items():
        L.append(f"| {ds} | {arm} | {len(v)} | " + " / ".join(f"{x:.1f}%" for x in v) + f" | {st.mean(v):.1f}% |")
    L += ["", "## Reading", "", READING, ""]
    (OUT / "DIP_analysis_report.md").write_text("\n".join(L) + "\n")
    with (OUT / "dip_intervals.csv").open("w") as fh:
        keys = ["round", "acc", "residual", "dip", "recovery"] + list(QUANT) + ["top_partition", "top_tier", "top_tau", "top_w", "lag", "lag_latest", "n_w_top"]
        fh.write("dataset,prefix,cell,seed," + ",".join(keys) + "\n")
        for r in R:
            for q in r["intervals"]:
                fh.write(f"{r['ds']},{r['prefix']},{r['cell']},{r['seed']}," + ",".join(str(q.get(k, "")) for k in keys) + "\n")
    json.dump([{k: v for k, v in r.items() if k != "intervals"} | {"compare": compare(r["intervals"])} for r in R],
              open(OUT / "dip_summary.json", "w"), indent=1, default=str)
    print(f"wrote {OUT / 'DIP_analysis_report.md'}")


if __name__ == "__main__":
    main()
