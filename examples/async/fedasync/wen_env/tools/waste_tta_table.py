"""Waste / TTA@90% table for Random, Wen and Ours, S1 vs S2, with and without Stage 0.

Reads only the final `[WenEnv] summary` / `[WenStage0] summary` lines and the
`Target accuracy reached` marker of each run's server log, so it can be re-run
while other runs are still going (unfinished runs are marked "running").

    python wen_env/tools/waste_tta_table.py [--seed 4126] [--out wen_env/evidence/waste_tta_s4126.md]
"""

import argparse
import csv
import glob
import os
import re
import subprocess

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (method, display arm, S1 run, S2 run without Stage 0, S2 run with Stage 0)
ARMS = [
    ("Random", "random", "S1_random", "S2_random", "S2_random_s0"),
    ("Wen et al. (original, `wen`)", "wen", "S1_wen", "S2_wen", "S2_wen_s0"),
    ("Wen et al. (fresh F3, `wen_fresh`)", "wen_fresh", "S1_wen_fresh", "S2_wen_fresh", "S2_wen_fresh_s0"),
    ("Wen et al. (fresh F3) + GPU plug-in", "wen_fresh_gpu", "S1_wen_fresh_gpu", "S2_wen_fresh_gpu", "S2_wen_fresh_gpu_s0"),
    ("Ours (GPU estimator)", "gpu", "S1_gpu", "S2_gpu", "S2_gpu_stage0"),
]

SUMMARY = re.compile(
    r"round=(\d+) cycles=(\d+) t=([\d.]+) waste xavier=(\d+)/(\d+) orin=(\d+)/(\d+) thor=(\d+)/(\d+)"
    r".*? E=([\d.]+) J \(total E ([\d.]+) J\)"
)
STAGE0 = re.compile(r"candidates=(\d+) rejected=(\d+) \(([\d.]+)%\) fallback_cycles=(\d+)")
ACC = re.compile(r"Global model accuracy: ([\d.]+)%")

# Runs that died before 90% but that the user decided count as reaching it
# (wen_faithful_v1_step6_footnote.md, Reading B). TTA is the first evaluated
# round at the run's maximum accuracy; the other rows are recomputed from the
# dispatch log up to that round, the same way the [WenEnv] summary counts them.
ROUNDED_TO_90 = {"S1_gpu_s4126": 0.8975, "S1_cpu_s4126": 0.8968}
TIERS = ("xavier", "orin", "thor")


def server_log(run):
    logs = sorted(glob.glob(os.path.join(BASE, "runs/wen_env", run, "server_*.log")))
    if logs:
        return logs[-1]
    path = os.path.join(BASE, "wen_env/logs", run + ".log")
    if os.path.exists(path):
        return path
    # A run that died is moved to _failed/<run>_<reason>; shown, but marked invalid.
    logs = sorted(glob.glob(os.path.join(BASE, "runs/wen_env/_failed", run + "_hung*", "server_*.log")))
    return logs[-1] if logs else None


def is_running(run):
    pat = f"wen_fedasync.py -c wen_env/configs/runs/{run}.toml"
    return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0


def parse(run):
    log = server_log(run)
    if log is None:
        return None
    text = open(log, errors="replace").read()
    sums = SUMMARY.findall(text)
    accs = ACC.findall(text)
    st0 = STAGE0.findall(text)
    r = {"run": run, "running": is_running(run), "failed": "/_failed/" in log, "reached": "Target accuracy reached" in text,
         "acc": float(accs[-1]) if accs else None}
    if sums:
        s = sums[-1]
        r.update(round=int(s[0]), cycles=int(s[1]), t=float(s[2]),
                 waste=[(int(s[3]), int(s[4])), (int(s[5]), int(s[6])), (int(s[7]), int(s[8]))],
                 e_waste=float(s[9]), e_total=float(s[10]))
    if r["failed"] and run in ROUNDED_TO_90:
        r.update(rounded_stats(os.path.dirname(log), ROUNDED_TO_90[run]))
    if st0:
        c, rej, pct, fb = st0[-1]
        r["stage0"] = f"{float(pct):.1f}% (fallback {fb})"
    return r


def rounded_stats(d, acc):
    res = glob.glob(os.path.join(d, "results", "*.csv"))[0]
    hit = next(x for x in csv.DictReader(open(res)) if float(x["accuracy"]) >= acc)
    rnd = int(hit["round"])
    cyc = max(int(x["cycle"]) for x in csv.DictReader(open(os.path.join(d, "cycle_log.csv")))
              if int(x["round"]) == rnd)
    waste = {t: [0, 0] for t in TIERS}
    e_tot = e_waste = 0.0
    for x in csv.DictReader(open(os.path.join(d, "dispatch_log.csv"))):
        if int(x["cycle"]) > cyc:
            continue
        lost = x["accepted"] != "1"
        waste[x["tier"]][0] += lost
        waste[x["tier"]][1] += 1
        e = float(x["E_train"] or 0) + float(x["E_up"] or 0)
        e_tot += e
        e_waste += e * lost
    return {"reached": True, "rounded": acc, "round": rnd, "cycles": cyc, "t": float(hit["elapsed_time"]),
            "waste": [tuple(waste[t]) for t in TIERS], "e_waste": e_waste, "e_total": e_tot}


def tta_levels(run, levels=(0.80, 0.85, 0.90)):
    """First (time, round) at each accuracy level from the run's results CSV, plus the last row."""
    log = server_log(run)
    if log is None:
        return None
    res = glob.glob(os.path.join(os.path.dirname(log), "results", "*.csv"))
    if not res:  # log under wen_env/logs: results live in the run directory
        res = glob.glob(os.path.join(BASE, "runs/wen_env", run, "results", "*.csv"))
    if not res:
        return None
    rows = list(csv.DictReader(open(res[0])))
    thr = {lv: ROUNDED_TO_90[run] if lv == 0.90 and run in ROUNDED_TO_90 else lv for lv in levels}
    hits = {}
    for lv in levels:
        hit = next((x for x in rows if float(x["accuracy"]) >= thr[lv] - 1e-12), None)
        hits[lv] = (float(hit["elapsed_time"]), int(hit["round"])) if hit else None
    return hits, rows[-1]


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "—"


def cells(r):
    """Column values for one run, in table row order."""
    if r is None:
        return ["not run"] + [""] * 8
    if "round" not in r:
        return ["no summary yet"] + [""] * 8
    if r.get("rounded"):
        tta = f"**{r['t']:.0f}** † ({100 * r['rounded']:.2f}% counted as 90%)"
    elif r["reached"]:
        tta = f"**{r['t']:.0f}**"
    elif r["failed"]:
        tta = f"invalid: run died at r{r['round']} ({r['acc']:.2f}%)"
    elif r["running"]:
        tta = f"running (r{r['round']}, {r['acc']:.2f}%)"
    else:
        tta = f"not reached (last {r['acc']:.2f}%)"
    w = r["waste"]
    tot = (sum(a for a, _ in w), sum(b for _, b in w))
    return [
        tta,
        str(r["round"]),
        str(r["cycles"]),
        pct(*tot),
        pct(*w[0]),
        pct(*w[1]),
        pct(*w[2]),
        f"{r['e_waste'] / 1e3:.1f} / {r['e_total'] / 1e3:.1f} kJ",
        r.get("stage0", "—"),
    ]


ROWS = ["TTA@90% (s)", "Round", "Dispatch cycles", "Waste (all)", "Waste: Xavier", "Waste: Orin",
        "Waste: Thor", "Wasted / total energy", "Stage 0 rejected"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="4126")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sfx = f"_s{a.seed}"
    out = [f"# Waste and TTA@90%, seed {a.seed}", "",
           "Generated by `wen_env/tools/waste_tta_table.py` from each run's final summary line.",
           "S1 + Stage 0 is not run: in S1 coverage is the whole loop, so Stage 0 rejects nothing",
           "(Step 0c: 0/5,602 rejected, dispatch log identical to S1 gpu).",
           "† Run hung when the disk filled; counted as reaching 90% by user decision",
           "(wen_faithful_v1_step6_footnote.md, Reading B). TTA = first evaluated round at its max accuracy.", ""]
    for name, _, s1, s2, s2s0 in ARMS:
        rs = [parse(x + sfx) for x in (s1, s2, s2s0)]
        out += [f"## {name}", "", f"| | S1 (`{s1}`) | S2 (`{s2}`) | S2 + Stage 0 (`{s2s0}`) |",
                "|---|---|---|---|"]
        cols = [cells(r) for r in rs]
        for i, label in enumerate(ROWS):
            out.append(f"| {label} | " + " | ".join(c[i] for c in cols) + " |")
        out.append("")
    # Side-by-side against Random in the same column (same scenario / Stage 0 setting).
    res = {name: [parse(x + sfx) for x in runs] for name, _, *runs in ARMS}
    rnd = res["Random"]
    heads = ["S1", "S2", "S2 + Stage 0"]
    out += [f"## Compared with Random (seed {a.seed})", "",
            "Δ = (method − Random) / Random in the same column; negative TTA Δ means faster than Random.",
            "Running runs show their current state and have no Δ yet.", "",
            "### TTA@90% (s)", "", "| method | " + " | ".join(heads) + " |", "|---|---|---|---|"]

    def done(r):
        return r is not None and "round" in r and r["reached"]

    for name, rs in res.items():
        row = []
        for r, b in zip(rs, rnd):
            if not done(r):
                row.append(cells(r)[0] if r is not None and "round" in r else "not run")
            elif name == "Random" or not done(b):
                row.append(f"{r['t']:.0f}")
            else:
                row.append(f"{r['t']:.0f} ({100 * (r['t'] - b['t']) / b['t']:+.1f}%)")
        out.append(f"| {name} | " + " | ".join(row) + " |")
    out += ["", "### Waste (lost updates / dispatched)", "", "| method | " + " | ".join(heads) + " |",
            "|---|---|---|---|"]
    for name, rs in res.items():
        row = []
        for r, b in zip(rs, rnd):
            if r is None or "round" not in r:
                row.append("not run")
                continue
            lost, sent = (sum(v) for v in zip(*r["waste"]))
            txt = f"{100 * lost / sent:.1f}%"
            if r["running"]:
                txt += " (running)"
            elif name != "Random" and b is not None and "round" in b:
                bl, bs = (sum(v) for v in zip(*b["waste"]))
                txt += f" ({100 * (lost / sent - bl / bs):+.1f} pp)"
            row.append(txt)
        out.append(f"| {name} | " + " | ".join(row) + " |")
    out.append("")

    out += [f"## TTA at 80 / 85 / 90% (seed {a.seed})", "",
            "Simulated seconds (round). Δ is against Random in the same column. † = 90% by user decision.", ""]
    for lv in (0.80, 0.85, 0.90):
        out += [f"### TTA@{lv:.0%}", "", "| method | " + " | ".join(heads) + " |", "|---|---|---|---|"]
        for name, _, *runs in ARMS:
            row = []
            for run, base in zip(runs, ARMS[0][2:]):
                got = tta_levels(run + sfx)
                if got is None:
                    row.append("not run")
                    continue
                hits, last = got
                h = hits[lv]
                if h is None:
                    state = "running" if is_running(run + sfx) else "not reached"
                    row.append(f"{state} (r{last['round']}, {100 * float(last['accuracy']):.2f}%)")
                    continue
                txt = f"{h[0]:.0f} (r{h[1]})" + (" †" if lv == 0.90 and run + sfx in ROUNDED_TO_90 else "")
                b = tta_levels(base + sfx)
                if name != "Random" and b is not None and b[0][lv] is not None:
                    txt += f" {100 * (h[0] - b[0][lv][0]) / b[0][lv][0]:+.1f}%"
                row.append(txt)
            out.append(f"| {name} | " + " | ".join(row) + " |")
        out.append("")

    out += [f"## Seconds per round (seed {a.seed})", "",
            "Cumulative = TTA / round at that level. Phase = (Δ seconds) / (Δ rounds) between levels;",
            "TTA = rounds × s/round, so this splits a TTA gap into a round-count part and a per-round part.", ""]
    segs = [("0→80%", None, 0.80), ("80→85%", 0.80, 0.85), ("85→90%", 0.85, 0.90)]
    for scen_i, head in enumerate(heads):
        out += [f"### {head}", "", "| method | rounds to 80 / 85 / 90% | cumulative s/round at 80 / 85 / 90% | "
                + " | ".join(f"phase {g[0]}" for g in segs) + " |", "|---|---|---|" + "---|" * len(segs)]
        for name, _, *runs in ARMS:
            got = tta_levels(runs[scen_i] + sfx)
            if got is None:
                continue
            hits = got[0]
            rounds = " / ".join(str(hits[lv][1]) if hits[lv] else "—" for lv in (0.80, 0.85, 0.90))
            cum = " / ".join(f"{hits[lv][0] / hits[lv][1]:.2f}" if hits[lv] else "—" for lv in (0.80, 0.85, 0.90))
            ph = []
            for _, lo, hi in segs:
                a0 = (0.0, 0) if lo is None else hits[lo]
                if a0 is None or hits[hi] is None or hits[hi][1] == a0[1]:
                    ph.append("—")
                else:
                    ph.append(f"{(hits[hi][0] - a0[0]) / (hits[hi][1] - a0[1]):.2f} ({hits[hi][1] - a0[1]} r)")
            out.append(f"| {name} | {rounds} | {cum} | " + " | ".join(ph) + " |")
        out.append("")

    out += ["## TTA@90% across seeds (s)", "",
            "Seed spread is large (~1,000 s), so single-seed TTA gaps of a few hundred seconds are not conclusive.", "",
            "| run | s4126 | s4127 | s4128 |", "|---|---|---|---|"]
    for _, _, *runs in ARMS:
        for run in runs:
            vals = []
            for sd in ("4126", "4127", "4128"):
                r = parse(f"{run}_s{sd}")
                vals.append(cells(r)[0] if r is None or "round" in r else "no summary yet")
            if any(v != "not run" for v in vals):
                out.append(f"| {run} | " + " | ".join(vals) + " |")
    out.append("")
    md = "\n".join(out)
    if a.out:
        with open(a.out, "w") as f:
            f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
