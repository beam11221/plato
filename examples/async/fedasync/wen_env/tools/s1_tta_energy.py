"""S1: TTA@80/85/90% and ground-truth energy to each, per seed (random, gpu, wen, wen_fresh, wen_fresh_gpu, polaris)."""
import statistics as st
import sys
import wen_main_report as R
from wen_ptable import run_dir

ARMS = [("random", "S1_random"), ("Ours (gpu)", "S1_gpu"), ("Wen et al. (wen)", "S1_wen"),
        ("Wen et al. (wen_fresh)", "S1_wen_fresh"),
        ("Wen et al. + GPU (wen_fresh_gpu)", "S1_wen_fresh_gpu"), ("Polaris", "S1_polaris")]
THR = (0.80, 0.85, 0.90)
READING_B = {("S1_gpu", 4126)}
res = {}
for name, cell in ARMS:
    for seed in (4126, 4127, 4128):
        run = run_dir(cell, seed)
        if not any((run / "results").glob("*.csv")):
            continue
        curve = R.accuracy_curve(run)
        d = R._rows(run / "dispatch_log.csv")
        out = {}
        for thr in THR:
            t = next((t for _, t, a in curve if a >= thr), None)
            if t is None and thr == 0.90 and (cell, seed) in READING_B:
                best = max(a for _, _, a in curve)
                t = next(t for _, t, a in curve if a == best)
            e = None if t is None else sum(float(r["E_train"]) + float(r["E_up"]) for r in d
                                           if float(r["t_finish_clean"]) <= t) / 1000
            out[thr] = (t, e)
        out["max"] = max(a for _, _, a in curve)
        res[(name, seed)] = out

def cell(v, f):
    return "not reached" if v is None else f.format(v)

lines = ["| method | seed | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | energy to 80% (kJ) | energy to 85% (kJ) | energy to 90% (kJ) | max accuracy |",
         "|---|---|---|---|---|---|---|---|---|"]
summary = ["| method | seeds | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | energy to 80% (kJ) | energy to 85% (kJ) | energy to 90% (kJ) |",
           "|---|---|---|---|---|---|---|---|"]
for name, _ in ARMS:
    seeds = [s for s in (4126, 4127, 4128) if (name, s) in res]
    for s in seeds:
        o = res[(name, s)]
        lines.append(f"| {name} | {s} | " + " | ".join(cell(o[t][0], "{:,.0f}") for t in THR)
                     + " | " + " | ".join(cell(o[t][1], "{:.1f}") for t in THR) + f" | {o['max']:.4f} |")
    def agg(i, t, f):
        vals = [res[(name, s)][t][i] for s in seeds]
        hit = [v for v in vals if v is not None]
        if len(hit) < len(vals):
            return f"{len(hit)}/{len(vals)} reached"
        return f.format(st.fmean(hit)) + (f" ± {f.format(st.stdev(hit))}" if len(hit) > 1 else "")
    summary.append(f"| {name} | {len(seeds)} | " + " | ".join(agg(0, t, "{:,.0f}") for t in THR)
                   + " | " + " | ".join(agg(1, t, "{:.1f}") for t in THR) + " |")
print("\n".join(summary)); print(); print("\n".join(lines))
