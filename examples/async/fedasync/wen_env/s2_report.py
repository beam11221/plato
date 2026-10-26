"""
Prompt L Step 6: Scenario 2, seed 4126, every S2-B arm (mechanical).

    PYTHONPATH=sumo_close_loop:<repo> python wen_env/s2_report.py \
        --out wen_env/evidence/s2_s4126_tables.md

Definitions (Prompt L S2-D, S2-F):
    energy to X %   E_train + E_up of every dispatch finished by TTA@X, lost uploads
                    included; "of which wasted" = the lost ones among them
    STR             merged / dispatched over the whole run (= 1 - waste fraction)
    uplink airtime  sum of T_up of dispatches finished by TTA@X; a lost upload is
                    charged its full T_up (the environment has no partial upload:
                    an upload is lost when the vehicle leaves coverage at any time
                    during it, wen_truth.py)
    staleness       of a merged update = round at merge - round at dispatch
"""

from __future__ import annotations

import argparse
import statistics as st
from collections import Counter
from pathlib import Path

import s1_final_report as F
import wen_main_report as R
import wen_study1 as S1m
from polaris_report import fmt, rows
from study2_report import stage0_stats
from wen_ptable import run_dir

HERE = Path(__file__).resolve().parent
SEED = 4126
THR = (0.80, 0.85, 0.90)
TIERS = ("xavier", "orin", "thor")
MAIN = [("Random", "S2_random"), ("Ours (gpu + Stage 0)", "S2_gpu_stage0"),
        ("Polaris", "S2_polaris"), ("Wen et al. (wen_fresh)", "S2_wen_fresh")]
PLUGIN = [("random + S0", "S2_random_s0"), ("wen_fresh + S0", "S2_wen_fresh_s0"),
          ("wen_fresh + GPU", "S2_wen_fresh_gpu"), ("wen_fresh + GPU + S0", "S2_wen_fresh_gpu_s0"),
          ("Polaris + S0", "S2_polaris_s0")]
ABLATION = [("Ours without Stage 0 (gpu)", "S2_gpu")]
ARMS = MAIN + PLUGIN + ABLATION
PAIRS = [("S2_random", "S2_random_s0"), ("S2_wen_fresh", "S2_wen_fresh_s0"),
         ("S2_wen_fresh_gpu", "S2_wen_fresh_gpu_s0"), ("S2_polaris", "S2_polaris_s0"),
         ("S2_wen_fresh", "S2_wen_fresh_gpu")]
NAME = dict((c, n) for n, c in ARMS)


def staleness(run: Path) -> float:
    """Mean (merge round - dispatch round) over merged (accepted) updates; the
    k-th merge of a client pairs with its k-th dispatch."""
    disp, merges = rows(run / "dispatch_log.csv"), rows(run / "merge_log.csv")
    by_client: dict[int, list[int]] = {}
    for d in disp:
        by_client.setdefault(int(d["client_id"]), []).append(int(d["round"]))
    seen = Counter()
    out = []
    for m in merges:
        c = int(m["client_id"])
        start = by_client[c][seen[c]]
        seen[c] += 1
        if m["accepted"] == "1":
            out.append(int(m["round"]) - start)
    return st.fmean(out)


def metrics(cell: str, seed: int = SEED) -> dict:
    run = run_dir(cell, seed)
    m = S1m.one(cell, seed)
    curve = R.accuracy_curve(run)
    disp = rows(run / "dispatch_log.csv")
    lost = [r for r in disp if r["accepted"] == "0"]
    m["max_acc"] = max(a for _, _, a in curve)
    m["hit"] = {}
    for thr in THR:
        hit = next(((rnd, t) for rnd, t, a in curve if a >= thr), None)
        if hit is None:
            m["hit"][thr] = dict(t=None, rounds=None, e=None, e_waste=None, air=None)
            continue
        done = [r for r in disp if float(r["t_finish_clean"]) <= hit[1]]
        m["hit"][thr] = dict(
            t=hit[1], rounds=hit[0],
            e=sum(float(r["E_train"]) + float(r["E_up"]) for r in done) / 1000,
            e_waste=sum(float(r["E_train"]) + float(r["E_up"]) for r in done
                        if r["accepted"] == "0") / 1000,
            air=sum(float(r["T_up_true"]) for r in done))
    m["str"] = 1 - len(lost) / len(disp)
    m["fail_gpu_s"] = sum(float(r["T_train_true"]) for r in lost)
    m["fail_kj"] = sum(float(r["E_train"]) + float(r["E_up"]) for r in lost) / 1000
    m["staleness"] = staleness(run)
    m["distinct"] = len({r["client_id"] for r in disp})
    m["cov"] = _coverage(cell, seed)
    m["s0"] = stage0_stats(run)
    m["curve"] = curve
    m["energy_curve"] = F.energy_curve(curve, disp)
    return m


def _coverage(cell: str, seed: int = SEED) -> dict:
    import wen_concentration as C
    by_check = {check: cov for check, _upto, _kl, cov, _n in
                C.step4("S2", cell.split("_", 1)[1], seed)}
    return {"r500": by_check[500], "r1000": by_check[1000], "end": by_check[4000]}


# ---------------------------------------------------------------------------
# 6.1
# ---------------------------------------------------------------------------

def _thr(m, key, f):
    return " / ".join(fmt(m["hit"][t][key], f) for t in THR)


def _s0(m) -> str:
    s = m["s0"]
    if s is None:
        return "— | — | —"
    reason = (" / ".join(fmt(s[k], "{:.3f}") for k in ("time", "energy", "both"))
              if s["time"] is not None else "not logged")
    tier = (" / ".join(fmt(s[t], "{:.3f}") for t in TIERS) if s["xavier"] is not None
            else "counts " + " / ".join(str(s.get("rej_by_tier_count", {}).get(t, 0))
                                        for t in TIERS))
    return f"{s['rate']:.3f} | {reason} | {tier}"


def table_main(data) -> list[str]:
    out = ["| arm | TTA@80 / 85 / 90% (s) | rounds to 80 / 85 / 90% | max accuracy | "
           "energy to 80 / 85 / 90% (kJ) | of which wasted (kJ) | **STR** | failed GPU-s | "
           "failed kJ | uplink airtime to 80 / 85 / 90% (s) | mean staleness (rounds) |",
           "|" + "---|" * 11]
    for name, cell in ARMS:
        m = data[cell]
        out.append(f"| {name} | {_thr(m, 't', '{:,.0f}')} | {_thr(m, 'rounds', '{:,.0f}')} | "
                   f"{m['max_acc']:.4f} | {_thr(m, 'e', '{:.1f}')} | "
                   f"{_thr(m, 'e_waste', '{:.2f}')} | **{m['str']:.3f}** | "
                   f"{m['fail_gpu_s']:,.0f} | {m['fail_kj']:.1f} | {_thr(m, 'air', '{:,.0f}')} | "
                   f"{m['staleness']:.2f} |")
    return out


def table_behaviour(data) -> list[str]:
    out = ["| arm | s/round whole / <80% / ≥80% | share X / O / T r1–500 | "
           "share X / O / T last 500 | Gini | distinct vehicles | "
           "image coverage r500 / r1000 / end | mean \\|x_upload − 1000\\| (m) | "
           "Stage 0 rejected (of eligible) | by reason time / energy / both | "
           "by tier X / O / T |", "|" + "---|" * 11]
    for name, cell in ARMS:
        m = data[cell]
        out.append(
            f"| {name} | {m['spr']:.3f} / {fmt(m['before'])} / {fmt(m['after'])} | "
            + " / ".join(f"{m[f'first_{t}']:.2f}" for t in TIERS) + " | "
            + " / ".join(f"{m[f'last_{t}']:.2f}" for t in TIERS)
            + f" | {m['gini']:.3f} | {m['distinct']} | "
            + " / ".join(f"{m['cov'][k]:.3f}" for k in ("r500", "r1000", "end"))
            + f" | {m['upload_dist']:.0f} | {_s0(m)} |")
    return out


# ---------------------------------------------------------------------------
# 6.2 pairwise, 6.3 plug-in pairs, 6.4 P-table
# ---------------------------------------------------------------------------

def pairwise(data) -> list[str]:
    ours = data["S2_gpu_stage0"]
    out = ["| Ours vs | TTA 80 / 85 / 90% | energy 80 / 85 / 90% | STR |", "|---|---|---|---|"]
    for name, cell in ARMS:
        if cell == "S2_gpu_stage0":
            continue
        o = data[cell]

        def sign(key):
            res = []
            for t in THR:
                a, b = ours["hit"][t][key], o["hit"][t][key]
                res.append("both n/r" if a is None and b is None
                           else "<" if S1m.lower(a, b) else ">")
            return " / ".join(res)
        str_sign = ">" if ours["str"] > o["str"] else "<" if ours["str"] < o["str"] else "="
        out.append(f"| {name} | {sign('t')} | {sign('e')} | {str_sign} "
                   f"({ours['str']:.3f} vs {o['str']:.3f}) |")
    return out


def _d(a, b, f):
    if a is None or b is None:
        return "n/r" if a is None and b is None else f"{fmt(a, f)} → {fmt(b, f)}"
    return f"{b - a:+{f[2:-1]}}"


def plugin_pairs(data) -> list[str]:
    out = ["| A → B | ΔTTA 80 / 85 / 90% (s) | Δrounds 80 / 85 / 90% | Δmax acc | "
           "Δenergy 80 / 85 / 90% (kJ) | ΔSTR | Δfailed GPU-s | Δfailed kJ | "
           "Δairtime 80 / 85 / 90% (s) | Δstaleness | Δs/round | ΔGini | Δdistinct | "
           "Δ\\|x_upload − 1000\\| (m) |", "|" + "---|" * 14]
    for a_cell, b_cell in PAIRS:
        a, b = data[a_cell], data[b_cell]
        thr = lambda k, f: " / ".join(_d(a["hit"][t][k], b["hit"][t][k], f) for t in THR)
        out.append(
            f"| {NAME[a_cell]} → {NAME[b_cell]} | {thr('t', '{:.0f}')} | "
            f"{thr('rounds', '{:.0f}')} | {b['max_acc'] - a['max_acc']:+.4f} | "
            f"{thr('e', '{:.1f}')} | {b['str'] - a['str']:+.3f} | "
            f"{b['fail_gpu_s'] - a['fail_gpu_s']:+,.0f} | {b['fail_kj'] - a['fail_kj']:+.1f} | "
            f"{thr('air', '{:.0f}')} | {b['staleness'] - a['staleness']:+.2f} | "
            f"{b['spr'] - a['spr']:+.3f} | {b['gini'] - a['gini']:+.3f} | "
            f"{b['distinct'] - a['distinct']:+d} | {b['upload_dist'] - a['upload_dist']:+.0f} |")
    return out


def p_table(data) -> list[str]:
    def tta(c):
        return data[c]["hit"][0.80]["t"]

    def e80(c):
        return data[c]["hit"][0.80]["e"]

    def row(p, text, ok, a, b):
        return (f"| {p} | {text} | {a} vs {b} | {'as predicted' if ok else 'not as predicted'} "
                "| single seed — not evaluated |")
    s = lambda c: f"{data[c]['str']:.3f}"
    t = lambda c: fmt(tta(c), "{:,.0f}")
    return [
        "| P | prediction (waste lower evaluated as STR higher, v3.6) | seed 4126 | direction | status |",
        "|---|---|---|---|---|",
        row("P25", "gpu (Stage 0 off) has lower TTA@80% than wen_fresh",
            S1m.lower(tta("S2_gpu"), tta("S2_wen_fresh")), t("S2_gpu"), t("S2_wen_fresh")),
        row("P26", "random + Stage 0 has a lower waste fraction than random (STR higher)",
            data["S2_random_s0"]["str"] > data["S2_random"]["str"], s("S2_random_s0"),
            s("S2_random")),
        row("P27", "random + Stage 0 has a lower TTA@80% than random",
            S1m.lower(tta("S2_random_s0"), tta("S2_random")), t("S2_random_s0"), t("S2_random")),
        row("P28", "wen_fresh + Stage 0 has a lower waste fraction than wen_fresh (STR higher)",
            data["S2_wen_fresh_s0"]["str"] > data["S2_wen_fresh"]["str"], s("S2_wen_fresh_s0"),
            s("S2_wen_fresh")),
        row("P33", "wen_fresh + GPU + Stage 0 waste fraction no higher than wen_fresh + GPU "
            "(STR no lower)", data["S2_wen_fresh_gpu_s0"]["str"] >= data["S2_wen_fresh_gpu"]["str"],
            s("S2_wen_fresh_gpu_s0"), s("S2_wen_fresh_gpu")),
        row("P38", "Ours has lower TTA@80% than Polaris",
            S1m.lower(tta("S2_gpu_stage0"), tta("S2_polaris")), t("S2_gpu_stage0"),
            t("S2_polaris")),
        row("P39", "Ours has a higher STR than Polaris",
            data["S2_gpu_stage0"]["str"] > data["S2_polaris"]["str"], s("S2_gpu_stage0"),
            s("S2_polaris")),
        row("P40", "Ours uses less energy to reach 80% than Polaris (failed included)",
            S1m.lower(e80("S2_gpu_stage0"), e80("S2_polaris")),
            fmt(e80("S2_gpu_stage0"), "{:.1f} kJ"), fmt(e80("S2_polaris"), "{:.1f} kJ")),
        row("P41", "Ours has lower TTA@80% than wen_fresh",
            S1m.lower(tta("S2_gpu_stage0"), tta("S2_wen_fresh")), t("S2_gpu_stage0"),
            t("S2_wen_fresh")),
        row("P42", "Polaris + Stage 0 has a higher STR than Polaris",
            data["S2_polaris_s0"]["str"] > data["S2_polaris"]["str"], s("S2_polaris_s0"),
            s("S2_polaris")),
        row("P43", "Polaris + Stage 0 has lower TTA@80% than Polaris",
            S1m.lower(tta("S2_polaris_s0"), tta("S2_polaris")), t("S2_polaris_s0"),
            t("S2_polaris")),
    ]


# ---------------------------------------------------------------------------
# 6.5 curves
# ---------------------------------------------------------------------------

def plot(data, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("main arms", [c for _, c in MAIN] + ["S2_gpu"]),
              ("plug-in pairs", ["S2_random", "S2_random_s0", "S2_wen_fresh", "S2_wen_fresh_s0",
                                 "S2_wen_fresh_gpu", "S2_wen_fresh_gpu_s0", "S2_polaris",
                                 "S2_polaris_s0"])]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for row, (title, cells) in zip(axes, groups):
        for cell in cells:
            m = data[cell]
            ls = "--" if cell.endswith("_s0") else "-"
            row[0].plot([t for _, t, _ in m["curve"]], [a for _, _, a in m["curve"]],
                        lw=1, ls=ls, label=NAME[cell])
            row[1].plot([e for e, _ in m["energy_curve"]], [a for _, a in m["energy_curve"]],
                        lw=1, ls=ls, label=NAME[cell])
        for ax, xl in ((row[0], "online clock (s)"),
                       (row[1], "cumulative ground-truth energy, failed included (kJ)")):
            for thr in THR:
                ax.axhline(thr, color="grey", lw=0.6, ls=":")
            ax.set_xlabel(xl)
            ax.set_ylabel("test accuracy")
            ax.set_ylim(0.3, 0.95)
            ax.set_title(f"S2, seed {SEED}: {title}")
            ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "evidence" / "s2_s4126_tables.md")
    args = parser.parse_args()
    data = {cell: metrics(cell) for _, cell in ARMS}
    curves = args.out.with_name(args.out.stem + "_curves.png")
    plot(data, curves)
    lines = ["## 6.1 S2 table, seed 4126", "",
             "### 6.1a Time, energy, STR, airtime, staleness", "", *table_main(data), "",
             "### 6.1b Selection behaviour and Stage 0", "", *table_behaviour(data), "",
             "## 6.2 Pairwise directions, Ours vs each arm (seed 4126, sign only)", "",
             "`<` = Ours lower (faster / less energy); STR `>` = Ours higher. A threshold Ours "
             "reaches and the other does not counts as Ours lower.", "", *pairwise(data), "",
             "## 6.3 Plug-in pairs (B − A; `x → y` where one side did not reach)", "",
             *plugin_pairs(data), "",
             "## 6.4 P-table (seed 4126)", "", *p_table(data), "",
             "## 6.5 Curves", "", f"![S2 curves]({curves.name})", ""]
    args.out.write_text("\n".join(lines))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
