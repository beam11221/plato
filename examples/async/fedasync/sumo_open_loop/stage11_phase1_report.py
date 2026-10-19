"""
Stage 11 (partial recurrence, p_return): the Step 3 set and the curve.

    python3 stage11_phase1_report.py                 # Phase 1: seed 4126 -> STAGE11_seed4126_report.md
    python3 stage11_phase1_report.py --seeds 4126 4127 4128 --out STAGE11_analysis.md

Endpoints (STAGE11_p_return_sweep.md):
  p = 0     Stage 10 Wave A open cells (stage10f1_s<seed>, stage10nf_s<seed>);
            identical identity sequence to p_return = 0 (Step 0 test)
  p = 1     anchor runs (gpu+F, baseline+F) compared with Stage 10 closed;
            if both land within +-3.5 %, Stage 10 closed is the p = 1 end,
            otherwise the anchor is (gap A); observed / cpu / none use
            Stage 10 closed at p = 1 (no anchor exists for them)

Per run: the Stage 10 set (analyze_stage9.per_run + analyze_stage10.filter_stats)
plus, split by occupant kind (commuter = identity C1..C20, transient = trace vehicle):
  dispatch share per tier per window; median G_i at dispatch per tier
  (rounds 2,001-4,000, all and seen) and the xavier : thor ratio; observations
  held at dispatch (mean / median / p90 / share >= 30); ever-dispatched per tier
  and never-dispatched; commuter share of dispatches vs of allocations.

Reading rules: a gap inside +-3.5 % is "inside noise", not a sign; with one
seed every verdict is provisional and nothing is quoted as a claim.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_stage7 as a7  # noqa: E402
import analyze_stage9 as a9  # noqa: E402
import analyze_stage10 as a10  # noqa: E402

TIERS = ("xavier", "orin", "thor")
WINDOWS = ((1, 1000), (1001, 2000), (2001, 4000))
PS = (0.25, 0.5, 0.75)
NOISE = 3.5
LOGS = HERE / "logs"
EVIDENCE = HERE.parent / "gpu_tiers_dataset" / "stage11_evidence"
ALLOC = re.compile(r"\[Stage11\] alloc t=(\S+) slot=(\d+) trace_veh_id=(\S+) identity_id=(\S+) "
                   r"kind=(\S+) tier=(\S+) state_carried=(\d) exhausted=(\d)")
SUMMARY = re.compile(r"\[Stage11\] summary p_return=(\S+) allocations=(\d+) commuter_allocations=(\d+) "
                     r"realised_share=(\S+) exhaustions=(\d+) commuter_reentries=(\d+) commuter_first=(\d+) "
                     r"transients=(\d+) identity_digest=(\S+)")
ARMS = ("gpu", "baseline", "observed", "cpu")


def pk(p):
    return f"p{round(p * 100):03d}"


def is_commuter(identity):
    return bool(identity) and re.fullmatch(r"C\d+", identity) is not None


# ---------------------------------------------------------------------------
# per run
# ---------------------------------------------------------------------------

def run_key(arm, p, seed, filtered=True):
    """(prefix, cell) for a Stage 11 or endpoint run."""
    if p == 0.0:
        return (f"stage10f1_s{seed}", f"{arm}_open") if filtered else (f"stage10nf_s{seed}", "baseline_open")
    if p == "closed":
        return (f"stage10f1_s{seed}", f"{arm}_closed") if filtered else (f"stage10nf_s{seed}", "baseline_closed")
    return (f"stage11{'f' if filtered else 'nf'}_{pk(p)}_s{seed}", f"{arm}_ret")


def by_kind(pfx, cell, run):
    """Step 3's commuter / transient split for one Stage 11 run."""
    allocs, summ = [], None
    for line in (LOGS / f"{pfx}_{cell}.log").open(errors="ignore"):
        m = ALLOC.search(line)
        if m:
            allocs.append({"t": float(m.group(1)), "identity": m.group(4), "kind": m.group(5), "tier": m.group(6)})
            continue
        m = SUMMARY.search(line)
        if m:
            summ = {"p_return": float(m.group(1)), "allocations": int(m.group(2)),
                    "commuter_allocations": int(m.group(3)), "realised_share": float(m.group(4)),
                    "exhaustions": int(m.group(5)), "commuter_reentries": int(m.group(6)),
                    "transients": int(m.group(8)), "identity_digest": m.group(9)}
    rep = [d for d in run.dispatches if d.raw_in_range is not None]
    kinds = {"commuter": [d for d in rep if is_commuter(d.identity_id)],
             "transient": [d for d in rep if not is_commuter(d.identity_id)]}
    last_t = max((d.start for d in rep), default=0.0)
    count = Counter(d.identity_id for d in rep)
    out = {"summary": summ,
           "commuter_share_of_dispatches": len(kinds["commuter"]) / len(rep) if rep else None,
           "commuter_share_of_allocations": summ["realised_share"] if summ else None}
    for kind, ds in kinds.items():
        k = {"dispatches": len(ds)}
        k["share"] = {}
        for lo, hi in ((1, 4000),) + WINDOWS:
            w = [d for d in ds if lo <= d.round <= hi]
            c = Counter(d.tier for d in w)
            k["share"][f"{lo}-{hi}"] = {t: c[t] / len(w) for t in TIERS} if w else {}
        late = [d for d in ds if d.round > 2000 and not math.isnan(d.g_i)]
        gi = {}
        for t in TIERS:
            all_ = [d.g_i for d in late if d.tier == t]
            seen = [d.g_i for d in late if d.tier == t and d.observations >= 1]
            gi[t] = {"all": st.median(all_) if all_ else None, "seen": st.median(seen) if seen else None,
                     "n": len(all_)}
        for basis in ("all", "seen"):
            x, th = gi["xavier"][basis], gi["thor"][basis]
            gi[f"xavier_thor_{basis}"] = x / th if x and th else None
        k["gi_late"] = gi
        obs = [d.observations for d in ds]
        k["observations"] = {"mean": st.mean(obs) if obs else None, "median": st.median(obs) if obs else None,
                             "p90": a7.percentile(obs, 0.9), "zero": sum(o == 0 for o in obs) / len(obs) if obs else None,
                             "ge30": sum(o >= 30 for o in obs) / len(obs) if obs else None,
                             "ge70": sum(o >= 70 for o in obs) / len(obs) if obs else None}
        # membership: every identity of this kind allocated before the horizon
        idents = {}
        for a in allocs:
            if a["kind"] == kind and a["t"] < last_t - 30.0:
                idents.setdefault(a["identity"], a["tier"])
        tiers = defaultdict(list)
        for i, t in idents.items():
            tiers[t].append(i)
        k["identities"] = {t: len(v) for t, v in tiers.items()}
        k["ever_dispatched"] = {t: sum(i in count for i in v) / len(v) for t, v in tiers.items() if v}
        k["never_dispatched"] = (sum(i not in count for i in idents) / len(idents)) if idents else None
        counts = [count.get(i, 0) for i in idents]
        k["dispatches_per_identity"] = {"mean": st.mean(counts) if counts else None, "jain": a9.jain(counts) if counts else None}
        out[kind] = k
    return out


def analyse(pfx, cell, arm, stage11):
    if not a9.available(pfx, cell):
        return None
    res = a9.per_run(pfx, cell, arm)
    res.pop("_recycle_seq", None)
    a7.PREFIX = pfx
    run = a7.parse(cell)
    if "nf" not in pfx.split("_")[0]:
        res["filter"] = a10.filter_stats(pfx, cell)
    if stage11:
        res["by_kind"] = by_kind(pfx, cell, run)
    return res


# ---------------------------------------------------------------------------
# curve, gaps, predictions
# ---------------------------------------------------------------------------

def pct(a, b):
    return 100.0 * (a - b) / b if a is not None and b is not None else None


def sign(v):
    if v is None:
        return "no data"
    return "inside noise" if abs(v) < NOISE else ("gpu faster" if v < 0 else "gpu slower")


def crossing(points):
    """Linear interpolation between the adjacent points that bracket zero."""
    pts = [(p, g) for p, g in points if g is not None]
    for (p0, g0), (p1, g1) in zip(pts, pts[1:]):
        if g0 == 0:
            return p0
        if (g0 < 0) != (g1 < 0):
            return p0 + (p1 - p0) * (0 - g0) / (g1 - g0)
    return None


def seed_analysis(seed, runs):
    T = {k: (r["T4000"] if r else None) for k, r in runs.items()}
    curve = {}
    for arm in ARMS:
        row = {"0": T.get((arm, 0.0)), "closed_stage10": T.get((arm, "closed"))}
        for p in PS:
            row[str(p)] = T.get((arm, p))
        if arm in ("gpu", "baseline"):
            row["1.0_anchor"] = T.get((arm, 1.0))
        curve[arm] = row
    curve["none"] = {"0": T.get(("none", 0.0)), "closed_stage10": T.get(("none", "closed")),
                     **{str(p): T.get(("none", p)) for p in PS}}

    anchor = {arm: pct(T.get((arm, 1.0)), T.get((arm, "closed"))) for arm in ("gpu", "baseline")}
    anchor_ok = all(v is not None and abs(v) <= NOISE for v in anchor.values())

    def T_at(arm, p):
        if p == 1.0:
            if arm in ("gpu", "baseline") and not anchor_ok:
                return T.get((arm, 1.0))
            return T.get((arm, "closed"))
        return T.get((arm, p))

    grid = (0.0,) + PS + (1.0,)
    gaps = {}
    for name, a, b in (("A", "gpu", "baseline"), ("B", "gpu", "observed"), ("C", "observed", "baseline")):
        gaps[name] = {str(p): pct(T_at(a, p), T_at(b, p)) for p in grid}
    g0, g1 = gaps["A"]["0.0"], gaps["A"]["1.0"]
    linear = {str(p): ((1 - p) * g0 + p * g1) if g0 is not None and g1 is not None else None for p in grid}
    departure = {str(p): (gaps["A"][str(p)] - linear[str(p)]) if gaps["A"][str(p)] is not None and linear[str(p)] is not None else None
                 for p in grid}
    p_star = crossing([(p, gaps["A"][str(p)]) for p in grid])
    p_star_linear = (g0 / (g0 - g1)) if g0 is not None and g1 is not None and (g0 < 0) != (g1 < 0) else None
    gaps["A_alt_p1"] = {"anchor": pct(T.get(("gpu", 1.0)), T.get(("baseline", 1.0))),
                        "stage10_closed": pct(T.get(("gpu", "closed")), T.get(("baseline", "closed")))}
    return {"curve": curve, "anchor_vs_stage10_closed_pct": anchor, "anchor_within_noise": anchor_ok,
            "p1_source_for_A": "Stage 10 closed" if anchor_ok else "anchor",
            "gaps": gaps, "A_linear": linear, "A_departure_from_linear": departure,
            "p_star": p_star, "p_star_linear": p_star_linear}


def predictions(sa, runs):
    A, B, C = sa["gaps"]["A"], sa["gaps"]["B"], sa["gaps"]["C"]
    grid = ["0.0", "0.25", "0.5", "0.75", "1.0"]
    out = {}
    a = [A[p] for p in grid]
    mono = all(x is not None for x in a) and all(y >= x for x, y in zip(a, a[1:]))
    ends = a[0] is not None and -15 <= a[0] <= -9 and a[-1] is not None and 4 <= a[-1] <= 11
    out[1] = ("held" if mono and ends else "not held" if a[0] is not None else "no data",
              f"A = {fmt_list(a)}; monotone {mono}; p = 0 in [-15, -9]: {a[0] is not None and -15 <= a[0] <= -9}; p = 1 in [+4, +11]: {a[-1] is not None and 4 <= a[-1] <= 11}")
    dep = sa["A_departure_from_linear"]
    above = [dep[p] is not None and dep[p] > 0 for p in ("0.25", "0.5")]
    inside = [dep[p] is not None and abs(dep[p]) < NOISE for p in ("0.25", "0.5")]
    ps = sa["p_star"]
    in_band = ps is not None and 0.4 <= ps <= 0.65
    v2 = ("held" if all(above) and in_band else
          "inside noise" if all(inside) else "not held")
    pstar_s = "no crossing" if ps is None else f"{ps:.3f}"
    plin_s = "—" if sa["p_star_linear"] is None else f"{sa['p_star_linear']:.3f}"
    out[2] = (v2, f"departure from linear at 0.25 / 0.5 = {fmt(dep['0.25'])} / {fmt(dep['0.5'])} pts; "
                  f"p* = {pstar_s} (linear line crosses at {plin_s}); predicted [0.4, 0.65]")
    rc, rt = [], []
    for p in PS + (1.0,):
        r = runs.get(("gpu", p))
        bk = (r or {}).get("by_kind") or {}
        rc.append(((bk.get("commuter") or {}).get("gi_late") or {}).get("xavier_thor_seen"))
        rt.append(((bk.get("transient") or {}).get("gi_late") or {}).get("xavier_thor_seen"))
    rises = all(x is not None for x in rc) and all(y >= x for x, y in zip(rc, rc[1:]))
    exceeds = all(c is not None and t is not None and c > t for c, t in zip(rc[:3], rt[:3]))
    out[3] = ("held" if rises and exceeds else "not held" if any(x is not None for x in rc) else "no data",
              f"gpu+F commuter xavier:thor (seen, rounds 2,001-4,000) at 0.25/0.5/0.75/1 = {fmt_list(rc, 2)}; "
              f"transient = {fmt_list(rt, 2)}; rises {rises}; exceeds transients at every p<1 {exceeds}")
    b0 = sa["curve"]["baseline"]["0"]
    dev = [pct(sa["curve"]["baseline"].get(str(p)), b0) for p in PS] + [pct(sa["curve"]["baseline"].get("1.0_anchor"), b0)]
    flat = all(d is not None and abs(d) <= NOISE for d in dev)
    out[4] = ("held" if flat else "not held" if any(d is not None for d in dev) else "no data",
              f"baseline+F vs p=0 at 0.25/0.5/0.75/1(anchor) = {fmt_list(dev)} %")
    b = [B[p] for p in grid[:4]]
    shrink = all(x is not None for x in b) and all(abs(y) <= abs(x) for x, y in zip(b, b[1:]))
    und = b[3] is not None and abs(b[3]) < NOISE
    out[5] = ("held" if shrink and und else "not held" if b[0] is not None else "no data",
              f"B at 0/0.25/0.5/0.75 = {fmt_list(b)}; |B| monotone non-increasing {shrink}; inside noise at 0.75 {und}")
    c = [C[p] for p in grid]
    neg = [x for x in c if x is not None and x < -NOISE]
    out[6] = ("held" if not neg else "not held", f"C at 0..1 = {fmt_list(c)}")
    an = sa["anchor_vs_stage10_closed_pct"]
    out[7] = ("held" if sa["anchor_within_noise"] else "not held",
              f"anchor vs Stage 10 closed: gpu+F {fmt(an['gpu'])} %, baseline+F {fmt(an['baseline'])} %")
    pl = {f"{k[0]}@{k[1]}": r["plateau"] for k, r in runs.items() if r and isinstance(k[1], float) and k[1] > 0}
    bad = {k: v for k, v in pl.items() if not 0.817 <= v <= 0.836}
    out[8] = ("held" if not bad else "not held",
              f"Stage 11 plateaus {100 * min(pl.values()):.2f}-{100 * max(pl.values()):.2f} %"
              + (f"; outside: {', '.join(f'{k} {100 * v:.2f}' for k, v in bad.items())}" if bad else "")
              if pl else "no data")
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def fmt(v, nd=2):
    return "—" if v is None else f"{v:+.{nd}f}"


def fmt_list(v, nd=2):
    return " / ".join("—" if x is None else f"{x:+.{nd}f}" if nd == 2 else f"{x:.{nd}f}" for x in v)


def pc(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def write_report(seeds, data, out_md, provisional):
    L = []
    title = "Stage 11 — seed 4126 (Phase 1)" if provisional else "Stage 11 — p_return sweep"
    L += [f"# {title}", "",
          "Generated by `sumo_open_loop/stage11_phase1_report.py` from the run logs. "
          "Spec: `prompts/STAGE11_p_return_sweep.md`.", ""]
    if provisional:
        L += ["> **Provisional (1 seed).** Phase 1 exists to catch a wrong design early. "
              "Nothing here is a claim. A gap smaller than ±3.5 % (execution noise accepted in Stage 10) "
              "is \"inside noise\", not a sign.", ""]
    for seed in seeds:
        d = data[seed]
        sa, runs = d["seed"], d["runs"]
        L += [f"## Seed {seed}", "", "### Soundness (Stage 11 runs)", "",
              "| run | T@4000 (s) | plateau | conservation | waste | fallback | pool after Stage 0 | commuter share (alloc) | exhaustions | dispatches to commuters |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for (arm, p), r in sorted(((k, v) for k, v in runs.items() if isinstance(k[1], float) and k[1] > 0),
                                  key=lambda kv: (kv[0][1], kv[0][0])):
            if not r:
                L.append(f"| {arm}{'' if arm == 'none' else '+F'} @ {p} | not finished | | | | | | | | |")
                continue
            f = r.get("filter") or {}
            bk = r.get("by_kind") or {}
            s = bk.get("summary") or {}
            L.append(f"| {arm}{'' if arm == 'none' else '+F'} @ {p} | {r['T4000']:.1f} | {100 * r['plateau']:.2f}% | "
                     f"{100 * r['conservation_rel_error']:+.2f}% | {pc(r['waste'])} | {pc(f.get('fallback_rate'))} | "
                     f"{f.get('pool_after_median', '—')} | {pc(s.get('realised_share'))} | {s.get('exhaustions', '—')} | "
                     f"{pc(bk.get('commuter_share_of_dispatches'))} |")
        L += ["", "### The curve: T @ 4,000 (s)", "",
              "| arm | p = 0 (Stage 10 open) | 0.25 | 0.5 | 0.75 | 1.0 anchor | Stage 10 closed |", "|---|---|---|---|---|---|---|"]
        for arm, row in sa["curve"].items():
            cells = [row.get("0"), row.get("0.25"), row.get("0.5"), row.get("0.75"), row.get("1.0_anchor"), row.get("closed_stage10")]
            L.append(f"| {arm}{'' if arm == 'none' else '+F'} | " + " | ".join("—" if c is None else f"{c:.1f}" for c in cells) + " |")
        an = sa["anchor_vs_stage10_closed_pct"]
        L += ["", f"**Anchor check** (p = 1 vs Stage 10 closed): gpu+F {fmt(an['gpu'])} %, baseline+F {fmt(an['baseline'])} % → "
              f"{'within ±3.5 %: Stage 10 closed is the p = 1 end' if sa['anchor_within_noise'] else 'outside ±3.5 %: the anchor is the p = 1 end of gap A'}.", "",
              "### Gap curves (%, negative = first arm faster)", "",
              "| gap | p = 0 | 0.25 | 0.5 | 0.75 | 1.0 |", "|---|---|---|---|---|---|"]
        for name, label in (("A", "A gpu+F − baseline+F"), ("B", "B gpu+F − observed+F"), ("C", "C observed+F − baseline+F")):
            g = sa["gaps"][name]
            L.append(f"| {label} | " + " | ".join(
                "—" if g[p] is None else f"{g[p]:+.2f} ({sign(g[p]) if name == 'A' else 'inside noise' if abs(g[p]) < NOISE else 'beyond noise'})"
                for p in ("0.0", "0.25", "0.5", "0.75", "1.0")) + " |")
        L.append("| A linear mixing line | " + " | ".join("—" if sa["A_linear"][p] is None else f"{sa['A_linear'][p]:+.2f}" for p in ("0.0", "0.25", "0.5", "0.75", "1.0")) + " |")
        L.append("| A − linear (pts; + = worse for gpu) | " + " | ".join("—" if sa["A_departure_from_linear"][p] is None else f"{sa['A_departure_from_linear'][p]:+.2f}" for p in ("0.0", "0.25", "0.5", "0.75", "1.0")) + " |")
        alt = sa["gaps"]["A_alt_p1"]
        pstar = "no crossing" if sa["p_star"] is None else f"{sa['p_star']:.3f}"
        plin = "—" if sa["p_star_linear"] is None else f"{sa['p_star_linear']:.3f}"
        L += ["", f"Gap A at p = 1 by source: anchor {fmt(alt['anchor'])} %, Stage 10 closed {fmt(alt['stage10_closed'])} % "
              f"(used: {sa['p1_source_for_A']}).", "",
              f"**Crossing p\\*** (measured, linear interpolation): **{pstar}**; linear mixing line crosses at {plin}.", ""]

        L += ["### By occupant kind (Step 3)", ""]
        for arm in ("gpu", "observed", "baseline", "cpu"):
            L += [f"#### {arm}+F", "",
                  "| p | kind | dispatches | share x / o / t (all) | x / o / t rounds 2,001–4,000 | G_i median x / o / t (late, seen) | x : t (seen) | x : t (all) | obs at dispatch mean / median / p90 | obs ≥ 30 | ever-dispatched x / o / t | never | identities |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for p in PS + ((1.0,) if arm in ("gpu", "baseline") else ()):
                r = runs.get((arm, p))
                if not r:
                    continue
                bk = r["by_kind"]
                for kind in ("commuter", "transient"):
                    k = bk[kind]
                    sh, shl = k["share"]["1-4000"], k["share"]["2001-4000"]
                    gi, ob, ev = k["gi_late"], k["observations"], k["ever_dispatched"]
                    L.append(f"| {p} | {kind} | {k['dispatches']} | "
                             + " / ".join(pc(sh.get(t)) for t in TIERS) + " | "
                             + " / ".join(pc(shl.get(t)) for t in TIERS) + " | "
                             + " / ".join("—" if gi[t]["seen"] is None else f"{gi[t]['seen']:.3f}" for t in TIERS) + " | "
                             + ("—" if gi["xavier_thor_seen"] is None else f"{gi['xavier_thor_seen']:.2f}") + " | "
                             + ("—" if gi["xavier_thor_all"] is None else f"{gi['xavier_thor_all']:.2f}") + " | "
                             + ("—" if ob["mean"] is None else f"{ob['mean']:.1f} / {ob['median']:.0f} / {ob['p90']:.0f}") + " | "
                             + pc(ob["ge30"]) + " | "
                             + " / ".join(pc(ev.get(t)) for t in TIERS) + " | " + pc(k["never_dispatched"]) + " | "
                             + " / ".join(str(k["identities"].get(t, 0)) for t in TIERS) + " |")
                L.append(f"| {p} | commuter share | dispatches {pc(bk['commuter_share_of_dispatches'])} vs allocations {pc(bk['commuter_share_of_allocations'])} | | | | | | | | | | |")
            L.append("")
        L += ["Stage 3 reference: k\\* ≈ 30–70 observations (the information a score needs). "
              "`obs ≥ 30` is the share of dispatches made with at least that much history on the identity.", "",
              "### Predictions" + (" — provisional (1 seed)" if provisional else ""), "",
              "| # | verdict | evidence |", "|---|---|---|"]
        for n, (v, ev) in d["predictions"].items():
            L.append(f"| {n} | {v}{' (provisional)' if provisional else ''} | {ev} |")
        L.append("")
    out_md.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[4126])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    provisional = args.seeds == [4126]
    out_md = args.out or EVIDENCE / ("STAGE11_seed4126_report.md" if provisional else "STAGE11_analysis.md")
    data = {}
    for seed in args.seeds:
        runs = {}
        for arm in ARMS:
            runs[(arm, 0.0)] = analyse(*run_key(arm, 0.0, seed), arm, False)
            runs[(arm, "closed")] = analyse(*run_key(arm, "closed", seed), arm, False)
            for p in PS + ((1.0,) if arm in ("gpu", "baseline") else ()):
                runs[(arm, p)] = analyse(*run_key(arm, p, seed), arm, True)
        runs[("none", 0.0)] = analyse(*run_key("baseline", 0.0, seed, False), "baseline", False)
        runs[("none", "closed")] = analyse(*run_key("baseline", "closed", seed, False), "baseline", False)
        for p in PS:
            runs[("none", p)] = analyse(*run_key("baseline", p, seed, False), "baseline", True)
        sa = seed_analysis(seed, runs)
        data[seed] = {"runs": runs, "seed": sa, "predictions": predictions(sa, runs)}
        missing = [f"{a}@{p}" for (a, p), r in runs.items() if r is None]
        print(f"seed {seed}: {sum(r is not None for r in runs.values())} runs analysed; missing: {missing or 'none'}")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    write_report(args.seeds, data, out_md, provisional)
    js = out_md.with_suffix(".json")
    js.write_text(json.dumps({str(s): {"seed": d["seed"], "predictions": {str(k): v for k, v in d["predictions"].items()},
                                       "runs": {f"{a}@{p}": r for (a, p), r in d["runs"].items()}}
                              for s, d in data.items()}, indent=2, default=str))
    print(f"wrote {out_md} and {js.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
