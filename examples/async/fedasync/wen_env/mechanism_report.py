"""
Prompt M A4: mechanism report for S1 (how each selector decides), from the
Prompt M mechanism logs. Mechanical: numbers and plots, no interpretation.

    PYTHONPATH=<repo>:sumo_close_loop python wen_env/mechanism_report.py \
        --gpu runs/wen_env/S1_gpu_s4126_pm --polaris runs/wen_env/S1_polaris_s4126_pm \
        --wen runs/wen_env/S1_wen_fresh_s4126_pm --random runs/wen_env/S1_random_s4126_pm \
        --out wen_env/evidence/mechanism_s1_s4126.md

`--maps-only --tag M3` draws plots 1, 6 and 9 only (Prompt M B4.4).

Axes shared by every arm: tau_hat_ours (rounds) and G_ours, the passive values
Ours' scorer would compute (wen_mechanism_log.py). Static counterfactuals
re-decide each LOGGED decision under another rule; they do not feed back into
G, F3 or the clock, so they show which term moves single picks, not what a run
under that rule would do.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import wen_main_report as R  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from wen_faithful import non_dominated, normalise  # noqa: E402

TIERS = ("xavier", "orin", "thor")
TIER_COLOR = {"xavier": "#c0392b", "orin": "#2471a3", "thor": "#239b56"}
BIN = 500


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def gini(counts) -> float:
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def counts30(picks) -> np.ndarray:
    c = np.zeros(30)
    for p in picks:
        c[int(p) - 1] += 1
    return c


def windows(run: Path) -> list[tuple[str, int, int]]:
    curve = R.accuracy_curve(run)
    last = curve[-1][0]
    hit = next((r for r, _, a in curve if a >= 0.90), None)
    end = hit if hit else last
    label = "last 200 before 90 %" if hit else f"last 200 (90 % not reached; r{last})"
    return [("rounds 1–200", 1, 200), ("rounds 1,000–1,200", 1000, 1200),
            (label, max(1, end - 199), end)]


def md_table(header, rows) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def f(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def per_vehicle_spearman(cand: pd.DataFrame, cols: dict[str, str]) -> list:
    picks = cand[cand.picked == 1].groupby("client").size().reindex(range(1, 31), fill_value=0)
    rows = []
    for label, col in cols.items():
        means = cand.groupby("client")[col].mean().reindex(range(1, 31))
        rho, p = spearmanr(picks.values, means.values, nan_policy="omit")
        rows.append([label, f(rho), f"{p:.2g}"])
    return rows


# ---------------------------------------------------------------------------
# candidate maps (plots 1, 6, 9, 14)
# ---------------------------------------------------------------------------

def _map_panel(ax, cand, x, y, title, color=None, cmap="viridis", picked_split=None,
               iso_v=None, clabel=None):
    ax.scatter(cand[x], cand[y], s=3, c="#cccccc", alpha=0.35, lw=0, label="candidates")
    pk = cand[cand.picked == 1]
    if picked_split is not None:
        a, b = pk[pk[picked_split] == 0], pk[pk[picked_split] == 1]
        ax.scatter(a[x], a[y], s=10, c="#c0392b", lw=0, label="picked (exploit)")
        ax.scatter(b[x], b[y], s=10, c="#2471a3", lw=0, label="picked (explored)")
    elif color is not None:
        sc = ax.scatter(cand[x], cand[y], s=4, c=cand[color], cmap=cmap, lw=0, alpha=0.6)
        ax.scatter(pk[x], pk[y], s=14, facecolors="none", edgecolors="k", lw=0.6,
                   label="picked")
        plt.colorbar(sc, ax=ax, label=clabel or color)
    else:
        ax.scatter(pk[x], pk[y], s=10, c="#c0392b", lw=0, label="picked")
    if iso_v is not None:
        xs = np.linspace(max(cand[x].min(), 0), cand[x].max(), 100)
        for v in iso_v:
            ax.plot(xs, v * np.sqrt(1 + xs), "k--", lw=0.7)
        ax.plot([], [], "k--", lw=0.7, label="iso-V (V_pick q25/50/75)")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(fontsize=6, loc="best")


def map_three_windows(cand, run, x, y, path, suptitle, **kw):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (label, lo, hi) in zip(axes, windows(run)):
        sub = cand[(cand["round"] >= lo) & (cand["round"] <= hi)]
        iso = None
        if kw.get("iso_from"):
            vp = sub[(sub.picked == 1) & (sub.explored == 0)]["V"]
            iso = list(np.percentile(vp, [25, 50, 75])) if len(vp) else None
        _map_panel(ax, sub, x, y, label, iso_v=iso,
                   **{k: v for k, v in kw.items() if k != "iso_from"})
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ours
# ---------------------------------------------------------------------------

def ours(run: Path, figdir: Path, tag: str, maps_only=False) -> list[str]:
    cand = load(run / "gpu_candidates.csv")
    out = [f"## Ours (`gpu`) — `{run.name}`", ""]
    p1 = figdir / f"{tag}01_ours_map.png"
    map_three_windows(cand, run, "tau_hat", "G", p1,
                      "Ours: candidates on (tau_hat, G); iso-V curves G = V·sqrt(1+tau_hat)",
                      picked_split="explored", iso_from=True)
    out += [f"**1. Candidate map** ![]({p1.name})", ""]
    if maps_only:
        return out

    ex = cand[(cand.picked == 1) & (cand.explored == 0)].copy()
    ex["bin"] = (ex["round"] - 1) // BIN * BIN + 1
    cat = np.where(ex.is_min_tau_hat & ex.is_max_G, "both",
                   np.where(ex.is_min_tau_hat == 1, "min tau_hat only",
                            np.where(ex.is_max_G == 1, "max G only", "neither")))
    ex["cat"] = cat
    tab = pd.crosstab(ex["bin"], ex["cat"], normalize="index")
    order = [c for c in ("min tau_hat only", "both", "max G only", "neither") if c in tab]
    tab = tab[order]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    tab.plot(kind="bar", stacked=True, ax=ax, width=0.85,
             color=["#2471a3", "#7d3c98", "#c0392b", "#aaaaaa"][:len(order)])
    ax.set_xlabel(f"round bin (start, {BIN} rounds)")
    ax.set_ylabel("share of exploit picks")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p2 = figdir / f"{tag}02_ours_pick_type.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    whole = ex["cat"].value_counts(normalize=True)
    out += ["**2. Exploit picks: the min-tau_hat candidate, the max-G candidate, both, "
            f"neither** ![]({p2.name})", "",
            *md_table(["bin start"] + order,
                      [[b] + [f(tab.loc[b, c], 2) for c in order] for b in tab.index]),
            "", "Whole run: " + ", ".join(f"{k} {v:.2f}" for k, v in whole.items()) +
            f" (exploit picks {len(ex):,}; explored "
            f"{int(((cand.picked == 1) & (cand.explored == 1)).sum()):,}).", ""]

    # 3. static counterfactual
    actual, by_tau, by_G, by_V = [], [], [], []
    change = {"tau_hat only": 0, "G only": 0, "V (exploit)": 0}
    n_ex = 0
    for _, d in cand.groupby("decision_id", sort=True):
        pick = int(d.loc[d.picked == 1, "client"].iloc[0])
        explored = int(d.loc[d.picked == 1, "explored"].iloc[0]) == 1
        t = int(d.sort_values(["tau_hat", "client"]).client.iloc[0])
        g = int(d.assign(m=-d.G).sort_values(["m", "client"]).client.iloc[0])
        v = int(d.assign(m=-d.V).sort_values(["m", "client"]).client.iloc[0])
        actual.append(pick)
        if explored:
            by_tau.append(pick), by_G.append(pick), by_V.append(pick)
            continue
        n_ex += 1
        by_tau.append(t), by_G.append(g), by_V.append(v)
        change["tau_hat only"] += t != pick
        change["G only"] += g != pick
        change["V (exploit)"] += v != pick
    rows = [["actual", "—", f(gini(counts30(actual)))]]
    for (k, n), picks in zip(change.items(), (by_tau, by_G, by_V)):
        rows.append([k, f(n / max(n_ex, 1)), f(gini(counts30(picks)))])
    out += ["**3. Static counterfactual on the logged exploit decisions** (explored "
            "decisions keep their pick in every rule)", "",
            *md_table(["rule", "pick-change share (exploit decisions)", "dispatch Gini"],
                      rows), ""]

    # 4. per-tier G / tau_hat over time, tier share
    cand["bin100"] = (cand["round"] - 1) // 100 * 100 + 50
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for tier in TIERS:
        s = cand[cand.tier == tier]
        if s.empty:
            continue
        for ax, col in zip(axes[:2], ("G", "tau_hat")):
            g = s.groupby("bin100")[col]
            ax.plot(g.median().index, g.mean(), color=TIER_COLOR[tier], label=tier)
            ax.fill_between(g.median().index, g.quantile(0.25), g.quantile(0.75),
                            color=TIER_COLOR[tier], alpha=0.2)
    picks = cand[cand.picked == 1]
    share = pd.crosstab(picks["bin100"], picks["tier"], normalize="index")
    for tier in TIERS:
        if tier in share:
            axes[2].plot(share.index, share[tier], color=TIER_COLOR[tier], label=tier)
    for ax, t in zip(axes, ("G (mean, IQR) per tier", "tau_hat (mean, IQR) per tier",
                            "tier share of picks")):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("round")
        ax.legend(fontsize=7)
    fig.tight_layout()
    p4 = figdir / f"{tag}04_ours_tiers_over_time.png"
    fig.savefig(p4, dpi=130)
    plt.close(fig)
    out += [f"**4. Per-tier G and tau_hat over time, tier share** ![]({p4.name})", ""]

    # 5. per vehicle
    out += ["**5. Per vehicle: dispatch count vs mean G and mean tau_hat (over its "
            "candidate rows)**", "",
            *md_table(["vs", "Spearman rho", "p"],
                      per_vehicle_spearman(cand, {"mean G": "G", "mean tau_hat": "tau_hat"})),
            ""]
    return out


# ---------------------------------------------------------------------------
# Polaris
# ---------------------------------------------------------------------------

def polaris(run: Path, figdir: Path, tag: str, maps_only=False) -> list[str]:
    cand = load(run / "polaris_candidates.csv")
    first = cand[cand.slot_k == 0].copy()
    first["pool"] = first.groupby("decision_id").client.transform("size")
    first["q_rel"] = first["q_i"] * first["pool"]
    cand = cand.merge(first[["decision_id", "client", "q_rel"]],
                      on=["decision_id", "client"], how="left")
    out = [f"## Polaris — `{run.name}`", ""]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _map_panel(axes[0], cand.dropna(subset=["q_rel"]), "tau_hat_ours", "G_i",
               "x = passive tau_hat_ours", color="q_rel", cmap="coolwarm",
               clabel="q_i x |pool| (1 = uniform)")
    _map_panel(axes[1], cand.dropna(subset=["q_rel"]), "tau_i", "G_i",
               "x = observed tau_i (staleness + 0.1)", color="q_rel", cmap="coolwarm",
               clabel="q_i x |pool| (1 = uniform)")
    fig.tight_layout()
    p6 = figdir / f"{tag}06_polaris_map.png"
    fig.savefig(p6, dpi=130)
    plt.close(fig)
    out += [f"**6. Candidate maps, colour = q_i relative to uniform** ![]({p6.name})", ""]
    if maps_only:
        return out

    dec = load(run / "polaris_decisions.csv")
    G = dec["G"].map(lambda s: np.array(eval(s)))
    tau = dec["tau"].map(lambda s: np.array(eval(s)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    axes[0].hist(first["q_rel"], bins=80, color="#7d3c98")
    axes[0].axvline(1, color="k", ls="--", lw=0.8)
    axes[0].set_title("q_i x |pool| per candidate (1 = uniform)", fontsize=9)
    axes[1].plot(dec["round"], dec["eff_pool"], lw=0.5, label="effective pool 1/sum q²")
    axes[1].plot(dec["round"], dec["pool"].map(lambda s: len(eval(s))), lw=0.5,
                 label="|pool|")
    axes[1].set_title("effective pool size", fontsize=9)
    axes[1].legend(fontsize=7)
    axes[2].plot(dec["round"], G.map(lambda a: a.max() / a.min()), lw=0.5, label="max/min G")
    axes[2].plot(dec["round"], tau.map(lambda a: a.max() / a.min()), lw=0.5,
                 label="max/min tau")
    axes[2].set_yscale("log")
    axes[2].set_title("spread per call", fontsize=9)
    axes[2].legend(fontsize=7)
    for ax in axes[1:]:
        ax.set_xlabel("round")
    fig.tight_layout()
    p7 = figdir / f"{tag}07_polaris_q.png"
    fig.savefig(p7, dpi=130)
    plt.close(fig)
    q = first["q_rel"]
    out += [f"**7. q vs uniform, effective pool, spread** ![]({p7.name})", "",
            f"q_i x |pool|: median {q.median():.3f}, 5–95 % {q.quantile(.05):.3f}–"
            f"{q.quantile(.95):.3f}; max q per call median {dec['max_q'].median():.4f}; "
            f"effective pool / |pool| median "
            f"{(dec['eff_pool'] / dec['pool'].map(lambda s: len(eval(s)))).median():.3f}.",
            ""]
    out += ["**8. Per vehicle: dispatch count vs mean G, tau, tau_hat_ours**", "",
            *md_table(["vs", "Spearman rho", "p"], per_vehicle_spearman(
                cand, {"mean G_i": "G_i", "mean tau_i": "tau_i",
                       "mean tau_hat_ours": "tau_hat_ours"})), ""]
    return out


# ---------------------------------------------------------------------------
# Wen
# ---------------------------------------------------------------------------

def wen_frames(run: Path) -> pd.DataFrame:
    w = load(run / "wen_candidates.csv").rename(columns={"decision": "decision_id",
                                                          "client_id": "client"})
    p = load(run / "passive_candidates.csv")
    return w.merge(p[["decision_id", "client", "tau_hat_ours", "G_ours", "V_ours"]],
                   on=["decision_id", "client"], how="left", validate="one_to_one")


def wen_decide(rows: list[tuple[int, tuple]], keep: tuple[int, ...]) -> int:
    pts = {c: tuple(o[i] for i in keep) for c, o in rows}
    pareto = non_dominated(pts)
    norm = normalise({c: pts[c] for c in pareto})
    score = {c: sum(v) / len(v) for c, v in norm.items()}
    return min(pareto, key=lambda c: (score[c], c))


def wen(run: Path, figdir: Path, tag: str, maps_only=False, t_e=None) -> list[str]:
    cand = wen_frames(run)
    out = [f"## Wen et al. (`wen_fresh`) — `{run.name}`", ""]
    p9 = figdir / f"{tag}09_wen_map.png"
    cand["score_plot"] = cand["score"]
    map_three_windows(cand.dropna(subset=["score_plot"]).assign(
        picked=lambda d: d.picked), run, "tau_hat_ours", "G_ours", p9,
        "Wen: Pareto members on the passive (tau_hat_ours, G_ours) axes; colour = "
        "Wen's score (lower = better); non-members not shown", color="score_plot",
        cmap="viridis_r", clabel="score (mean of nF1–nF4)")
    out += [f"**9. Candidate map** ![]({p9.name})", ""]
    if maps_only:
        return out

    pk = cand[cand.picked == 1].copy()
    pk["bin"] = (pk["round"] - 1) // BIN * BIN + 1
    share = pk.groupby("bin")[[f"is_min_F{i}" for i in range(1, 5)]].mean()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    names = {1: "F1 time", 2: "F2 energy", 3: "F3 own-data loss", 4: "F4 label KL"}
    for i in range(1, 5):
        ax.plot(share.index, share[f"is_min_F{i}"], marker="o", label=names[i])
    ax.set_ylabel("share of picks that are the min")
    ax.set_xlabel(f"round bin (start, {BIN})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p10 = figdir / f"{tag}10_wen_min_share.png"
    fig.savefig(p10, dpi=130)
    plt.close(fig)
    whole = pk[[f"is_min_F{i}" for i in range(1, 5)]].mean()
    out += [f"**10. Share of picks that are the minimum of each objective** ![]({p10.name})",
            "", "Whole run: " + ", ".join(f"{names[i]} {whole[f'is_min_F{i}']:.2f}"
                                          for i in range(1, 5)) + ".", ""]

    # 11. F3 loop
    ev = load(run / "wen_fresh_eval.csv")
    counts = pk.groupby("client").size().reindex(range(1, 31), fill_value=0)
    top5 = list(counts.sort_values(ascending=False).index[:5])
    top10 = list(counts.sort_values(ascending=False).index[:10])
    rest = [c for c in range(1, 31) if c not in top5]
    f3top = ev[[f"F3_{c}" for c in top5]].mean(axis=1)
    f3rest = ev[[f"F3_{c}" for c in rest]].mean(axis=1)
    kl = cand.groupby("client")["F4"].first()
    kl_rank = kl.rank(method="min")
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    axes[0].plot(ev["round"], f3top, label=f"top-5 most picked {top5}")
    axes[0].plot(ev["round"], f3rest, label="other 25")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("F3 (fresh loss, mean)")
    axes[0].legend(fontsize=7)
    axes[1].bar([str(c) for c in top10], [kl_rank[c] for c in top10], color="#7d3c98")
    axes[1].set_ylabel("F4 rank among 30 (1 = lowest KL)")
    axes[1].set_xlabel("top-10 most-picked vehicles (most first)")
    fig.tight_layout()
    p11 = figdir / f"{tag}11_wen_f3_f4.png"
    fig.savefig(p11, dpi=130)
    plt.close(fig)
    q = [0.0, 0.1, 0.5, 1.0]
    pts = [int(x * (len(ev) - 1)) for x in q]
    out += [f"**11. F3 feedback loop and F4 rank** ![]({p11.name})", "",
            *md_table(["model version at", "round", "F3 top-5", "F3 other 25", "ratio"],
                      [[f"{int(x * 100)} %", int(ev['round'][i]), f(f3top[i], 4),
                        f(f3rest[i], 4), f(f3rest[i] / f3top[i], 1)] for x, i in zip(q, pts)]),
            "", f"Top-10 most picked {top10}: dispatches "
            f"{[int(counts[c]) for c in top10]} of {int(counts.sum())}; F4 ranks "
            f"{[int(kl_rank[c]) for c in top10]}.", ""]

    # 12. F2 by tier, constraint cuts
    present = [t for t in TIERS if (cand.tier == t).any()]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.boxplot([cand.loc[cand.tier == t, "F2"] for t in present], tick_labels=present)
    if t_e:
        ax.axhline(t_e[1], color="r", ls="--", lw=0.8, label=f"E_max {t_e[1]:.2f} J")
        ax.legend(fontsize=7)
    ax.set_ylabel("F2 (J), Wen CPU model")
    fig.tight_layout()
    p12 = figdir / f"{tag}12_wen_f2_tier.png"
    fig.savefig(p12, dpi=130)
    plt.close(fig)
    dec = load(run / "wen_decisions.csv")
    rows = []
    for t in present:
        n = dec[f"n_candidates_{t}"].sum()
        ct = dec[f"n_cut_T_only_{t}"].sum()
        ce = dec[f"n_cut_E_only_{t}"].sum()
        cb = dec[f"n_cut_both_{t}"].sum()
        rows.append([t, f"{n:,}", f(ct / n), f(ce / n), f(cb / n), f((ct + ce + cb) / n),
                     f((pk.tier == t).mean())])
    out += [f"**12. F2 by tier and the constraint cuts** ![]({p12.name})", "",
            *md_table(["tier", "candidate rows", "cut by T only", "cut by E only",
                       "cut by both", "total cut", "share of picks"], rows),
            "", f"Fallback decisions: {int(dec['fallback'].sum())}.", ""]

    # 13. static counterfactual
    variants = {"actual rule (F1–F4)": (0, 1, 2, 3), "drop F1": (1, 2, 3),
                "drop F2": (0, 2, 3), "drop F3": (0, 1, 3), "drop F4": (0, 1, 2),
                "F1 + F2 only": (0, 1), "F3 + F4 only": (2, 3)}
    picks = {k: [] for k in variants}
    actual = []
    for _, d in cand.groupby("decision_id", sort=True):
        fb = int(d["fallback"].iloc[0]) == 1
        pool = d if fb else d[d.feasible == 1]
        rows_ = [(int(r.client), (r.F1, r.F2, r.F3, r.F4)) for r in pool.itertuples()]
        actual.append(int(d.loc[d.picked == 1, "client"].iloc[0]))
        for k, keep in variants.items():
            picks[k].append(wen_decide(rows_, keep))
    trows = [["logged picks", "—", f(gini(counts30(actual)))]]
    for k, p in picks.items():
        change = np.mean([a != b for a, b in zip(actual, p)])
        trows.append([k, f(change), f(gini(counts30(p)))])
    out += ["**13. Static counterfactual (same feasible set, re-decided per logged "
            "decision)**", "",
            *md_table(["rule", "pick-change share", "dispatch Gini"], trows), ""]
    return out


# ---------------------------------------------------------------------------
# cross-arm
# ---------------------------------------------------------------------------

def cross(runs: dict[str, Path], figdir: Path, tag: str) -> list[str]:
    frames = {}
    for name, run in runs.items():
        if run is None:
            continue
        files = {"Ours": "gpu_candidates.csv", "Polaris": "polaris_candidates.csv",
                 "Wen et al.": "passive_candidates.csv", "Random": "passive_candidates.csv"}
        frames[name] = load(run / files[name])
    fig, axes = plt.subplots(1, len(frames), figsize=(4.6 * len(frames), 4), sharey=True)
    rows = []
    for ax, (name, c) in zip(np.atleast_1d(axes), frames.items()):
        ax.scatter(c.tau_hat_ours, c.G_ours, s=2, c="#cccccc", alpha=0.3, lw=0)
        pk = c[c.picked == 1]
        for tier in TIERS:
            s = pk[pk.tier == tier]
            ax.scatter(s.tau_hat_ours, s.G_ours, s=5, c=TIER_COLOR[tier], lw=0, label=tier)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("tau_hat_ours (rounds)")
        ax.legend(fontsize=7)
        disp = load(runs[name] / "dispatch_log.csv")
        rows.append([name, f(pk.tau_hat_ours.mean()), f(c.tau_hat_ours.mean()),
                     f(pk.G_ours.mean()), f(c.G_ours.mean()),
                     f(pk.T_hat_train_gpu.mean()), f(pk.T_hat_up_gpu.mean()),
                     f(float(disp.T_train_true.mean())), f(float(disp.T_up_true.mean())),
                     f(float((disp.x_finish - 1000).abs().mean()), 0)])
    np.atleast_1d(axes)[0].set_ylabel("G_ours")
    fig.tight_layout()
    p = figdir / f"{tag}14_cross_arm_map.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return [f"## Cross-arm (14) ![]({p.name})", "",
            "Grey = every candidate the arm saw; dots = its picks, by tier. Axes are the "
            "passive values Ours' scorer would compute, so position on them means the "
            "same in every panel.", "",
            *md_table(["arm", "tau_hat_ours picked", "tau_hat_ours all candidates",
                       "G_ours picked", "G_ours all candidates", "T_hat_train picked (s)",
                       "T_hat_up picked (s)", "true T_train (s)", "true T_up (s)",
                       "mean |x_finish − 1000| (m)"], rows), ""]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=Path)
    ap.add_argument("--polaris", type=Path)
    ap.add_argument("--wen", type=Path)
    ap.add_argument("--random", type=Path)
    ap.add_argument("--wen-limits", type=float, nargs=2, metavar=("T_MAX", "E_MAX"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--maps-only", action="store_true")
    a = ap.parse_args()
    figdir = a.out.parent
    tag = f"{a.out.stem}_{a.tag + '_' if a.tag else ''}"
    lines = [f"# Mechanism report — {a.out.stem} (Prompt M A4; mechanical)", "",
             "Axes: `tau_hat_ours` (rounds) and `G_ours` are the passive values of Ours' "
             "scorer (gpu T̂_train + T̂_up through the queue kernel; G from the update "
             "norm, alpha = 1), logged for every arm. Static counterfactuals re-decide "
             "logged decisions and do not feed back.", ""]
    if a.gpu:
        lines += ours(a.gpu, figdir, tag, a.maps_only)
    if a.polaris:
        lines += polaris(a.polaris, figdir, tag, a.maps_only)
    if a.wen:
        lines += wen(a.wen, figdir, tag, a.maps_only, a.wen_limits)
    if not a.maps_only:
        lines += cross({"Ours": a.gpu, "Polaris": a.polaris, "Wen et al.": a.wen,
                        "Random": a.random}, figdir, tag)
    a.out.write_text("\n".join(lines))
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
