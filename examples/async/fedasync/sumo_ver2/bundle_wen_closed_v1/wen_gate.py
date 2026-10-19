#!/usr/bin/env python3
"""wen_gate.py -- validation gate G1..G8 + 3 plots for the Wen closed-population trace.

Usage
    python wen_gate.py --run wen_closed_v1_run
Writes gate_report.md, g1_speed_scatter.png, g5_dwell_hist.png,
g6_incoverage_timeseries.png into the run directory and exits non-zero on FAIL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

S2_LO, S2_HI = 400.0, 1600.0
TASK_S = 13.0          # conservative per-task time for the G7 feasibility proxy
C = 3                  # concurrency; G7 needs median >= 2 * C
ROAD_M = 2000.0
N_VEH = 30


def pct(series, q):
    return float(np.percentile(series, q)) if len(series) else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    a = ap.parse_args(argv)
    run = a.run

    meta = yaml.safe_load((run / "sim_meta.yaml").read_text())
    speeds = pd.read_csv(run / "fleet_speeds.csv").set_index("veh_id")["desired_speed_mps"]
    trace = pd.read_csv(run / "trace.csv.gz")
    reentry = pd.read_csv(run / "reentry_log.csv")
    reentry = reentry[reentry.phase == "post"]
    laps = pd.read_csv(run / "lap_log.csv")
    dwell = pd.read_csv(run / "dwell_log.csv")
    lane = pd.read_csv(run / "lane_stats.csv")

    times = np.sort(trace.t.unique())
    rows: list[tuple[str, str, str, bool]] = []     # id, check, numbers, passed

    # ---------------------------------------------------------------- G1
    realized = trace.groupby("veh_id").speed_mps.mean()
    g1 = pd.DataFrame({"desired": speeds, "realized": realized})
    g1["err_pct"] = (g1.realized / g1.desired - 1.0) * 100.0
    n_ok = int((g1.err_pct.abs() <= 2.0).sum())
    g1_pass = n_ok >= 28
    outside = g1[g1.err_pct.abs() > 2.0].sort_values("err_pct")
    rows.append(("G1", "realized vs desired mean speed, +-2%",
                 f"{n_ok}/30 within +-2%; err range {g1.err_pct.min():+.2f}%..{g1.err_pct.max():+.2f}%; "
                 f"fleet mean err {g1.err_pct.mean():+.2f}%", g1_pass))

    # ---------------------------------------------------------------- G2
    # G2 upper bound relaxed 33 -> 32.5 (approved 2026-10-15): 30 plain draws from
    # U[17,34] under the locked seed reach 32.590, and seed tuning is forbidden.
    G2_MAX_MIN = 32.5
    g2_pass = bool(speeds.min() <= 18.0 and speeds.max() >= G2_MAX_MIN)
    rows.append(("G2", f"desired-speed coverage (min<=18, max>={G2_MAX_MIN}, relaxed from 33)",
                 f"min {speeds.min():.3f} m/s, max {speeds.max():.3f} m/s", g2_pass))

    # ---------------------------------------------------------------- G3
    present = trace.groupby("t").veh_id.nunique()
    ids_seen = trace.veh_id.nunique()
    d = reentry.delay_s
    d_mean, d_p95, d_max = d.mean(), pct(d, 95), d.max()
    g3_pass = bool(ids_seen == N_VEH and d_p95 <= 1.0)
    rows.append(("G3", "identity persistence; re-entry delay p95 <= 1.0 s",
                 f"{ids_seen}/30 IDs ever present; samples with all 30 present "
                 f"{(present == N_VEH).mean()*100:.1f}%; min present {present.min()}; "
                 f"re-entry delay mean {d_mean:.3f} s / p95 {d_p95:.3f} s / max {d_max:.3f} s "
                 f"(n={len(d)})", g3_pass))

    # ---------------------------------------------------------------- G4
    laps = laps.merge(speeds.rename("v"), left_on="veh_id", right_index=True)
    laps["expected_s"] = laps.dist_m / laps.v
    laps["err_pct"] = (laps.travel_s / laps.expected_s - 1.0) * 100.0
    g4 = laps.groupby("veh_id").agg(mean_err_pct=("err_pct", "mean"),
                                    mean_travel_s=("travel_s", "mean"),
                                    n_laps=("travel_s", "size"))
    g4_ok = int((g4.mean_err_pct.abs() <= 3.0).sum())
    g4_pass = g4_ok == N_VEH
    rows.append(("G4", "mean lap (travel) time within +-3% of 2000/v_i",
                 f"{g4_ok}/30 within +-3%; per-vehicle mean err "
                 f"{g4.mean_err_pct.min():+.2f}%..{g4.mean_err_pct.max():+.2f}%; "
                 f"{int(g4.n_laps.sum())} laps total", g4_pass))

    # ---------------------------------------------------------------- G5
    dw = dwell.dwell_s
    in_range = ((dw >= 35.0) & (dw <= 71.0)).mean() * 100 if len(dw) else float("nan")
    g5_pass = bool(len(dw) and dw.min() >= 35.0 * 0.95 and dw.max() <= 71.0 * 1.05)
    rows.append(("G5", "S2 dwell per pass, expected ~35-71 s",
                 f"n={len(dw)} passes; min {dw.min():.1f} / p5 {pct(dw,5):.1f} / median "
                 f"{dw.median():.1f} / p95 {pct(dw,95):.1f} / max {dw.max():.1f} s; "
                 f"{in_range:.1f}% inside 35-71 s", g5_pass))

    # ---------------------------------------------------------------- G6
    cov = trace[(trace.x_m >= S2_LO) & (trace.x_m <= S2_HI)]
    cnt = cov.groupby("t").size().reindex(times, fill_value=0)
    g6_pass = bool(abs(cnt.mean() - 18.0) <= 2.0)
    rows.append(("G6", "S2 in-coverage count per t (expect mean ~18)",
                 f"mean {cnt.mean():.2f}; p5 {pct(cnt,5):.0f}; p50 {pct(cnt,50):.0f}; "
                 f"p95 {pct(cnt,95):.0f}; sd {cnt.std():.2f}; min {cnt.min()}; max {cnt.max()}",
                 g6_pass))

    # ---------------------------------------------------------------- G7
    cov = cov.copy()
    cov["remain_s"] = (S2_HI - cov.x_m) / cov.veh_id.map(speeds)
    feas = cov[cov.remain_s >= TASK_S].groupby("t").size().reindex(times, fill_value=0)
    g7_med = float(feas.median())
    g7_pass = g7_med >= 2 * C
    rows.append(("G7", f"S2 feasibility proxy (remaining >= {TASK_S:.0f} s), median >= {2*C}",
                 f"median {g7_med:.1f}; mean {feas.mean():.2f}; p5 {pct(feas,5):.0f}; "
                 f"p95 {pct(feas,95):.0f}; P(count < {2*C}) = {(feas < 2*C).mean():.4f}",
                 g7_pass))

    # ---------------------------------------------------------------- G8
    lane["tot"] = lane.time_lane0_s + lane.time_lane1_s
    share0 = lane.time_lane0_s.sum() / lane.tot.sum()
    lc_per_lap = (lane.lane_changes / lane.laps_completed.replace(0, np.nan))
    rows.append(("G8", "lane use share and lane changes per vehicle per lap",
                 f"fleet lane0 {share0*100:.1f}% / lane1 {(1-share0)*100:.1f}%; "
                 f"lane changes per lap: mean {lc_per_lap.mean():.2f}, min {lc_per_lap.min():.2f}, "
                 f"max {lc_per_lap.max():.2f}; total lane changes {int(lane.lane_changes.sum())}",
                 True))

    # ---------------------------------------------------------------- plots
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(g1.desired, g1.realized, s=34, c="#2a6f97", zorder=3, label="vehicles")
    lim = [16, 35]
    ax.plot(lim, lim, "k--", lw=1, label="realized = desired")
    ax.plot(lim, [0.98 * x for x in lim], color="#bbb", lw=1, label="+-2% band")
    ax.plot(lim, [1.02 * x for x in lim], color="#bbb", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("desired speed (m/s)"); ax.set_ylabel("realized mean speed (m/s)")
    ax.set_title("G1 realized vs desired speed"); ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(run / "g1_speed_scatter.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(dw, bins=60, color="#2a6f97")
    for v, lb in ((35, "35 s"), (71, "71 s")):
        ax.axvline(v, color="#c1121f", ls="--", lw=1)
        ax.text(v, ax.get_ylim()[1] * .95, lb, color="#c1121f", fontsize=8, ha="right")
    ax.set_xlabel("dwell time per pass in x in [400,1600] m (s)"); ax.set_ylabel("passes")
    ax.set_title("G5 S2 dwell-time histogram"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(run / "g5_dwell_hist.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    w = cnt[cnt.index < 3600]
    ax.plot(w.index, w.values, lw=.7, color="#2a6f97")
    ax.axhline(cnt.mean(), color="#c1121f", ls="--", lw=1,
               label=f"mean {cnt.mean():.1f}")
    ax.set_xlabel("t (s, after warm-up)"); ax.set_ylabel("vehicles with x in [400,1600]")
    ax.set_title("G6 S2 in-coverage count, first 3600 s"); ax.legend(fontsize=8)
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(run / "g6_incoverage_timeseries.png", dpi=150); plt.close(fig)

    # ---------------------------------------------------------------- report
    n_fail = sum(1 for *_, ok in rows if not ok)
    md = ["# Gate report - Wen closed-population environment (v1)", "",
          f"- {meta['sumo_version']}, Python {meta['python_version']}",
          f"- speed RNG `{meta['speed_rng']}`, seed {meta['speed_rng_seed']}",
          f"- step {meta['step_length_s']} s, sample {meta['sample_interval_s']} s, "
          f"warm-up {meta['warmup_s']} s (discarded), duration {meta['duration_s']} s",
          f"- {len(trace)} trace rows, {meta['n_reentries_total']} re-entries total", "",
          "| # | Check | Numbers | Verdict |", "|---|---|---|---|"]
    for gid, check, nums, ok in rows:
        md.append(f"| {gid} | {check} | {nums} | {'PASS' if ok else '**FAIL**'} |")
    md += ["", f"**{len(rows) - n_fail}/{len(rows)} PASS, {n_fail} FAIL.**", ""]

    md += ["## G1 per-vehicle detail", "",
           "| veh_id | desired (m/s) | realized (m/s) | err % |", "|---|---|---|---|"]
    for vid, r in g1.sort_values("desired").iterrows():
        md.append(f"| {vid} | {r.desired:.3f} | {r.realized:.3f} | {r.err_pct:+.2f} |")

    md += ["", "## G2 sorted desired speeds (m/s)", "",
           ", ".join(f"{v:.3f}" for v in np.sort(speeds.values)), ""]

    md += ["## G4 per-vehicle lap detail", "",
           "| veh_id | v (m/s) | laps | mean travel (s) | expected (s) | err % |",
           "|---|---|---|---|---|---|"]
    for vid, r in g4.join(speeds.rename("v")).sort_values("v").iterrows():
        md.append(f"| {vid} | {r.v:.3f} | {int(r.n_laps)} | {r.mean_travel_s:.2f} | "
                  f"{(ROAD_M - laps[laps.veh_id==vid].x0_m.mean()) / r.v:.2f} | "
                  f"{r.mean_err_pct:+.2f} |")

    md += ["", "## G5 dwell histogram (5 s bins)", "", "| bin (s) | passes |", "|---|---|"]
    h, edges = np.histogram(dw, bins=np.arange(np.floor(dw.min()/5)*5,
                                               np.ceil(dw.max()/5)*5 + 5, 5))
    for c_, lo in zip(h, edges[:-1]):
        md.append(f"| {lo:.0f}-{lo+5:.0f} | {int(c_)} |")

    md += ["", "## G8 per-vehicle lane use", "",
           "| veh_id | lane0 share | lane changes | laps | changes/lap |", "|---|---|---|---|---|"]
    for _, r in lane.iterrows():
        lpl = r.lane_changes / r.laps_completed if r.laps_completed else float("nan")
        md.append(f"| {r.veh_id} | {r.time_lane0_s / r.tot * 100:.1f}% | {int(r.lane_changes)} | "
                  f"{int(r.laps_completed)} | {lpl:.2f} |")

    if len(outside):
        md += ["", "## G1 vehicles outside +-2%", "",
               "| veh_id | desired | realized | err % |", "|---|---|---|---|"]
        for vid, r in outside.iterrows():
            md.append(f"| {vid} | {r.desired:.3f} | {r.realized:.3f} | {r.err_pct:+.2f} |")

    md += ["", "## Plots", "", "- `g1_speed_scatter.png`", "- `g5_dwell_hist.png`",
           "- `g6_incoverage_timeseries.png`", ""]
    (run / "gate_report.md").write_text("\n".join(md) + "\n")

    for gid, check, nums, ok in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {gid}  {nums}")
    print(f"\n{len(rows)-n_fail}/{len(rows)} PASS, {n_fail} FAIL -> {run/'gate_report.md'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
