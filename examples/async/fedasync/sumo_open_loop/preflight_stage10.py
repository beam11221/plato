"""
Stage 10, Step 2: pre-flight for the open-population Stage 0 gate.

Runs the Stage 7b checks (`preflight_stage7b.check_run`) and the Stage 9 checks
(`preflight_stage9.stage9_checks`) unchanged, then checks EVERY candidate the
gate considered, recomputing each value independently of the server:

  tier       == the occupant's tier (trace vehicle at p=0, slot's fleet
                vehicle at p=1)
  T_hat      tier_spec: one value per tier; gpu / baseline == the GPU spec
                (15 / 1.923077 / 1.0 s); cpu == its own spec (the per-tier
                values its startup log prints from tiers_cpu.csv)
             observed_per_vehicle: == the identity's own mean of delivered
                durations so far, or the fleet mean of all delivered durations
                (8.323077 s before the first) if it has none -- recomputed from
                the report lines' `duration=` field
  R_i        == |x_exit - x(t)| / v_max from the trace vehicle's own 1 Hz
                position at the logged t
  pass       == (R_i >= T_hat)
  round line eligible_after_stage0 == passes; fallback == (passes == 0)
  selected   every selected slot passed, unless the round fell back; its
             logged T_hat / R_i == the candidate line's

and reports pool sizes after each gate (median / p10 / p90), the fallback rate
and the rejection rate.  Stop conditions: any check < 100%, fallback > 10% of
rounds, or median eligible_after_stage0 < C = 4.

Usage:
    python3 preflight_stage10.py --prefix stage10f1_s4126_preflight --alpha 1 [runs ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import preflight_stage9  # noqa: E402
from occupant_identity import load_fleet  # noqa: E402
from open_stage0 import FLEET_MEAN_T_HAT_S, OpenStage0Config, load_trace_positions, remaining_range_s  # noqa: E402
from preflight_stage7b import check_run  # noqa: E402

FEDASYNC = HERE.parent
TRACE_DIR = HERE / "sumo_trace" / "e1_open_poisson_seed4126_range400"
FLEET = load_fleet(FEDASYNC / "sumo_close_loop" / "input" / "clients_n20r400.csv",
                   FEDASYNC / "sumo_close_loop" / "output" / "client_sleep_times_n20r400.csv")
GEOM = OpenStage0Config("tier_spec", 400.0, 19.444444, 600.0, str(TRACE_DIR / "trace.csv"))
GPU_T_HAT = {"xavier": 15.0, "orin": 1.923077, "thor": 1.0}
C = 4
RUNS = [f"{a}_{r}" for r in ("open", "closed") for a in ("baseline", "gpu", "cpu", "observed")]

CAND = re.compile(
    r"\[Stage0\] cand round=(-?\d+) t=(\S+) slot=(\d+) veh_id=(\S+) identity_id=(\S+) "
    r"tier=(\S+) t_hat=(\S+) t_hat_basis=(\S+) r_i=(\S+) pass=([01])")
ROUND = re.compile(
    r"\[Stage0\] round=(-?\d+) t_hat_source=\S+ r_i_mode=\S+ eligible_after_range_gate=(\d+) "
    r"eligible_after_stage0=(\d+) stage0_rejected_ids=\S* ?.*stage0_fallback=([01])")
SEL = re.compile(r"\[Stage0\] round=(-?\d+) selected_id=(\d+) selected_tier=(\S+) "
                 r"t_hat_used=(\S+) R_i_selected=(\S+) t_hat_basis=(\S+)")
REPORT = re.compile(r"\[Stage7\] report round=\d+ slot=\d+ veh_id=\S+ tier=\S+ .*uploaded=([01]) "
                    r".* identity_id=(\S+) duration=(\S+)")
CPU_TIER = re.compile(r"\[PolarisVi\] T_round per-tier: (\S+)\s+median=(\S+) s")

POSITIONS = None
TRACE_TIER = {}
WINDOW_EXIT = {}


class Row:
    def __init__(self):
        self.ok = self.total = 0
        self.example = ""

    def add(self, good, example=""):
        self.total += 1
        self.ok += bool(good)
        if not good and not self.example:
            self.example = example


def expected_tier(p_recur, slot, veh):
    return FLEET[slot][0] if p_recur == 1.0 else TRACE_TIER[veh]


def stage10_checks(log: Path, p_recur: float, arm: str) -> tuple[dict, dict]:
    rows = defaultdict(Row)
    source = "observed_per_vehicle" if arm == "observed" else "tier_spec"
    spec = dict(GPU_T_HAT)
    if arm == "cpu":
        spec = {}
    own_sum, own_n = defaultdict(float), defaultdict(int)
    fleet_sum, fleet_n = 0.0, 0
    # One gate invocation = its CAND lines, then one ROUND line, then its SEL
    # lines (all from the single server process, so never interleaved).  Track
    # by invocation, not by round number: one round can have several SEL lines,
    # and a round number can repeat after an empty round.
    current: dict[int, tuple] = {}
    n_pass = 0
    phase = "sel"
    pool_range, pool_after, fallbacks, rounds = [], [], 0, 0
    spec_seen = defaultdict(set)

    for line in log.open(errors="ignore"):
        m = CPU_TIER.search(line)
        if m and arm == "cpu":
            spec.setdefault(m.group(1), float(m.group(2)))
            continue
        m = REPORT.search(line)
        if m:
            if m.group(1) == "1":
                ident, dur = m.group(2), float(m.group(3))
                own_sum[ident] += dur
                own_n[ident] += 1
                fleet_sum += dur
                fleet_n += 1
            continue
        m = CAND.search(line)
        if m:
            rnd, t, slot, veh, ident, tier, t_hat, basis, r_i, ok = m.groups()
            rnd, t, slot, t_hat, ok = int(rnd), float(t), int(slot), float(t_hat), ok == "1"
            where = f"round {rnd} slot {slot} veh {veh}"
            rows["[10] filter tier == occupant tier"].add(tier == expected_tier(p_recur, slot, veh),
                                                        f"{where}: {tier}")
            if source == "tier_spec":
                spec_seen[tier].add(round(t_hat, 6))
                exp = spec.get(tier)
                rows["[10] T_hat == spec value for occupant tier"].add(
                    exp is not None and math.isclose(t_hat, exp, abs_tol=1e-5) and basis == "spec",
                    f"{where}: {t_hat} vs {exp}")
            else:
                if own_n[ident]:
                    exp, eb = own_sum[ident] / own_n[ident], "own"
                else:
                    exp, eb = (fleet_sum / fleet_n if fleet_n else FLEET_MEAN_T_HAT_S), "fleet"
                rows["[10] T_hat == observed_per_vehicle recomputed"].add(
                    math.isclose(t_hat, exp, abs_tol=1e-5) and basis == eb,
                    f"{where}: {t_hat} ({basis}) vs {exp:.6f} ({eb})")
            r_exp = remaining_range_s(GEOM, POSITIONS, veh, t)
            if r_i == "nan" or r_exp is None:
                # No bracketing 1 Hz sample: both sides must agree it is
                # undefined, and it may only happen in the last second before
                # the vehicle's window closes (past its last in-coverage sample),
                # where the true R_i < 1 s is below every T_hat anyway.
                rows["[10] R_i == vmax_distance from trace position"].add(
                    r_i == "nan" and r_exp is None, f"{where}: {r_i} vs {r_exp}")
                rows["[10] undefined R_i only in a window's last second"].add(
                    WINDOW_EXIT[veh] - t <= 1.0 + 1e-9, f"{where}: {WINDOW_EXIT[veh] - t:.3f} s before exit")
            else:
                rows["[10] R_i == vmax_distance from trace position"].add(
                    math.isclose(float(r_i), r_exp, abs_tol=1e-5), f"{where}: {r_i} vs {r_exp}")
            r_val = float(r_i) if r_i != "nan" else None
            rows["[10] pass == (R_i >= T_hat)"].add(
                ok == (r_val is not None and r_val >= t_hat), f"{where}: pass={ok} R={r_i} T={t_hat}")
            if phase != "cand":
                current, n_pass, phase = {}, 0, "cand"
            current[slot] = (t_hat, r_val, ok)
            n_pass += int(ok)
            continue
        m = ROUND.search(line)
        if m:
            rnd, before, after, fb = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4) == "1"
            rounds += 1
            pool_range.append(before)
            pool_after.append(after)
            fallbacks += fb
            rows["[10] round line: eligible_after_stage0 == passes"].add(
                phase == "cand" and after == n_pass and before == len(current),
                f"round {rnd}: {after}/{before} vs {n_pass}/{len(current)}")
            rows["[10] round line: fallback == (no pass)"].add(fb == (after == 0), f"round {rnd}")
            phase = "round"
            continue
        m = SEL.search(line)
        if m:
            rnd, slot = int(m.group(1)), int(m.group(2))
            phase = "sel"
            c = current.get(slot)
            fell_back = n_pass == 0
            rows["[10] selected passed (or round fell back)"].add(
                c is not None and (c[2] or fell_back), f"round {rnd} slot {slot}")
            rows["[10] selected T_hat / R_i == candidate line"].add(
                c is not None and math.isclose(float(m.group(4)), c[0], abs_tol=1e-6)
                and (m.group(5) == "nan" if c[1] is None else math.isclose(float(m.group(5)), c[1], abs_tol=1e-6)),
                f"round {rnd} slot {slot}")

    if source == "tier_spec":
        rows["[10] exactly one T_hat per tier"].add(all(len(v) == 1 for v in spec_seen.values()),
                                                    str(dict(spec_seen)))

    def pct(v, q):
        v = sorted(v)
        return v[min(len(v) - 1, int(q * len(v)))] if v else None
    pools = {
        "rounds": rounds,
        "after_range_gate": {"median": st.median(pool_range) if pool_range else None,
                             "p10": pct(pool_range, .1), "p90": pct(pool_range, .9)},
        "after_stage0": {"median": st.median(pool_after) if pool_after else None,
                         "p10": pct(pool_after, .1), "p90": pct(pool_after, .9)},
        "fallback_rate": fallbacks / rounds if rounds else None,
        "reject_rate": 1 - sum(pool_after) / sum(pool_range) if sum(pool_range) else None,
    }
    return rows, pools


def main() -> int:
    global POSITIONS
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="stage10f1_s4126_preflight")
    ap.add_argument("--alpha", type=float, choices=[1.0, 10.0], default=1.0)
    ap.add_argument("runs", nargs="*", default=RUNS)
    args = ap.parse_args()

    with (TRACE_DIR / "coverage_windows.csv").open() as fh:
        for r in csv.DictReader(fh):
            TRACE_TIER[r["veh_id"]] = r["tier"]
            WINDOW_EXIT[r["veh_id"]] = float(r["t_exit_s"])
    POSITIONS = load_trace_positions(TRACE_DIR / "trace.csv")
    preflight_stage9.ALPHA = args.alpha

    failed, out = False, {}
    for name in args.runs:
        arm, regime = name.split("_")
        p_recur = 1.0 if regime == "closed" else 0.0
        log = HERE / "logs" / f"{args.prefix}_{name}.log"
        res = check_run(log, p_recur, proactive=arm in ("gpu", "cpu"), polaris=arm != "baseline")
        rows = {k: (v["ok"], v["total"], (v["examples"] or [""])[0]) for k, v in res["checks"].items()}
        s9 = preflight_stage9.stage9_checks(log, p_recur, arm)
        if args.alpha == 10.0:
            # Polaris default package: const 0.01 unseen tau -> 7b's row applies;
            # the Stage 9 running-mean rows do not.
            s9 = {k: r for k, r in s9.items() if "observed" not in k and "tau_unseen" not in k}
        else:
            rows.pop("tau_hat == tau_obs (observed arm)", None)
        for k, r in s9.items():
            rows[k] = (r.ok, r.total, r.example)
        s10, pools = stage10_checks(log, p_recur, arm)
        for k, r in s10.items():
            rows[k] = (r.ok, r.total, r.example)
        rows["[10] fallback < 10% of rounds"] = (int((pools["fallback_rate"] or 0) < 0.10), 1,
                                                 f"{pools['fallback_rate']}")
        rows["[10] median eligible_after_stage0 >= C"] = (
            int((pools["after_stage0"]["median"] or 0) >= C), 1, f"{pools['after_stage0']}")
        print(f"\n=== {args.prefix}_{name}: {res['dispatches']} dispatches, {pools['rounds']} gated rounds")
        print(f"    pool after range gate {pools['after_range_gate']}  after Stage 0 {pools['after_stage0']}"
              f"  fallback {100 * (pools['fallback_rate'] or 0):.1f}%  rejected {100 * (pools['reject_rate'] or 0):.1f}%")
        for k, (ok, tot, ex) in sorted(rows.items()):
            good = tot > 0 and ok == tot
            failed |= not good
            if not good or k.startswith("[10]"):
                print(f"   [{'PASS' if good else 'FAIL'}] {k:58s} {ok:5d}/{tot:<5d}" + ("" if good else f"  e.g. {ex}"))
        n_other = sum(1 for k in rows if not k.startswith("[10]"))
        n_other_ok = sum(1 for k, (ok, tot, _) in rows.items() if not k.startswith("[10]") and tot and ok == tot)
        print(f"   Stage 7b + 9 checks: {n_other_ok}/{n_other} at 100%")
        out[name] = {"checks": {k: [ok, tot] for k, (ok, tot, _) in rows.items()}, "pools": pools}

    dest = FEDASYNC / "gpu_tiers_dataset" / "stage10_evidence" / f"{args.prefix}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}\n{'PRE-FLIGHT FAILED' if failed else 'PRE-FLIGHT PASSED: all checks 100%'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
