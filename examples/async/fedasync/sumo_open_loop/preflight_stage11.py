"""
Stage 11, Step 1: pre-flight for partial recurrence, read from the smoke logs.

The identity sequence is recomputed INDEPENDENTLY: the real timeline is
replayed with ReturningPopulation seeded from the physics seed, and every
[Stage11] alloc line must match it -- slot, trace vehicle, identity, kind.
Then every dispatch and every Stage 0 candidate is checked against the
identity occupying that slot at that moment:

  identity / tier / sleep / partition        == the occupant identity's
  predictor tier (proactive arms)            == the occupant's tier
  filter tier, T_hat (tier_spec)             == the occupant's tier / spec value
  R_i                                        == vmax_distance from the TRACE vehicle's position
  pass, round-line counts, fallback, selection consistency  (as Stage 10)
  training seed                              == identity_seed(run seed, identity, round)
  state: carried == commuter re-entries; reset == transients + first commuter entries
  occupancy: recycle sequence == Stage 7b baseline_open's (common prefix)
  identity digest: equal across every smoke run at the same p_return

and reports the realised commuter share vs nominal, exhaustion events, and the
tier mix of arrivals by kind.

Usage:  python3 preflight_stage11.py [--seed 4126] [--tag preflight]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_stage7 as a7  # noqa: E402
from occupant_identity import IdentityResolver, identity_seed, load_fleet  # noqa: E402
from open_stage0 import OpenStage0Config, load_trace_positions, remaining_range_s  # noqa: E402
from sumo_open_mobility import OpenCoverageTimeline  # noqa: E402

FEDASYNC = HERE.parent
TRACE_DIR = HERE / "sumo_trace" / "e1_open_poisson_seed4126_range400"
FLEET = load_fleet(FEDASYNC / "sumo_close_loop" / "input" / "clients_n20r400.csv",
                   FEDASYNC / "sumo_close_loop" / "output" / "client_sleep_times_n20r400.csv")
GEOM = OpenStage0Config("tier_spec", 400.0, 19.444444, 600.0, "")
GPU_T_HAT = {"xavier": 15.0, "orin": 1.923077, "thor": 1.0}
PHYS = 4126
P_VALUES = (0.25, 0.5, 0.75, 1.0)

ALLOC = re.compile(r"\[Stage11\] alloc t=(\S+) slot=(\d+) trace_veh_id=(\S+) identity_id=(\S+) "
                   r"kind=(\S+) tier=(\S+) state_carried=([01]) exhausted=([01])")
SUMMARY = re.compile(r"\[Stage11\] summary p_return=(\S+) allocations=(\d+) commuter_allocations=(\d+) "
                     r"realised_share=(\S+) exhaustions=(\d+) commuter_reentries=(\d+) commuter_first=(\d+) "
                     r"transients=(\d+) identity_digest=(\S+)")
DISPATCH = re.compile(r"\[Stage7\] dispatch round=(-?\d+) slot=(\d+) veh_id=(\S+) tier=(\S+) observations=\d+ "
                      r"G_i=\S+ G_i_at_prior=[01] tau_hat=\S+ identity_id=(\S+) pred_tier=(\S+) tau_obs=\S+ "
                      r"partition=(\d+) sleep_s=(\S+)")
CAND = re.compile(r"\[Stage0\] cand round=(-?\d+) t=(\S+) slot=(\d+) veh_id=(\S+) identity_id=(\S+) "
                  r"tier=(\S+) t_hat=(\S+) t_hat_basis=(\S+) r_i=(\S+) pass=([01])")
ROUND = re.compile(r"\[Stage0\] round=(-?\d+) .* eligible_after_range_gate=(\d+) "
                   r"eligible_after_stage0=(\d+) .*stage0_fallback=([01])")
SEL = re.compile(r"\[Stage0\] round=(-?\d+) selected_id=(\d+) selected_tier=(\S+) t_hat_used=(\S+) "
                 r"R_i_selected=(\S+) t_hat_basis=(\S+)")
SEED = re.compile(r"\[ClientSeededTrainer\] client_id=(\d+) identity_id=(\S+) round=(\d+) -> seed (\d+)")
RECYCLE = re.compile(r"\[Stage7\] recycle slot_id=(\d+) .* state_reset=([01])")

POS = None
EXIT = {}


class Row:
    def __init__(self):
        self.ok = self.total = 0
        self.ex = ""

    def add(self, good, ex=""):
        self.total += 1
        self.ok += bool(good)
        if not good and not self.ex:
            self.ex = ex


def replay(p_return: float, n_alloc: int):
    """The identity sequence recomputed from the trace and the physics seed."""
    logging.disable(logging.INFO)
    tl = OpenCoverageTimeline.from_csv(TRACE_DIR / "coverage_windows.csv", 20, 0.0, PHYS)
    res = IdentityResolver(tl, PHYS, 0.0, 20, FLEET, p_return=p_return)
    seq = []
    orig = res.returning.assign

    def assign(slot, veh, tier, base_seed, slots):
        ident, kind, ex = orig(slot, veh, tier, base_seed, slots)
        seq.append((slot, veh, ident.identity_id, kind, int(ex)))
        return ident, kind, ex
    res.returning.assign = assign
    res.attach(tl.allocator)
    t = 0.0
    while len(seq) < n_alloc and tl.peek_next_time() is not None:
        t = tl.peek_next_time()
        tl.advance_to(t)
    logging.disable(logging.NOTSET)
    return seq[:n_alloc], res


def check(prefix: str, run: str, p_return: float, run_seed: int) -> dict:
    arm = run.split("_")[0]
    log = HERE / "logs" / f"{prefix}_{run}.log"
    rows = defaultdict(Row)
    alloc = []
    for line in log.open(errors="ignore"):
        m = ALLOC.search(line)
        if m:
            alloc.append(m.groups())
    seq, res = replay(p_return, len(alloc))
    for (t, slot, veh, ident, kind, tier, carried, ex), exp in zip(alloc, seq):
        rows["[11] alloc == independent replay (slot, veh, identity, kind)"].add(
            (int(slot), veh, ident, kind, int(ex)) == exp, f"{(slot, veh, ident, kind)} vs {exp}")
    rows["[11] alloc count == replay"].add(len(alloc) == len(seq), f"{len(alloc)} vs {len(seq)}")

    # Walk the log in order, tracking the occupant identity of each slot.
    occ: dict[int, tuple] = {}             # slot -> (identity_id, tier, sleep, partition)
    id_info = {f"C{k}": (tier, sleep, k - 1) for k, (tier, sleep) in FLEET.items()}
    seen_commuters, carried_n, reentries, first_c, transients = set(), 0, 0, 0, 0
    resets = 0
    kind_tiers = {"commuter": Counter(), "transient": Counter()}
    cur_cands, n_pass, phase = {}, 0, "sel"
    pools_after, fallbacks, rounds = [], 0, 0
    spec = dict(GPU_T_HAT) if arm in ("gpu", "baseline") else None
    summary = None
    from stage7_population import partition_index, vehicle_sleep_s

    for line in log.open(errors="ignore"):
        m = ALLOC.search(line)
        if m:
            t, slot, veh, ident, kind, tier, carried, ex = m.groups()
            slot = int(slot)
            if kind == "commuter":
                ctier, csleep, cpart = id_info[ident]
                occ[slot] = (ident, ctier, csleep, cpart)
                rows["[11] commuter tier == its fleet tier"].add(tier == ctier, f"{ident} {tier}")
                if ident in seen_commuters:
                    reentries += 1
                    rows["[11] state carried on every commuter re-entry"].add(carried == "1", ident)
                else:
                    first_c += 1
                    rows["[11] first commuter entry starts from the prior"].add(carried == "0", ident)
                seen_commuters.add(ident)
            else:
                ttier = tier
                occ[slot] = (veh, ttier, vehicle_sleep_s(PHYS, veh, ttier), partition_index(veh, 20))
                transients += 1
                rows["[11] transient identity == trace vehicle, no state carried"].add(
                    ident == veh and carried == "0", f"{ident} vs {veh}")
            carried_n += carried == "1"
            kind_tiers[kind][occ[slot][1]] += 1
            continue
        m = RECYCLE.search(line)
        if m:
            resets += m.group(2) == "1"
            continue
        m = DISPATCH.search(line)
        if m:
            rnd, slot, veh, tier, ident, pred_tier, part, sleep_s = m.groups()
            slot = int(slot)
            e = occ.get(slot)
            w = f"round {rnd} slot {slot}"
            rows["[11] dispatch identity == occupant"].add(e is not None and ident == e[0], f"{w}: {ident} vs {e}")
            rows["[11] dispatch tier == occupant tier"].add(e is not None and tier == e[1], f"{w}: {tier} vs {e}")
            rows["[11] dispatch sleep == occupant sleep"].add(
                e is not None and abs(float(sleep_s) - e[2]) < 1e-5, f"{w}: {sleep_s} vs {e}")
            rows["[11] dispatch partition == occupant partition"].add(
                e is not None and int(part) == e[3], f"{w}: {part} vs {e}")
            if arm in ("gpu", "cpu"):
                rows["[11] predictor tier == occupant tier"].add(e is not None and pred_tier == e[1], w)
            continue
        m = CAND.search(line)
        if m:
            rnd, t, slot, veh, ident, tier, t_hat, basis, r_i, ok = m.groups()
            slot, t, t_hat, ok = int(slot), float(t), float(t_hat), ok == "1"
            e = occ.get(slot)
            w = f"round {rnd} slot {slot}"
            rows["[11] filter identity / tier == occupant"].add(
                e is not None and ident == e[0] and tier == e[1], f"{w}: {ident}/{tier} vs {e}")
            if spec is not None:
                rows["[11] T_hat == GPU spec for occupant tier"].add(
                    math.isclose(t_hat, spec[tier], abs_tol=1e-5) and basis == "spec", f"{w}: {t_hat}")
            r_exp = remaining_range_s(GEOM, POS, veh, t)
            if r_i == "nan" or r_exp is None:
                rows["[11] R_i == vmax_distance from the trace vehicle"].add(r_i == "nan" and r_exp is None, w)
                rows["[11] undefined R_i only in a window's last second"].add(EXIT[veh] - t <= 1.0 + 1e-9, w)
            else:
                rows["[11] R_i == vmax_distance from the trace vehicle"].add(
                    math.isclose(float(r_i), r_exp, abs_tol=1e-5), f"{w}: {r_i} vs {r_exp}")
            rv = None if r_i == "nan" else float(r_i)
            rows["[11] pass == (R_i >= T_hat)"].add(ok == (rv is not None and rv >= t_hat), w)
            if phase != "cand":
                cur_cands, n_pass, phase = {}, 0, "cand"
            cur_cands[slot] = (t_hat, rv, ok)
            n_pass += ok
            continue
        m = ROUND.search(line)
        if m:
            after, fb = int(m.group(3)), m.group(4) == "1"
            rows["[11] round line == candidate counts"].add(
                phase == "cand" and after == n_pass and int(m.group(2)) == len(cur_cands), f"round {m.group(1)}")
            rows["[11] fallback == (no pass)"].add(fb == (after == 0), f"round {m.group(1)}")
            pools_after.append(after)
            fallbacks += fb
            rounds += 1
            phase = "round"
            continue
        m = SEL.search(line)
        if m:
            phase = "sel"
            c = cur_cands.get(int(m.group(2)))
            rows["[11] selected passed (or fell back) with the candidate's values"].add(
                c is not None and (c[2] or n_pass == 0) and math.isclose(float(m.group(4)), c[0], abs_tol=1e-6),
                f"round {m.group(1)} slot {m.group(2)}")
            continue
        m = SEED.search(line)
        if m:
            _cid, ident, rnd, seed = m.groups()
            rows["[11] seed == identity_seed(run seed, identity, round)"].add(
                int(seed) == identity_seed(run_seed, ident, int(rnd)), f"{ident} r{rnd}")
            continue
        m = SUMMARY.search(line)
        if m:
            summary = m.groups()

    rows["[11] state resets == transients + first commuter entries (recycle lines)"].add(
        resets == transients + first_c, f"{resets} vs {transients}+{first_c}")
    rows["[11] state carried == commuter re-entries"].add(carried_n == reentries, f"{carried_n} vs {reentries}")
    if rounds:
        rows["[11] fallback < 10% of rounds"].add(fallbacks / rounds < 0.10, f"{fallbacks}/{rounds}")
        rows["[11] median eligible_after_stage0 >= 4"].add(st.median(pools_after) >= 4, f"{st.median(pools_after)}")

    a7.PREFIX = "stage7b"
    ref = a7.parse("baseline_open").recycle_seq
    a7.PREFIX = prefix
    mine = a7.parse(run).recycle_seq
    n = min(len(ref), len(mine))
    rows["[11] occupancy == Stage 7b over the common prefix"].add(ref[:n] == mine[:n], f"prefix {n}")

    n_alloc = len(alloc)
    commuters = sum(1 for a in alloc if a[4] == "commuter")
    return {
        "rows": {k: [r.ok, r.total, r.ex] for k, r in rows.items()},
        "allocations": n_alloc, "commuter_allocations": commuters,
        "realised_share": commuters / n_alloc if n_alloc else None,
        "exhaustions": sum(1 for a in alloc if a[7] == "1"),
        "kind_tiers": {k: dict(v) for k, v in kind_tiers.items()},
        "identity_digest_replay": res.returning.digest() if len(seq) == n_alloc else None,
        "summary_digest": summary[8] if summary else None,
        "pool_after_median": st.median(pools_after) if pools_after else None,
        "fallback_rate": fallbacks / rounds if rounds else None,
    }


def main() -> int:
    global POS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4126)
    ap.add_argument("--tag", default="preflight")
    ap.add_argument("--arms", nargs="+", default=["gpu", "baseline"])
    args = ap.parse_args()
    with (TRACE_DIR / "coverage_windows.csv").open() as fh:
        for r in csv.DictReader(fh):
            EXIT[r["veh_id"]] = float(r["t_exit_s"])
    logging.disable(logging.INFO)
    POS = load_trace_positions(TRACE_DIR / "trace.csv")
    logging.disable(logging.NOTSET)

    failed, out = False, {}
    for p in P_VALUES:
        prefix = f"stage11f_p{int(round(p * 100)):03d}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
        seqs = {}
        for arm in args.arms:
            run = f"{arm}_ret"
            if not (HERE / "logs" / f"{prefix}_{run}.log").exists():
                print(f"{prefix}_{run}: MISSING")
                failed = True
                continue
            res = check(prefix, run, p, args.seed)
            out[f"{prefix}_{run}"] = res
            # Runs stop at different allocation counts, so their digests cover
            # different lengths.  Per run: the logged digest must equal the
            # independent replay's at that run's own count.
            res["rows"]["[11] identity digest == replay digest at this run's count"] = [
                int(res["summary_digest"] is not None and res["summary_digest"] == res["identity_digest_replay"]), 1,
                f"{res['summary_digest']} vs {res['identity_digest_replay']}"]
            seqs[arm] = [m.groups()[1:5] for m in map(ALLOC.search, (HERE / "logs" / f"{prefix}_{run}.log")
                         .open(errors="ignore")) if m]
            share = res["realised_share"]
            print(f"\n=== {prefix}_{run}: {res['allocations']} allocations, commuter share "
                  f"{100 * share:.1f}% (nominal {100 * p:.0f}%), exhaustions {res['exhaustions']}, "
                  f"pool after Stage 0 median {res['pool_after_median']}, fallback {100 * (res['fallback_rate'] or 0):.1f}%")
            print(f"    arrivals by kind: {res['kind_tiers']}")
            for k, (ok, tot, ex) in sorted(res["rows"].items()):
                good = tot > 0 and ok == tot
                failed |= not good
                print(f"   [{'PASS' if good else 'FAIL'}] {k:70s} {ok:5d}/{tot:<5d}" + ("" if good else f"  e.g. {ex}"))
        # Across arms: the allocation sequences agree on their common prefix.
        n = min((len(v) for v in seqs.values()), default=0)
        same = n > 0 and len({tuple(v[:n]) for v in seqs.values()}) == 1
        print(f"  identity sequence at p={p}: {'identical across arms' if same else 'DIFFERS'} "
              f"over the common prefix of {n} allocations")
        failed |= not same

    dest = FEDASYNC / "gpu_tiers_dataset" / "stage11_evidence" / f"preflight_s{args.seed}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {dest}\n{'PRE-FLIGHT FAILED' if failed else 'PRE-FLIGHT PASSED: all checks 100%'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
