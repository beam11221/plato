"""
Stage 7b, Step 2: the pre-flight, read from the smoke-run LOGS.

Every check is a per-dispatch fraction that must be exactly 100%.  The
expected values are recomputed here independently of the server (from the
trace vehicle id, the fleet files and the shared derivations), so a check
passes only if what the run logged agrees with what the specification says,
not merely with itself.

    predictor tier == occupant tier          (proactive arms)
    sleep == occupant's sleep                (dispatch line, and the report's)
    partition == occupant's partition        (client line; sampler id; and the
                                             partition fingerprint must group by
                                             partition, never by slot)
    p_recur = 1: slot k's identity, tier and partition identical in every window
    p_recur = 0: state_reset == recycles; identity == trace vehicle
    tau_hat logged == tau_hat the selector used  (vs the prediction-accuracy line)

plus the checks Stage 7b's Step 1 added:

    training seed == identity_seed(base, identity, round)
    a never-seen identity's G_i == the current unexplored proxy (p_recur = 0)
    observations logged == that identity's earlier DELIVERED reports

Usage:
    python3 preflight_stage7b.py [--stage 7b_preflight] [run ...]
Exit status is non-zero if any check is below 100% or any run is missing.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from occupant_identity import identity_seed, load_fleet  # noqa: E402
from stage7_population import partition_index, vehicle_sleep_s  # noqa: E402

FEDASYNC = HERE.parent
FLEET = load_fleet(
    FEDASYNC / "sumo_close_loop" / "input" / "clients_n20r400.csv",
    FEDASYNC / "sumo_close_loop" / "output" / "client_sleep_times_n20r400.csv",
)
M = 20
BASE_SEED = 4126
# Startup T_round the Polaris server computes from the fleet file (Stages 3-6).
FLEET_T_ROUND = {"gpu": 2.080769, "cpu": 3.620192}
# The gpu arm's three tau_hat values in every Stage 6 log (T_pred / 2.080769).
GPU_TAU_HAT = {"xavier": 7.208872, "orin": 0.924214, "thor": 0.480591}
RUNS = [f"{a}_{r}" for r in ("open", "closed") for a in ("baseline", "gpu", "cpu", "observed")]

DISPATCH = re.compile(
    r"\[Stage7\] dispatch round=(-?\d+) slot=(\d+) veh_id=(\S+) tier=(\S+) "
    r"observations=(\d+) G_i=(\S+) G_i_at_prior=([01]) tau_hat=(\S+) "
    r"identity_id=(\S+) pred_tier=(\S+) tau_obs=(\S+) partition=(\d+) "
    r"sleep_s=(\S+) G_proxy=(\S+)"
)
REPORT = re.compile(
    r"\[Stage7\] report round=\d+ slot=(\d+) veh_id=(\S+) tier=(\S+) .*"
    r"uploaded=([01]) raw_in_range=[01] sleep=(\S+) identity_id=(\S+)"
)
CLIENT = re.compile(
    r"\[Stage7\] client slot=(\d+) veh_id=(\S+) identity_id=(\S+) tier=(\S+) "
    r"partition=(\d+) sampler_id=(\d+) fingerprint=(\S+)"
)
RECYCLE = re.compile(r"\[Stage7\] recycle slot_id=(\d+) .* state_reset=([01])")
ACCURACY = re.compile(r"Staleness prediction accuracy: cid=(\d+)\s+predicted=(\S+) rounds")
SEED = re.compile(
    r"\[ClientSeededTrainer\] client_id=(\d+) identity_id=(\S+) round=(\d+) -> seed (\d+)"
)
STARTUP = re.compile(r"\[Stage7b\] startup T_round=(\S+) s \(fleet file, kept")


class Check:
    def __init__(self) -> None:
        self.ok = 0
        self.total = 0
        self.examples: list[str] = []

    def add(self, passed: bool, example: str = "") -> None:
        self.total += 1
        if passed:
            self.ok += 1
        elif len(self.examples) < 3:
            self.examples.append(example)

    @property
    def pct(self) -> float | None:
        return None if self.total == 0 else 100.0 * self.ok / self.total


def expected_identity(p_recur: float, slot: int, veh_id: str, trace_tier: str):
    if p_recur == 1.0:
        tier, sleep = FLEET[slot]
        return str(slot), tier, sleep, slot - 1
    return veh_id, trace_tier, vehicle_sleep_s(BASE_SEED, veh_id, trace_tier), \
        partition_index(veh_id, M)


def check_run(log: Path, p_recur: float, proactive: bool, polaris: bool) -> dict:
    checks: dict[str, Check] = defaultdict(Check)
    trace_tier: dict[str, str] = {}
    last_dispatch: dict[int, tuple] = {}      # slot -> (identity, tau_hat)
    delivered: dict[str, int] = defaultdict(int)
    fp_by_partition: dict[int, set] = defaultdict(set)
    partition_by_fp: dict[str, set] = defaultdict(set)
    slot_identity: dict[int, set] = defaultdict(set)
    recycles = resets = 0
    startup = None
    dispatches = 0

    # The trace tier of each vehicle, for the p_recur = 0 expectations.
    import csv
    with (HERE / "sumo_trace" / "e1_open_poisson_seed4126_range400" /
          "coverage_windows.csv").open() as fh:
        for row in csv.DictReader(fh):
            trace_tier[row["veh_id"]] = row["tier"]

    for line in log.open(errors="ignore"):
        m = DISPATCH.search(line)
        if m:
            dispatches += 1
            (rnd, slot, veh, tier, obs, g_i, at_prior, tau_hat, ident, pred_tier,
             tau_obs, part, sleep_s, g_proxy) = m.groups()
            slot, part, obs = int(slot), int(part), int(obs)
            e_id, e_tier, e_sleep, e_part = expected_identity(p_recur, slot, veh, trace_tier[veh])
            where = f"round {rnd} slot {slot} veh {veh}"
            checks["identity == spec"].add(ident == e_id, f"{where}: {ident} vs {e_id}")
            checks["tier == occupant tier"].add(tier == e_tier, f"{where}: {tier} vs {e_tier}")
            checks["sleep_s == occupant sleep (dispatch)"].add(
                abs(float(sleep_s) - e_sleep) < 1e-5, f"{where}: {sleep_s} vs {e_sleep:.6f}")
            checks["partition == occupant partition (dispatch)"].add(
                part == e_part, f"{where}: {part} vs {e_part}")
            if proactive:
                checks["predictor tier == occupant tier"].add(
                    pred_tier == e_tier, f"{where}: pred {pred_tier} vs {e_tier}")
            if proactive and "gpu" in log.name:
                # Fleet-file T_round: the gpu arm must reproduce Stages 3-6's
                # per-tier tau_hat exactly, now on the OCCUPANT's tier.
                checks["gpu tau_hat == Stage 3-6 value for occupant tier"].add(
                    abs(float(tau_hat) - GPU_TAU_HAT[e_tier]) < 5e-5,
                    f"{where}: {tau_hat} vs {GPU_TAU_HAT[e_tier]} ({e_tier})")
            if polaris:
                checks["observations == earlier delivered reports"].add(
                    obs == delivered[ident], f"{where}: {obs} vs {delivered[ident]}")
                if proactive is False:
                    checks["tau_hat == tau_obs (observed arm)"].add(
                        math.isclose(float(tau_hat), float(tau_obs), abs_tol=1e-6),
                        f"{where}: {tau_hat} vs {tau_obs}")
                if p_recur == 0.0 and obs == 0 and g_proxy != "-":
                    checks["never-seen G_i == current proxy"].add(
                        math.isclose(float(g_i), float(g_proxy), rel_tol=1e-6)
                        and at_prior == "1", f"{where}: G_i {g_i} vs proxy {g_proxy}")
            if p_recur == 1.0:
                slot_identity[slot].add((ident, tier, part))
            last_dispatch[slot] = (ident, tau_hat)
            continue

        m = REPORT.search(line)
        if m:
            slot, veh, tier, uploaded, sleep, ident = m.groups()
            e_id, e_tier, e_sleep, _ = expected_identity(p_recur, int(slot), veh, trace_tier[veh])
            checks["sleep == occupant sleep (report)"].add(
                abs(float(sleep) - e_sleep) < 1e-5 and ident == e_id,
                f"slot {slot} veh {veh}: {sleep} vs {e_sleep:.6f}")
            if uploaded == "1":
                delivered[ident] += 1
            continue

        m = CLIENT.search(line)
        if m:
            slot, veh, ident, tier, part, sampler_id, fp = m.groups()
            slot, part, sampler_id = int(slot), int(part), int(sampler_id)
            e_id, e_tier, _, e_part = expected_identity(p_recur, slot, veh, trace_tier[veh])
            checks["partition == occupant partition (client)"].add(
                part == e_part and ident == e_id, f"slot {slot} veh {veh}: {part} vs {e_part}")
            checks["sampler_id == partition + 1"].add(
                sampler_id == part + 1, f"slot {slot}: sampler {sampler_id} partition {part}")
            fp_by_partition[part].add(fp)
            partition_by_fp[fp].add(part)
            continue

        m = RECYCLE.search(line)
        if m:
            recycles += 1
            resets += int(m.group(2))
            continue

        m = ACCURACY.search(line)
        if m and polaris:
            cid, predicted = int(m.group(1)), float(m.group(2))
            if cid in last_dispatch:
                _, tau_hat = last_dispatch[cid]
                checks["tau_hat logged == tau_hat used"].add(
                    abs(float(tau_hat) - predicted) < 5e-5,
                    f"slot {cid}: logged {tau_hat} vs used {predicted}")
            continue

        m = SEED.search(line)
        if m:
            _cid, ident, rnd, seed = m.groups()
            checks["seed == identity_seed(identity, round)"].add(
                int(seed) == identity_seed(BASE_SEED, ident, int(rnd)),
                f"identity {ident} round {rnd}")
            continue

        m = STARTUP.search(line)
        if m:
            startup = float(m.group(1))

    # Fingerprints must group by partition: one fingerprint per partition, and
    # no fingerprint shared by two partitions.
    for part, fps in fp_by_partition.items():
        checks["one fingerprint per partition"].add(len(fps) == 1, f"partition {part}: {fps}")
    for fp, parts in partition_by_fp.items():
        checks["one partition per fingerprint"].add(len(parts) == 1, f"fp {fp}: {parts}")

    if p_recur == 1.0:
        for slot, seen in slot_identity.items():
            checks["p=1: slot keeps one identity/tier/partition"].add(
                len(seen) == 1, f"slot {slot}: {seen}")
        # Only a virgin slot's first use may be a reset.
        checks["p=1: resets only on first use"].add(resets <= M, f"{resets} resets")
    else:
        checks["p=0: state_reset == recycles"].add(resets == recycles, f"{resets}/{recycles}")

    if proactive:
        expected = FLEET_T_ROUND["cpu" if "cpu" in log.name else "gpu"]
        checks["startup T_round == fleet-file value"].add(
            startup is not None and abs(startup - expected) < 1e-5,
            f"logged {startup} vs {expected}")


    return {
        "dispatches": dispatches,
        "recycles": recycles,
        "state_resets": resets,
        "startup_t_round": startup,
        "partitions_seen": len(fp_by_partition),
        "checks": {
            name: {"ok": c.ok, "total": c.total, "pct": c.pct, "examples": c.examples}
            for name, c in sorted(checks.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="7b_preflight")
    parser.add_argument("runs", nargs="*", default=RUNS)
    parser.add_argument("--out", type=Path, default=FEDASYNC / "gpu_tiers_dataset" /
                        "stage7_evidence" / "stage7b_preflight.json")
    args = parser.parse_args()

    results, failed = {}, False
    for name in args.runs:
        log = HERE / "logs" / f"stage{args.stage}_{name}.log"
        arm, regime = name.split("_")
        if not log.exists():
            print(f"{name}: MISSING {log}")
            failed = True
            continue
        res = check_run(log, 1.0 if regime == "closed" else 0.0,
                        proactive=arm in ("gpu", "cpu"), polaris=arm != "baseline")
        results[name] = res
        print(f"\n=== {name}: {res['dispatches']} dispatches, {res['recycles']} recycles, "
              f"{res['state_resets']} resets, {res['partitions_seen']} partitions, "
              f"startup T_round {res['startup_t_round']}")
        if res["dispatches"] == 0:
            print("   NO DISPATCHES")
            failed = True
        for cname, c in res["checks"].items():
            mark = "PASS" if c["pct"] == 100.0 else "FAIL"
            if c["pct"] != 100.0:
                failed = True
            print(f"   [{mark}] {cname:48s} {c['ok']:5d}/{c['total']:<5d}"
                  + (f"  e.g. {c['examples'][0]}" if c["examples"] else ""))

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}\n{'PRE-FLIGHT FAILED' if failed else 'PRE-FLIGHT PASSED: all checks 100%'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
