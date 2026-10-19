"""
Stage 9, Step 2: pre-flight for the neutral priors (alpha = 1, fleet running-
mean tau for never-seen vehicles), read from the smoke-run LOGS.

All Stage 7b checks via `preflight_stage7b.check_run`, unchanged, except its
"tau_hat == tau_obs (observed arm)" row, which by design no longer holds for a
never-seen vehicle and is replaced here by the split below.  Added:

  G_prior    never-seen dispatch (p=0, observations = 0, proxy set):
             logged G_i == G_proxy, and G_proxy == 1.0 x G_proxy_basis
             (the batch-mean norm) -- i.e. not 10x
  tau rule   observed, seen: tau_hat == tau_obs (its own last staleness)
             observed, never-seen: tau_hat == tau_unseen logged, AND
             tau_unseen == the running mean recomputed independently from the
             delivered staleness values in the log ("actual=" of the
             prediction-accuracy lines), starting at C - 1 = 3
  V_i used   every selected client's logged V_i == G_i / sqrt(1 + tau_hat)
             of its dispatch line -- so G_i and tau_hat are what V_i used

Usage:
    python3 preflight_stage9.py [--prefix stage9_s4126_preflight] [runs ...]
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from preflight_stage7b import check_run  # noqa: E402

FEDASYNC = HERE.parent
C_MINUS_1 = 3.0
ALPHA = 1.0
RUNS = [f"{a}_{r}" for r in ("open", "closed") for a in ("baseline", "gpu", "cpu", "observed")]

DISPATCH = re.compile(
    r"\[Stage7\] dispatch round=(-?\d+) slot=(\d+) veh_id=\S+ tier=\S+ "
    r"observations=(\d+) G_i=(\S+) G_i_at_prior=[01] tau_hat=(\S+) "
    r"identity_id=\S+ pred_tier=\S+ tau_obs=(\S+) partition=\d+ "
    r"sleep_s=\S+ G_proxy=(\S+) G_proxy_basis=(\S+) tau_unseen=(\S+)"
)
ACCURACY = re.compile(r"Staleness prediction accuracy: cid=\d+\s+predicted=\S+ rounds\s+actual=(\S+) rounds")
SELECTED = re.compile(r"\[PolarisVi\] Round (\d+) — selected clients \[[^\]]*\] with V_i scores (\{[^}]*\})")


class Row:
    def __init__(self):
        self.ok = self.total = 0
        self.example = ""

    def add(self, good, example=""):
        self.total += 1
        self.ok += bool(good)
        if not good and not self.example:
            self.example = example


def stage9_checks(log: Path, p_recur: float, arm: str) -> dict:
    rows = {k: Row() for k in (
        "[9] never-seen G_i == proxy == 1.0 x basis",
        "[9] observed seen: tau_hat == tau_obs",
        "[9] observed never-seen: tau_hat == tau_unseen logged",
        "[9] tau_unseen logged == running mean recomputed from log",
        "[9] logged V_i == G_i / sqrt(1 + tau_hat)",
    )}
    tau_sum, tau_n = 0.0, 0
    # "selected ... V_i scores" is logged BEFORE that round's dispatch lines,
    # so hold each selected client's V_i until its dispatch line arrives.
    pending_v: dict[int, tuple[int, float]] = {}
    for line in log.open(errors="ignore"):
        m = ACCURACY.search(line)
        if m:
            tau_sum += float(m.group(1))
            tau_n += 1
            continue
        m = DISPATCH.search(line)
        if m:
            rnd, slot, obs, g_i, tau_hat, tau_obs, g_proxy, basis, tau_unseen = m.groups()
            rnd, slot, obs = int(rnd), int(slot), int(obs)
            g_i, tau_hat = float(g_i), float(tau_hat)
            where = f"round {rnd} slot {slot}"
            if slot in pending_v and pending_v[slot][0] == rnd:
                v = pending_v.pop(slot)[1]
                # V_i, G_i and tau_hat are each logged to 6 decimals, so the
                # recomputed V can differ by the propagated rounding:
                # 5e-7 (V) + 5e-7 / sqrt(1+tau) (G) + V * 5e-7 / (2 (1+tau)) (tau).
                # At alpha = 10, G ~ 35 and the tau term dominates (~4e-6).
                d = 1 + max(tau_hat, 0.0)
                expected = g_i / math.sqrt(d)
                bound = max(2e-6, 1.01 * (5e-7 + 5e-7 / math.sqrt(d) + expected * 5e-7 / (2 * d)))
                rows["[9] logged V_i == G_i / sqrt(1 + tau_hat)"].add(
                    abs(v - expected) <= bound,
                    f"{where}: V {v} vs {g_i}/sqrt(1+{tau_hat}) (|diff| {abs(v - expected):.2e} > {bound:.2e})")
            if arm != "baseline" and p_recur == 0.0 and obs == 0 and g_proxy != "-":
                rows["[9] never-seen G_i == proxy == 1.0 x basis"].add(
                    math.isclose(g_i, float(g_proxy), rel_tol=1e-5, abs_tol=1e-6)
                    and math.isclose(float(g_proxy), ALPHA * float(basis), rel_tol=1e-5, abs_tol=1e-6),
                    f"{where}: G_i {g_i} proxy {g_proxy} basis {basis}")
            if arm == "observed":
                if obs >= 1:
                    rows["[9] observed seen: tau_hat == tau_obs"].add(
                        math.isclose(tau_hat, float(tau_obs), abs_tol=1e-6), f"{where}: {tau_hat} vs {tau_obs}")
                else:
                    rows["[9] observed never-seen: tau_hat == tau_unseen logged"].add(
                        tau_unseen != "-" and math.isclose(tau_hat, float(tau_unseen), abs_tol=1e-5),
                        f"{where}: {tau_hat} vs {tau_unseen}")
                expected = tau_sum / tau_n if tau_n else C_MINUS_1
                rows["[9] tau_unseen logged == running mean recomputed from log"].add(
                    tau_unseen != "-" and math.isclose(float(tau_unseen), expected, abs_tol=1e-5),
                    f"{where}: logged {tau_unseen} vs recomputed {expected:.6f} (n={tau_n})")
            continue
        m = SELECTED.search(line)
        if m and arm != "baseline":
            rnd, scores = int(m.group(1)), ast.literal_eval(m.group(2))
            for cid, v in scores.items():
                pending_v[int(cid)] = (rnd, float(v))
    if arm != "baseline":
        rows["[9] every selected V_i matched a dispatch line"] = Row()
        rows["[9] every selected V_i matched a dispatch line"].add(
            not pending_v, f"{len(pending_v)} unmatched: {list(pending_v.items())[:3]}")
    return {k: r for k, r in rows.items() if r.total}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="stage9_s4126_preflight")
    ap.add_argument("runs", nargs="*", default=RUNS)
    args = ap.parse_args()

    failed, out = False, {}
    for name in args.runs:
        arm, regime = name.split("_")
        p_recur = 1.0 if regime == "closed" else 0.0
        log = HERE / "logs" / f"{args.prefix}_{name}.log"
        res = check_run(log, p_recur, proactive=arm in ("gpu", "cpu"), polaris=arm != "baseline")
        rows = {k: (v["ok"], v["total"], (v["examples"] or [""])[0])
                for k, v in res["checks"].items()
                if k != "tau_hat == tau_obs (observed arm)"}
        for k, r in stage9_checks(log, p_recur, arm).items():
            rows[k] = (r.ok, r.total, r.example)
        print(f"\n=== {args.prefix}_{name}: {res['dispatches']} dispatches, "
              f"{res['recycles']} recycles, {res['state_resets']} resets")
        for k, (ok, tot, ex) in sorted(rows.items()):
            good = tot > 0 and ok == tot
            failed |= not good
            print(f"   [{'PASS' if good else 'FAIL'}] {k:58s} {ok:5d}/{tot:<5d}"
                  + ("" if good else f"  e.g. {ex}"))
        out[name] = {k: [ok, tot] for k, (ok, tot, _) in rows.items()}

    dest = FEDASYNC / "gpu_tiers_dataset" / "stage9_evidence" / f"{args.prefix}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}\n{'PRE-FLIGHT FAILED' if failed else 'PRE-FLIGHT PASSED: all checks 100%'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
