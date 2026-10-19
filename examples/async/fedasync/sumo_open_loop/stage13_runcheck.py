"""
Stage 13 (CIFAR-100 pilot): per-run soundness check, run by run_stage13.sh as
each run ends.

    python3 stage13_runcheck.py <prefix> <arm_regime> [rounds]    # rounds: 4000 (Stage 13), 10000 (Stage 13B)

Only the soundness checks the spec keeps (the plateau rule is disabled; a
failure is REPORTED and the queue continues):
  reached the configured last round
  conservation-law error <= 3 %
  filtered runs: stage0_fallback <= 10 % of rounds, median eligible_after_stage0 >= C = 4
  predictor tier == occupant tier on every dispatch (proactive arms)
Exit 0 = sound, 1 = a check failed.  Writes logs/stage13_runcheck.<prefix>_<run>.json.
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_stage7 as a7  # noqa: E402

ROUND = re.compile(r"\[Stage0\] round=-?\d+ .* eligible_after_stage0=(\d+) .*stage0_fallback=([01])")
C = 4


def main() -> int:
    prefix, run = sys.argv[1], sys.argv[2]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    a7.PREFIX = prefix
    r = a7.parse(run)
    cons = a7.conservation(r)
    res = {"prefix": prefix, "run": run, "max_round": max(r.clock) if r.clock else None,
           "rounds": rounds, f"T{rounds}": a7.clock_at(r, rounds), "conservation_rel_error": cons.get("rel_error")}
    fail = []
    if not r.clock or max(r.clock) < rounds - 1:   # a run's last merge can be logged one round early
        fail.append(f"ended at round {res['max_round']} < {rounds}")
    if cons.get("rel_error") is None or abs(cons["rel_error"]) > 0.03:
        fail.append(f"conservation error {cons.get('rel_error')}")
    after, fb = [], 0
    for line in (HERE / "logs" / f"{prefix}_{run}.log").open(errors="ignore"):
        m = ROUND.search(line)
        if m:
            after.append(int(m.group(1)))
            fb += m.group(2) == "1"
    if after:
        res.update(fallback_rate=fb / len(after), median_eligible_after_stage0=st.median(after))
        if res["fallback_rate"] > 0.10:
            fail.append(f"fallback {res['fallback_rate']:.3f} > 10%")
        if res["median_eligible_after_stage0"] < C:
            fail.append(f"median eligible_after_stage0 {res['median_eligible_after_stage0']} < {C}")
    elif not re.match(r"stage\w*?nf_", prefix):   # stage13nf_*, stage13bnf_*
        fail.append("filtered run logged no [Stage0] round lines")
    agree = a7.predictor_agreement(r)
    res["predictor_tier_agreement"] = agree
    if agree is not None and agree != 1.0:
        fail.append(f"predictor tier agreement {agree}")
    res["plateau_info_only"] = a7.curve_stats(r).get("plateau")
    res["soundness_failures"] = fail
    (HERE / "logs" / f"stage13_runcheck.{prefix}_{run}.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, default=str))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
