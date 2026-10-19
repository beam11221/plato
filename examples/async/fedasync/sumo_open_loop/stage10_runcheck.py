"""
Stage 10: per-run stop-condition check, run by run_stage10.sh as each run ends.

    python3 stage10_runcheck.py <prefix> <arm_regime>

Checks (STAGE10_filter_open.md, "Stop conditions"):
  conservation-law error <= 3%
  plateau >= 80.1%
  gated arms: stage0_fallback <= 10% of rounds, median eligible_after_stage0 >= C = 4
  proactive arms: predictor tier == occupant tier on every dispatch
  occupancy: realised recycle sequence == Stage 7b baseline_open's over the
             common prefix (the trace-determined stream every run must share)
  baseline / none re-runs: |T @ 4,000 - Stage 7b/8 same cell| <= 3%  -> else
             Wave B is blocked (the harness would have drifted)

Exit 0 = pass; 1 = stop condition (stop the stage); 2 = block Wave B only.
Writes logs/stage10_runcheck.<prefix>_<run>.json.
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
REF_OF_SEED = {4126: "stage7b", 4127: "stage8_s4127", 4128: "stage8_s4128"}


def main() -> int:
    prefix, run = sys.argv[1], sys.argv[2]
    arm, regime = run.split("_")
    seed = int(re.search(r"_s(\d{4})", prefix).group(1))

    a7.PREFIX = prefix
    r = a7.parse(run)
    cons, curve = a7.conservation(r), a7.curve_stats(r)
    res = {"prefix": prefix, "run": run, "T4000": a7.clock_at(r, 4000),
           "conservation_rel_error": cons["rel_error"], "plateau": curve["plateau"]}
    stop, block = [], []
    if abs(cons["rel_error"]) > 0.03:
        stop.append(f"conservation error {cons['rel_error']:.4f}")
    if curve["plateau"] < 0.801:
        stop.append(f"plateau {curve['plateau']:.4f} < 0.801")

    after, fb = [], 0
    log = HERE / "logs" / f"{prefix}_{run}.log"
    if not log.exists():
        stop.append(f"log {log.name} missing")
    for line in (log.open(errors="ignore") if log.exists() else []):
        m = ROUND.search(line)
        if m:
            after.append(int(m.group(1)))
            fb += m.group(2) == "1"
    if after:
        res["gated_rounds"] = len(after)
        res["median_eligible_after_stage0"] = st.median(after)
        res["fallback_rate"] = fb / len(after)
        if res["fallback_rate"] > 0.10:
            stop.append(f"fallback {res['fallback_rate']:.3f} > 10%")
        if res["median_eligible_after_stage0"] < C:
            stop.append(f"median eligible_after_stage0 {res['median_eligible_after_stage0']} < {C}")
    elif not re.match(r"stage\d+nf_", prefix):   # no-filter runs (stage10nf_*, stage11nf_*) log no [Stage0] lines
        stop.append("filtered run logged no [Stage0] round lines")

    agree = a7.predictor_agreement(r)
    res["predictor_tier_agreement"] = agree
    if agree is not None and agree != 1.0:
        stop.append(f"predictor tier agreement {agree}")

    a7.PREFIX = "stage7b"
    ref = a7.parse("baseline_open").recycle_seq
    n = min(len(ref), len(r.recycle_seq))
    res["occupancy_prefix"] = n
    res["occupancy_identical"] = ref[:n] == r.recycle_seq[:n]
    if not res["occupancy_identical"]:
        stop.append("occupancy sequence differs from Stage 7b")

    if prefix.startswith("stage10nf"):
        a7.PREFIX = REF_OF_SEED[seed]
        t_ref = a7.clock_at(a7.parse(run), 4000)
        res["T4000_reference"] = t_ref
        res["vs_reference_pct"] = 100 * (res["T4000"] - t_ref) / t_ref
        if abs(res["vs_reference_pct"]) > 3.0:
            block.append(f"baseline/none differs from {REF_OF_SEED[seed]} by {res['vs_reference_pct']:+.2f}%")

    res["stop"], res["block_wave_b"] = stop, block
    (HERE / "logs" / f"stage10_runcheck.{prefix}_{run}.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, default=str))
    return 1 if stop else (2 if block else 0)


if __name__ == "__main__":
    raise SystemExit(main())
