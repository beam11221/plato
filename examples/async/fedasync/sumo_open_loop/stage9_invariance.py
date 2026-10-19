"""
Stage 9, Step 2: baseline invariance.

The Stage 9 `baseline_open` (seed 4126, both Stage 9 knobs present in its
config) must reproduce Stage 7b's `baseline_open` exactly: the same dispatch
sequence (round, slot, veh_id) and the same reconstructed cumulative time at
round 4,000.  Pass -> the knobs do not touch the arm that ignores them, and
the six baseline cells of 7b / 8 can be reused.

Exit status 0 on an exact match, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_stage7 as a7  # noqa: E402

OUT = HERE.parent / "gpu_tiers_dataset" / "stage9_evidence" / "baseline_invariance.json"


def load(prefix):
    a7.PREFIX = prefix
    return a7.parse("baseline_open")


def main() -> int:
    ref, new = load("stage7b"), load("stage9_s4126")
    seq_ref = [(d.round, d.slot, d.veh_id) for d in ref.dispatches]
    seq_new = [(d.round, d.slot, d.veh_id) for d in new.dispatches]
    first_diff = next((i for i, (a, b) in enumerate(zip(seq_ref, seq_new)) if a != b), None)
    t_ref, t_new = ref.clock.get(4000), new.clock.get(4000)
    res = {
        "dispatches": [len(seq_ref), len(seq_new)],
        "dispatch_sequence_identical": seq_ref == seq_new,
        "first_difference_index": first_diff,
        "T4000": [t_ref, t_new],
        "T4000_identical": t_ref == t_new,
        "recycle_sequence_identical": ref.recycle_seq == new.recycle_seq,
    }
    res["pass"] = (res["dispatch_sequence_identical"] and res["T4000_identical"]
                   and res["recycle_sequence_identical"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
