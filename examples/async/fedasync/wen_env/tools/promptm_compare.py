"""
Prompt M A2 / A3: compare a new run with the original seed-4126 run.

Row check: the deterministic dispatch-log columns, row by row over the rows both
runs have; prints the first differing row. Overview: TTA@80/85/90 %, rounds,
s/round, tier share in rounds 1-500, max accuracy, and online = replayed clock.

Usage (from examples/async/fedasync):
    python wen_env/tools/promptm_compare.py NEW ORIGINAL [NEW ORIGINAL ...]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import wen_clock_replay  # noqa: E402
from wen_main_report import accuracy_curve  # noqa: E402

COLUMNS = ("cycle", "round", "client_id", "tier", "t_dispatch", "x_dispatch",
           "T_train_true", "x_finish", "chi2", "T_up_true", "duration", "accepted")
TIERS = ("xavier", "orin", "thor")


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def row_check(new: Path, old: Path) -> str:
    a, b = _rows(new / "dispatch_log.csv"), _rows(old / "dispatch_log.csv")
    n = min(len(a), len(b))
    for i in range(n):
        ra, rb = tuple(a[i][c] for c in COLUMNS), tuple(b[i][c] for c in COLUMNS)
        if ra != rb:
            diff = {c: (a[i][c], b[i][c]) for c in COLUMNS if a[i][c] != b[i][c]}
            return f"first difference at dispatch row {i + 1} (of {n} compared): {diff}"
    return f"identical over {n} rows (new {len(a)}, original {len(b)})"


def overview(run: Path) -> dict:
    curve = accuracy_curve(run)
    d = _rows(run / "dispatch_log.csv")
    out = {"rounds": curve[-1][0] if curve else 0,
           "max_acc": max((a for _, _, a in curve), default=None)}
    for target in (0.80, 0.85, 0.90):
        hit = next(((r, t) for r, t, a in curve if a >= target), None)
        out[f"tta{int(target * 100)}"] = round(hit[1]) if hit else None
        out[f"r{int(target * 100)}"] = hit[0] if hit else None
    out["s_per_round"] = round(curve[-1][1] / curve[-1][0], 3) if curve else None
    early = [r for r in d if int(r["round"]) <= 500]
    out["share_r1_500"] = "/".join(f"{sum(r['tier'] == t for r in early) / len(early):.2f}"
                                   for t in TIERS) if early else None
    out["clock_max_diff_s"] = wen_clock_replay.replay(run)["max_abs_diff_s"]
    return out


def main(argv: list[str]) -> int:
    pairs = list(zip(argv[0::2], argv[1::2]))
    for new, old in pairs:
        new, old = Path(new), Path(old)
        print(f"## {new.name} vs {old.name}")
        print("rows:", row_check(new, old))
        for name, run in (("new", new), ("original", old)):
            print(f"{name:9s}", overview(run))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
