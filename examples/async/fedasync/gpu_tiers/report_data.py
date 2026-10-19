#!/usr/bin/env python3
"""Extract the numbers behind the H-sweep / mix-sweep report, on the CLEAN clock.

Every time here is PLATO's *simulated* clock rebuilt by recompute_wall_time.py
with ``--floor 0``: tier sleep only, no host serialisation or transfer cost.
That is the axis to use when the runs did not share a GPU and some ran in
parallel while others ran alone — the reported ``elapsed_time`` carries those
differences (HANDOFF § 4.8), the clean one does not.

Writes report_data.json next to this file.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from plot_time_to_accuracy import first_crossing, load_curve  # noqa: E402

RUNS = HERE.parent / "runs"
LOGS = HERE / "logs"
SMOOTH = 25
TARGET = 55.0
TIERS = ("xavier", "orin", "thor")

# H-sweep: seed 8888 is dropped on purpose (see report). 1234/5678 both live in
# gap_sweep_ms and both now have a reconstructed clean clock, so the two seeds
# are read off the same axis — no mixing of clean and reported.
H_SEEDS = [1234, 5678]
H_GAPS = ["01", "10", "20", "30", "40"]
ARMS = ["gpu", "cpu", "baseline"]
MIXES = ["m70_20_10", "m50_35_15", "m34_33_33", "m15_35_50"]

DISPATCH_RE = re.compile(r"\[GpuTierSleep\] round=(\d+) client_id=(\d+) tier_id=(\w+)")


def dispatch_share(log: Path) -> dict:
    """Share of dispatches per tier, deduplicated by (round, client).

    The client logs one line per dispatch it receives, so this counts what the
    server actually handed out — it works for `baseline` too, which emits no
    V_i line for analyze_steering.py to parse.
    """
    seen, count = set(), Counter()
    with log.open(errors="ignore") as fh:
        for line in fh:
            m = DISPATCH_RE.search(line)
            if not m:
                continue
            key = (m.group(1), m.group(2))
            if key in seen:
                continue
            seen.add(key)
            count[m.group(3)] += 1
    n = sum(count.values())
    return {"n": n, **{t: 100.0 * count[t] / n for t in TIERS}} if n else {}


def pool_share(clients_csv: Path) -> dict:
    c = Counter()
    with clients_csv.open() as fh:
        for row in csv.DictReader(fh):
            c[row["tier_id"]] += 1
    n = sum(c.values())
    return {t: 100.0 * c[t] / n for t in TIERS}


def cell(run_dir: Path, clean_csv: Path) -> dict | None:
    cands = sorted(
        (run_dir / "results").glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    cands = [p for p in cands if not p.stem.endswith(("_binned", "_accuracy_bins"))]
    if not cands:
        return None
    df = load_curve(cands[0], stop_at=None, smooth=SMOOTH, clean_time=clean_csv)
    if len(df) < 50 or df["time_source"].iloc[0] != "clean":
        return None
    hit = first_crossing(df, TARGET)
    return {
        "hours": None if hit is None else hit[1] / 3600.0,
        "round": None if hit is None else hit[0],
        "curve": {
            "hours": [round(h, 5) for h in df["hours"].tolist()],
            "acc": [round(a, 3) for a in df["smooth_pct"].tolist()],
        },
    }


def main() -> None:
    out: dict = {"target": TARGET, "smooth": SMOOTH, "seeds": H_SEEDS}

    # ---------------- H-sweep ----------------
    h_pool = pool_share(HERE / "input" / "clients.csv")
    h: dict = {"pool": h_pool, "cells": {}}
    for arm in ARMS:
        for gap in H_GAPS:
            per_seed, shares, curves = {}, [], {}
            for seed in H_SEEDS:
                name = f"{arm}_xa20_gap{gap}_s{seed}"
                c = cell(RUNS / "gap_sweep_ms" / name,
                         HERE / "figures" / "multiseed" / "clean_time" / f"{name}.csv")
                if c is None:
                    continue
                per_seed[seed] = c["hours"]
                curves[seed] = c["curve"]
                s = dispatch_share(LOGS / f"{name}.log")
                if s:
                    shares.append(s)
            got = [v for v in per_seed.values() if v is not None]
            h["cells"][f"{arm}_{gap}"] = {
                "arm": arm, "gap": gap,
                "per_seed": per_seed,
                "mean_hours": float(np.mean(got)) if got else None,
                "spread": (max(got) - min(got)) / 2 if len(got) > 1 else None,
                "share": {t: float(np.mean([s[t] for s in shares])) for t in TIERS}
                if shares else None,
                "curves": curves,
            }
            print(f"  H {arm:9s} G={gap}  "
                  f"{h['cells'][f'{arm}_{gap}']['mean_hours']}", flush=True)
    out["hsweep"] = h

    # ---------------- mix sweep ----------------
    m: dict = {"cells": {}, "pool": {}}
    for mix in MIXES:
        m["pool"][mix] = pool_share(HERE / "input" / f"clients_{mix}.csv")
    for arm in ARMS:
        for mix in MIXES:
            name = f"{arm}_xa20_gap30_{mix}"
            c = cell(RUNS / "mix_sweep" / name,
                     HERE / "figures" / "mix" / "clean_time" / f"{name}.csv")
            s = dispatch_share(LOGS / f"{name}.log")
            m["cells"][f"{arm}_{mix}"] = {
                "arm": arm, "mix": mix,
                "hours": None if c is None else c["hours"],
                "round": None if c is None else c["round"],
                "share": s or None,
                "curve": None if c is None else c["curve"],
            }
            print(f"  M {arm:9s} {mix}  {m['cells'][f'{arm}_{mix}']['hours']}", flush=True)
    out["mix"] = m

    (HERE / "report_data.json").write_text(json.dumps(out))
    print("wrote report_data.json")


if __name__ == "__main__":
    main()
