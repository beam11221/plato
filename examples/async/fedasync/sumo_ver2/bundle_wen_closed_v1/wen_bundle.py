#!/usr/bin/env python3
"""wen_bundle.py -- assemble bundle_wen_closed_v1/ for RunPod and zip it.

Usage
    python wen_bundle.py --run wen_closed_v1_run --out bundle_wen_closed_v1
Refuses to bundle unless gate_report.md reports 0 FAIL (override: --force).
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

FILES = [
    "trace.csv.gz", "fleet_speeds.csv",
    "wen.nod.xml", "wen.edg.xml", "wen.net.xml", "wen.rou.xml", "wen.sumocfg",
    "gate_report.md",
    "g1_speed_scatter.png", "g5_dwell_hist.png", "g6_incoverage_timeseries.png",
    "reentry_log.csv", "lap_log.csv", "dwell_log.csv", "lane_stats.csv",
    "sim_meta.yaml",
]
SCRIPTS = ["wen_closed.py", "wen_gate.py", "wen_bundle.py"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    report = (a.run / "gate_report.md").read_text()
    m = re.search(r"\*\*(\d+)/(\d+) PASS, (\d+) FAIL\.\*\*", report)
    if m is None:
        print("gate_report.md has no PASS/FAIL summary -- refusing to bundle")
        return 2
    if int(m.group(3)) and not a.force:
        print(f"gate_report.md reports {m.group(3)} FAIL -- refusing to bundle "
              f"(use --force to override)")
        return 2

    out = a.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in FILES:
        shutil.copy2(a.run / name, out / name)
    here = Path(__file__).resolve().parent
    for name in SCRIPTS:
        shutil.copy2(here / name, out / name)

    meta = yaml.safe_load((a.run / "sim_meta.yaml").read_text())
    manifest = {
        "bundle": out.name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec": "Wen et al., Information Sciences 664 (2024), 5.3.4 highway setting",
        "sumo_version": meta["sumo_version"],
        "netconvert_version": subprocess.run(
            ["netconvert", "--version"], capture_output=True, text=True
        ).stdout.splitlines()[0],
        "python_version": meta["python_version"],
        "platform": platform.platform(),
        "speed_rng": meta["speed_rng"],
        "speed_rng_seed": meta["speed_rng_seed"],
        "step_length_s": meta["step_length_s"],
        "sample_interval_s": meta["sample_interval_s"],
        "warmup_s": meta["warmup_s"],
        "duration_s": meta["duration_s"],
        "road_length_m": meta["road_length_m"],
        "lanes": meta["lanes"],
        "lane_speed_mps": meta["lane_speed_mps"],
        "n_vehicles": meta["n_vehicles"],
        "desired_speed_dist": meta["desired_speed_dist"],
        "motion_model": meta["motion_model"],
        "car_follow_model": meta["car_follow_model"],
        "lane_change_model": meta["lane_change_model"],
        "closed_loop": meta["closed_loop"],
        "adr": "docs/adr/ADR-0001-wen-closed-environment.md",
        "rsu_modelled_in_sumo": False,
        "rsu_downstream": {"x_m": 1000.0, "lateral_offset_m": 10.0,
                           "s1_coverage": "100%", "s2_coverage_m": [400.0, 1600.0]},
        "trace_schema": ["t", "veh_id", "x_m", "lane", "speed_mps", "lap"],
        "trace_time_base": meta["trace_time_base"],
        "lap_counting": meta["lap_counting"],
        "sha256": {p.name: sha256(p) for p in sorted(out.iterdir())},
    }
    (out / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            z.write(p, p.relative_to(out.parent))
    print(f"{zip_path}  sha256={sha256(zip_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
