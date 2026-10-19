"""
Stage 11 launch gate: "If Stage 10 Wave B shows alpha = 10 beating alpha = 1 for
gpu + F in the open population in EVERY seed, stop and report before launching."

Runs analyze_stage10 --wave B and reads comparison I (gpu_open): the per-seed
% gap alpha=1 minus alpha=10 (positive = alpha = 1 slower = alpha = 10 beats it).
Exit 0 = launch allowed; 1 = stop (alpha = 10 wins every seed); 2 = data missing.
"""
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
out = HERE.parent / "gpu_tiers_dataset" / "stage10_evidence" / "stage10_analysis_waveB.json"
subprocess.run([sys.executable, str(HERE / "analyze_stage10.py"), "--wave", "B"], check=True)
I = json.loads(out.read_text())["comparisons"].get("I", {}).get("gpu_open", {})
per_seed = I.get("per_seed", {})
print(f"Stage 10 comparison I, gpu+F open, alpha=1 vs alpha=10 (%): {per_seed}")
if len(per_seed) < 3:
    print("STAGE11_ALPHA_CHECK: data missing"); sys.exit(2)
if all(v > 0 for v in per_seed.values()):
    print("STAGE11_ALPHA_CHECK: STOP -- alpha = 10 beats alpha = 1 in every seed"); sys.exit(1)
print("STAGE11_ALPHA_CHECK: PASS -- alpha = 1 stays"); sys.exit(0)
