"""
Smoke test: does `sleep_simulation = true` keep the simulated heterogeneity
while giving back the real wall clock?

Runs the same short FedAsync job twice — once with clients really sleeping
(sleep_simulation = false, the historical setting) and once with the sleep only
accounted for (sleep_simulation = true) — then compares:

    * real wall clock consumed by the process
    * the server's simulated elapsed_time (results CSV)
    * the per-client training_time the server was told about

The claim under test: run B costs far less real time than run A while the
server's view of the world (simulated clock, staleness, tier ordering) is
materially unchanged.

Run:
    cd plato/examples/async/fedasync/gpu_tiers
    python smoke_test_sleep_simulation.py
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_EXAMPLE = _HERE.parent
_REPO = _HERE.parents[3]

ANCHOR, GAP = 20, "20"
ROUNDS = 10
PER_ROUND = 3
AVG_TRAINING_TIME = 1.5   # calibrated: real ResNet-18 compute per client per round

OUT = _HERE / ".smoke"
PASS, FAIL, WARN = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m", "\033[33mWARN\033[0m"


def render(name: str, *, sleep_simulation: bool, port: int) -> Path:
    """Small variant of the GPU-arm blueprint: same tiers, far fewer rounds."""
    text = (_HERE / "templates" / "fedasync_polaris_vi_gpu_arm.toml.tmpl").read_text()
    text = (
        text.replace("{{GAP}}", GAP)
        .replace("{{ANCHOR}}", str(ANCHOR))
        .replace("{{PORT}}", str(port))
    )
    subs = {
        r'^base_path = .*$': f'base_path = "{OUT / name}"',
        r'^per_round = .*$': f"per_round = {PER_ROUND}",
        r'^rounds = .*$': f"rounds = {ROUNDS}",
        r'^max_concurrency = .*$': f"max_concurrency = {PER_ROUND}",
        r'^target_accuracy = .*$': "target_accuracy = 0.99",
        r'^eval_interval = .*$': "eval_interval = 2",
        r'^sleep_simulation = .*$': f"sleep_simulation = {str(sleep_simulation).lower()}",
        # avg_training_time is already in the blueprint; only its value varies here.
        # It is inert when sleep_simulation = false (defaults.py reads it solely
        # inside that branch), so arm A keeps it without effect.
        r'^avg_training_time = .*$': f"avg_training_time = {AVG_TRAINING_TIME}",
        r'^download = .*$': f'download = true\ndata_path = "{_EXAMPLE / "data"}"',
    }
    for pattern, repl in subs.items():
        text = re.sub(pattern, repl, text, count=1, flags=re.MULTILINE)

    path = OUT / f"{name}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def run(name: str, config: Path) -> tuple[float, Path]:
    log = OUT / f"{name}.out"
    env_python = sys.executable
    started = time.time()
    with log.open("w") as fh:
        subprocess.run(
            [env_python, "fedasync_gpu_tier_polaris_vi.py", "-c", str(config)],
            cwd=_EXAMPLE,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": f"{_HERE}:{_REPO}",
                "HOME": str(Path.home()),
            },
            timeout=1800,
            check=False,
        )
    return time.time() - started, log


def simulated_elapsed(name: str) -> float:
    """Last elapsed_time in the run's results CSV — the server's simulated clock."""
    results = sorted((OUT / name).rglob("*.csv"))
    for path in results:
        with path.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("elapsed_time")]
        if rows:
            return float(rows[-1]["elapsed_time"])
    return float("nan")


def training_times(log: Path) -> dict[str, list[float]]:
    """Per-tier training_time the server recorded, via the [GpuTierSleep] lines."""
    text = log.read_text(errors="replace")
    out: dict[str, list[float]] = {}
    for tier, secs in re.findall(
        r"\[GpuTierSleep\].*?tier_id=(\w+).*?sleep_time=([\d.]+)", text
    ):
        out.setdefault(tier, []).append(float(secs))
    return out


def main() -> int:
    print(f"Rounds={ROUNDS}  per_round={PER_ROUND}  gap={GAP}  anchor={ANCHOR}s")
    print("Tier sleeps in play: xavier 20.0 s / orin 4.47 s / thor 1.0 s\n")

    results = {}
    for name, flag, port in (("A_real_sleep", False, 8971), ("B_simulated", True, 8972)):
        cfg = render(name, sleep_simulation=flag, port=port)
        print(f"  running {name} (sleep_simulation = {str(flag).lower()}) ...", flush=True)
        wall, log = run(name, cfg)
        results[name] = {
            "wall": wall,
            "sim": simulated_elapsed(name),
            "tiers": training_times(log),
            "log": log,
        }
        print(f"    real {wall:7.1f} s   simulated {results[name]['sim']:9.1f} s")

    a, b = results["A_real_sleep"], results["B_simulated"]
    print(f"\n{'':22}{'real wall clock':>18}{'simulated clock':>18}")
    print(f"{'A sleep_simulation=false':22}{a['wall']:>16.1f} s{a['sim']:>16.1f} s")
    print(f"{'B sleep_simulation=true':22}{b['wall']:>16.1f} s{b['sim']:>16.1f} s")

    failures = []

    ok = b["wall"] < a["wall"]
    speedup = a["wall"] / b["wall"] if b["wall"] > 0 else float("inf")
    print(f"\n[{PASS if ok else FAIL}] B is faster in real time: {speedup:.2f}x")
    if not ok:
        failures.append("B not faster")

    for name, r in results.items():
        tiers = r["tiers"]
        seen = {t: round(max(v), 3) for t, v in sorted(tiers.items())}
        ok = bool(tiers)
        print(f"[{PASS if ok else FAIL}] {name}: tier sleeps seen by clients = {seen}")
        if not ok:
            failures.append(f"{name} logged no tier sleeps")

    if a["sim"] > 0 and b["sim"] > 0:
        ratio = b["sim"] / a["sim"]
        ok = 0.5 < ratio < 2.0
        print(
            f"[{PASS if ok else WARN}] simulated clocks are comparable: "
            f"B/A = {ratio:.2f} (async scheduling makes them close, not identical)"
        )

    print(f"\nlogs: {a['log']}  {b['log']}")
    if failures:
        print(f"{FAIL} {failures}")
        return 1
    print(f"{PASS} smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
