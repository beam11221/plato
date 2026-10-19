"""
Check one cell's inputs before spending GPU time on it.

Everything here is derived from the files on disk, so a mismatch between the
fleet, the sleep times, the coverage trace and the configs is caught in a second
instead of 70 seconds into a six-hour run. It also prints the numbers the run
should reproduce -- T_round, tau_hat per tier, expected waste -- so the first
log lines can be checked against an expectation rather than eyeballed.

Usage:  python3 preflight.py --cell n20
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent

FAILURES: list[str] = []


def ok(message: str) -> None:
    print(f"  ok    {message}")


def fail(message: str) -> None:
    FAILURES.append(message)
    print(f"  FAIL  {message}")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def check_files(cell: str) -> tuple[list[dict], list[dict], list[dict]]:
    print(f"\n[1] input files for {cell}")
    clients = read_csv(HERE / "input" / f"clients_{cell}.csv")
    mapping = read_csv(HERE / "input" / f"mapping_{cell}.csv")
    coverage = read_csv(HERE / "input" / f"coverage_{cell}.csv")
    sleeps = read_csv(HERE / "output" / f"client_sleep_times_{cell}.csv")

    n = len(clients)
    ok(f"clients_{cell}.csv        {n} clients")

    tiers = {int(r["client_id"]): r["tier_id"] for r in clients}
    counts = {t: sum(1 for v in tiers.values() if v == t) for t in sorted(set(tiers.values()))}
    ok(f"composition              {counts}")

    if {int(r["client_id"]) for r in mapping} != set(tiers):
        fail("mapping_*.csv does not cover the same client IDs as clients_*.csv")
    elif any(tiers[int(r["client_id"])] != r["tier_id"] for r in mapping):
        fail("mapping_*.csv assigns a tier that clients_*.csv disagrees with")
    else:
        ok(f"mapping_{cell}.csv        tier-preserving over {len(mapping)} vehicles")

    covered = {int(r["client_id"]) for r in coverage}
    if covered != set(tiers):
        fail(f"coverage_*.csv is missing clients {sorted(set(tiers) - covered)}")
    else:
        ok(f"coverage_{cell}.csv       {len(coverage)} windows, all {n} clients")

    if {int(r["client_id"]) for r in sleeps} != set(tiers):
        fail("sleep-time CSV does not cover the same clients")
    else:
        ok(f"client_sleep_times_{cell}.csv  {len(sleeps)} clients")

    return clients, coverage, sleeps


def check_speed_model(clients: list[dict], sleeps: list[dict]) -> dict[str, float]:
    print("\n[2] speed model")
    tiers = {int(r["client_id"]): r["tier_id"] for r in clients}
    per_client = {int(r["client_id"]): float(r["sleep_time"]) for r in sleeps}

    per_tier: dict[str, float] = {}
    for tier in sorted(set(tiers.values())):
        values = {per_client[c] for c, t in tiers.items() if t == tier}
        if len(values) != 1:
            fail(f"tier {tier} has more than one sleep time: {sorted(values)}")
            continue
        per_tier[tier] = values.pop()
        ok(f"{tier:8s} T = {per_tier[tier]:8.4f} s")

    if "xavier" in per_tier and "thor" in per_tier:
        gap = per_tier["xavier"] / per_tier["thor"]
        ok(f"nominal gap xavier/thor  {gap:.2f}x")

    _report_implied_efficiency(per_tier)
    return per_tier


# FLOPs per sample per epoch for ResNet-18 on 32x32, forward + backward.
# Mirrors gpu_tier_sleep.W_MODEL_FLOPS; kept local so preflight needs no imports.
W_MODEL_FLOPS = 1.1e9


def _report_implied_efficiency(per_tier: dict[str, float]) -> None:
    """State the physical claim the sleep times are making (ADR-0004).

    The tier formula divides a FLOPs budget by peak throughput times
    `eta_nominal`, and the CSVs set eta_nominal = 1.0 -- i.e. 100% of peak FP16,
    which nothing achieves. The anchor absorbs the whole difference, so the real
    assumption is an *effective* efficiency. It used to be inert; with SUMO it
    sets T_train against a fixed coverage window and therefore decides the waste
    rate, so it is printed on every run rather than left in a CSV column.
    """
    tiers_csv = None
    for arm in ("gpu", "baseline"):
        for path in HERE.glob(f"configs/{arm}_*.toml"):
            with path.open("rb") as f:
                tiers_csv = Path(tomllib.load(f)["gpu_tier"]["tiers_csv"])
            break
        if tiers_csv:
            break
    if tiers_csv is None or not tiers_csv.exists():
        return

    partition = 2500
    for path in HERE.glob("configs/gpu_*.toml"):
        with path.open("rb") as f:
            partition = tomllib.load(f)["data"]["partition_size"]
        break

    for row in read_csv(tiers_csv):
        tier = row["tier_id"]
        if tier not in per_tier:
            continue
        peak_flops = float(row["gpu_tflops_fp16"]) * 1e12
        raw = W_MODEL_FLOPS * partition / (peak_flops * float(row["eta_nominal"]))
        ok(
            f"{tier:8s} implied efficiency {100 * raw / per_tier[tier]:5.2f}% of peak "
            f"(raw {1000 * raw:7.2f} ms -> {per_tier[tier]:.3f} s)"
        )
    ok("^ the physical claim these runs make. See ADR-0004.")


def check_round_clock(
    clients: list[dict], per_tier: dict[str, float], per_round: int
) -> None:
    print("\n[3] derived round clock")
    tiers = [r["tier_id"] for r in clients]
    times = [per_tier[t] for t in tiers if t in per_tier]

    # "queue" model: C slots kept full, each a renewal process of mean E[T].
    expected_t = statistics.mean(times)
    t_round = expected_t / per_round
    ok(f"E[T]                     {expected_t:.4f} s")
    ok(f"T_round (queue, C={per_round})     {t_round:.4f} s")
    for tier in sorted(per_tier):
        ok(f"tau_hat {tier:8s}         {per_tier[tier] / t_round:.4f} rounds")


def check_waste(coverage: list[dict], per_tier: dict[str, float]) -> None:
    print("\n[4] expected waste (dispatch uniform within a coverage window)")
    residences = [float(r["residence_s"]) for r in coverage]
    ok(f"residence                mean {statistics.mean(residences):.2f} s, "
       f"median {statistics.median(residences):.1f} s")

    for tier in sorted(per_tier):
        duration = per_tier[tier]
        success = sum(max(0.0, w - duration) / w for w in residences) / len(residences)
        ok(f"{tier:8s} T = {duration:6.3f} s -> waste {100 * (1 - success):5.1f}%")


def check_time_budget(
    coverage: list[dict], clients: list[dict], per_tier: dict[str, float],
    per_round: int, rounds: int,
) -> None:
    """Will the run outlast the trace?

    Three different clocks get confused here, so be explicit:

      * REAL time is how long the GPU is busy (~11 h). The trace knows nothing
        about it.
      * The SIMULATED clock is `server.wall_time`, which advances by simulated
        training times. This is the one mapped onto the trace.
      * TRACE time is the `t` column, 0 .. ~36 000 s.

    The simulated clock advances by roughly E[T_selected] / concurrency per
    dispatch cycle, and `rounds` counts only the cycles that aggregated, so

        simulated seconds  ~  rounds / (1 - waste) * E[T_selected] / C

    E[T_selected] depends on how hard the arm steers toward fast vehicles, so
    this is a bracket, not a number. Past the end of the trace the oracle wraps
    and the mobility pattern repeats -- sound for a closed loop in steady state,
    but it should be a decision, not a surprise.

    Measured on the 200-round test builds (n20, 2026-09-06), for reference: the
    gpu arm ran at 1.853 simulated s per round (51% of the trace over 10 000
    rounds) and the cpu arm at 2.263 s (63%). Both land inside the bracket below,
    which is what says the model is right.
    """
    print("\n[6] simulated-time budget against the trace")
    t_end = max(float(r["t_exit_s"]) for r in coverage)
    ok(f"trace covers             {t_end:.0f} s ({t_end / 3600:.1f} h of mobility)")

    tiers = [r["tier_id"] for r in clients]
    expected_t = statistics.mean(per_tier[t] for t in tiers if t in per_tier)
    residences = [float(r["residence_s"]) for r in coverage]

    # Pool-average waste, i.e. what a selector with no tier preference would see.
    waste = sum(
        (1 - sum(max(0.0, w - per_tier[t]) / w for w in residences) / len(residences))
        for t in tiers
        if t in per_tier
    ) / len(tiers)

    for label, mean_t in (
        ("steers to fast tiers", expected_t * 0.7),
        ("no steering (pool mean)", expected_t),
        ("steers to slow tiers", expected_t * 1.3),
    ):
        seconds = rounds / (1 - waste) * mean_t / per_round
        verdict = "fits" if seconds < t_end else "WRAPS the trace"
        line = (
            f"{label:<24} {seconds:8.0f} s ({seconds / 3600:4.1f} h)  "
            f"{100 * seconds / t_end:5.1f}% of trace  -> {verdict}"
        )
        (ok if seconds < t_end else fail)(line)


def check_configs(cell: str) -> tuple[int, int]:
    print("\n[5] configs")
    per_round = 0
    rounds = 0
    for arm in ("baseline", "gpu", "cpu", "observed"):
        path = HERE / "configs" / f"{arm}_{cell}.toml"
        if not path.exists():
            fail(f"{path.name} is missing")
            continue
        with path.open("rb") as f:
            config = tomllib.load(f)

        per_round = config["clients"]["per_round"]
        rounds = config["trainer"]["rounds"]
        problems = []
        for key, expected in (
            ("comm_simulation", True),
            ("sleep_simulation", True),
            ("speed_simulation", True),
            ("avg_training_time", 0.0),
        ):
            if config["clients"].get(key) != expected:
                problems.append(f"clients.{key}")
        if config["clients"].get("compute_comm_time"):
            problems.append("clients.compute_comm_time must stay off")
        if per_round != config["trainer"]["max_concurrency"]:
            problems.append("per_round != trainer.max_concurrency")
        for section, key in (("sumo", "coverage_csv"), ("gpu_tier", "clients_csv")):
            referenced = Path(config[section][key])
            if not referenced.exists():
                problems.append(f"{section}.{key} -> {referenced} does not exist")

        if problems:
            fail(f"{path.name}: {'; '.join(problems)}")
        else:
            ok(f"{path.name:22s} port {config['server']['port']}")

    return per_round, rounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--cell", choices=("n20", "n100"), default="n20")
    args = parser.parse_args()

    print(f"Preflight for cell {args.cell}")
    clients, coverage, sleeps = check_files(args.cell)
    per_tier = check_speed_model(clients, sleeps)
    per_round, rounds = check_configs(args.cell)
    if per_round:
        check_round_clock(clients, per_tier, per_round)
    check_waste(coverage, per_tier)
    if per_round and rounds:
        check_time_budget(coverage, clients, per_tier, per_round, rounds)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} problem(s) found.")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
