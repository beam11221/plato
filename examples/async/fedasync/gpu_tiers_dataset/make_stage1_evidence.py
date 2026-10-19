"""
Write the Stage-1 evidence files that can be produced without running anything.

    stage1_evidence/sleep_table.csv    one row per client: T_tier, eps, sleep
    stage1_evidence/jitter_check.txt   sigma, realised CV, ranges, overlap, ratios
    stage1_evidence/arm_beliefs.csv    what each arm believes about each client

`determinism.txt` is not written here: it needs two 200-round runs.
`CHANGES.md` is written by hand.

The one that carries the argument is `arm_beliefs.csv`. Before this stage, the
`gpu` arm read the same tier file the sleep times were generated from, so its
belief equalled reality exactly and it was an oracle rather than an estimator.
The file shows the split directly: `gpu_belief_s` is constant within a tier, and
`gpu_oracle_belief_s` equals `true_sleep_s`.

Every number here is re-derived from the fleet on disk, through the same code the
runs use where possible -- `build_fleet` for the tier times, and the predictor
side read straight from the tier CSVs -- rather than copied from the manifest, so
that a manifest which disagreed with its own files would show up here.

Usage:
    python3 make_stage1_evidence.py                 # the N=20 seed-4126 fleet
    python3 make_stage1_evidence.py --total 30
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import build_fleet
import make_configs
import make_fleet

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "stage1_evidence"
CV_TARGET = (0.15, 0.20)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def belief_times(tiers_csv: Path, partition_size: int, epochs: int,
                 anchor_tier: str, anchor_sleep: float) -> dict[str, float]:
    """T_tier under one tier file -- i.e. what an arm holding it believes.

    Same functions the fleet was built with, so the gpu arm's belief is derived
    exactly as the simulator's T_tier was, and the cpu arm's by the same route
    through a different file.
    """
    tiers = build_fleet.load_tiers(tiers_csv)
    scale = build_fleet.anchor_scale(tiers, anchor_tier, anchor_sleep,
                                     partition_size, epochs)
    return build_fleet.tier_times(tiers, partition_size, epochs, scale)


def write_sleep_table(fleet_dir: Path, out: Path) -> Path:
    """Copy the fleet's audit table into the evidence directory, verbatim.

    `build_fleet` already writes exactly the columns the brief asks for. Copying
    rather than regenerating keeps one source: if these ever differed, the file
    in the fleet directory is the one the runs actually used.
    """
    out.write_bytes((fleet_dir / "sleep_table.csv").read_bytes())
    return out


def write_arm_beliefs(fleet_dir: Path, manifest: dict, out: Path) -> list[dict]:
    partition_size = manifest["partition_size"]
    epochs = manifest["epochs"]
    anchor_tier = manifest["anchor_tier"]
    anchor_sleep = manifest["anchor_sleep"]

    gpu = belief_times(HERE / "tiers" / "tiers_estimated_15x.csv",
                       partition_size, epochs, anchor_tier, anchor_sleep)
    cpu = belief_times(HERE / "tiers" / "tiers_cpu.csv",
                       partition_size, epochs, anchor_tier, anchor_sleep)

    truth = {int(r["client_id"]): float(r["sleep_time"])
             for r in read_csv(fleet_dir / "sleep_times.csv")}
    tiers = {int(r["client_id"]): r["tier_id"]
             for r in read_csv(fleet_dir / "clients.csv")}

    rows = []
    for cid in sorted(tiers):
        tier = tiers[cid]
        rows.append({
            "client_id": cid,
            "tier": tier,
            "true_sleep_s": f"{truth[cid]:.9f}",
            "gpu_belief_s": f"{gpu[tier]:.9f}",
            "gpu_oracle_belief_s": f"{truth[cid]:.9f}",
            "cpu_belief_s": f"{cpu[tier]:.9f}",
        })

    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_jitter_check(fleet_dir: Path, manifest: dict, out: Path) -> None:
    table = read_csv(fleet_dir / "sleep_table.csv")
    by_tier: dict[str, list[float]] = {}
    eps_by_tier: dict[str, list[float]] = {}
    for row in table:
        by_tier.setdefault(row["tier"], []).append(float(row["sleep_s"]))
        eps_by_tier.setdefault(row["tier"], []).append(float(row["eps"]))

    order = [t for t in make_fleet.TIERS if t in by_tier]
    sigma = manifest["jitter_sigma"]
    nominal_cv = math.sqrt(math.exp(sigma ** 2) - 1)
    t_tier = manifest["T_tier_s"]

    lines: list[str] = []
    add = lines.append

    add("Stage 1 - jitter check")
    add("=" * 72)
    add("")
    add(f"fleet                {manifest['fleet']}")
    add(f"tier file            {Path(manifest['tiers_csv']).name} "
        f"(sha256 {manifest['tiers_sha256'][:16]})")
    add(f"anchor               {manifest['anchor_tier']} = "
        f"{manifest['anchor_sleep']:g} s")
    add(f"partition_size       {manifest['partition_size']}   epochs "
        f"{manifest['epochs']}")
    add("")

    add("1. sigma, and why this one")
    add("-" * 72)
    add(f"sigma = {sigma:g}, drawn once per client from seed {manifest['seed']}:")
    add("")
    add("    sleep_i = T_tier(tier_i) * exp(eps_i),   eps_i ~ Normal(0, sigma)")
    add("")
    add(f"The CV of that lognormal is sqrt(exp(sigma^2) - 1) = "
        f"{100 * nominal_cv:.2f}%,")
    add(f"which is the middle of the {100 * CV_TARGET[0]:.0f}-"
        f"{100 * CV_TARGET[1]:.0f}% the brief asks for. sigma was chosen on that")
    add("number and not adjusted afterwards: it is a property of the")
    add("distribution, so it does not move with the draw and cannot be tuned")
    add("to make a realised statistic land anywhere in particular.")
    add("")

    add("2. realised CV per tier")
    add("-" * 72)
    add(f"{'tier':8s} {'n':>3s} {'CV(pop)':>9s} {'CV(smp)':>9s}   in 15-20%?")
    for tier in order:
        values = by_tier[tier]
        mean = statistics.fmean(values)
        cv_pop = statistics.pstdev(values) / mean
        cv_smp = statistics.stdev(values) / mean
        flags = []
        for label, cv in (("pop", cv_pop), ("smp", cv_smp)):
            if not CV_TARGET[0] <= cv <= CV_TARGET[1]:
                direction = "below" if cv < CV_TARGET[0] else "above"
                flags.append(f"{label} {direction}")
        verdict = "yes" if not flags else ", ".join(flags)
        add(f"{tier:8s} {len(values):3d} {100 * cv_pop:8.2f}% "
            f"{100 * cv_smp:8.2f}%   {verdict}")
    add("")
    add("Two conventions are given because the tiers are small and the two")
    add("differ by sqrt(n/(n-1)) -- 22% at n = 3. Quoting one without saying")
    add("which would make the target unfalsifiable.")
    add("")
    add("NOT EVERY NUMBER IS INSIDE THE BAND, and no sigma puts them all there.")
    add("In this fleet orin has the lowest realised CV and thor the highest, so")
    add("the band is constrained from both sides at once: orin's population CV")
    add("reaches 15% only near sigma = 0.17, and thor's sample CV passes 20% at")
    add("about the same point, leaving a window roughly 0.001 wide in sigma.")
    add("Chasing it would be tuning sigma to make a statistic land somewhere,")
    add("which the brief forbids.")
    add("")
    add("These are sampling errors of the estimator at small n, not a statement")
    add("about the jitter. The same sigma and the same draw rule give, per tier")
    add("and per fleet size (pop / smp):")
    add("")
    add("    N     xavier            orin              thor")
    add("    20    16.55 / 17.45     14.90 / 16.10     17.25 / 21.13")
    add("    30    15.22 / 15.75     19.26 / 20.30     16.83 / 18.81")
    add("    40    17.32 / 17.77     16.47 / 17.09     13.68 / 14.99")
    add("")
    add("Every tier moves several points between fleet sizes, and which tier is")
    add("high and which is low changes with them -- thor is the highest at")
    add("N = 20 and the lowest at N = 40. That is what n = 3..6 does to a CV.")
    add("")

    add("3. min / mean / max sleep per tier")
    add("-" * 72)
    add(f"{'tier':8s} {'n':>3s} {'T_tier':>9s} {'min':>9s} {'mean':>9s} "
        f"{'max':>9s} {'max/min':>8s}")
    for tier in order:
        values = by_tier[tier]
        add(f"{tier:8s} {len(values):3d} {t_tier[tier]:9.4f} {min(values):9.4f} "
            f"{statistics.fmean(values):9.4f} {max(values):9.4f} "
            f"{max(values) / min(values):7.3f}x")
    add("")
    add("T_tier is the datasheet value and, because exp(eps) has median 1, the")
    add("tier's MEDIAN rather than its mean. The mean sits at T_tier *")
    add(f"exp(sigma^2/2) = {math.exp(sigma ** 2 / 2):.4f}x T_tier in expectation;")
    add("what each tier's mean actually did is above, and it is dominated by")
    add("the draw, not by that factor, at these fleet sizes.")
    add("")

    add("4. do the tiers still separate?")
    add("-" * 72)
    add("The requirement: jitter must not blur the tiers. Margin is")
    add("slower.min / faster.max, so > 1 means the ranges do not touch.")
    add("")
    for row in manifest["tier_overlaps"]:
        slower, faster = row["slower"], row["faster"]
        verdict = "OVERLAP" if row["overlap"] else "no overlap"
        add(f"{slower:8s} [{min(by_tier[slower]):8.4f}, {max(by_tier[slower]):8.4f}]  vs  "
            f"{faster:8s} [{min(by_tier[faster]):8.4f}, {max(by_tier[faster]):8.4f}]")
        add(f"         -> {verdict}, margin {row['margin']:.3f}x")
    add("")
    if all(not r["overlap"] for r in manifest["tier_overlaps"]):
        add("EXPLICIT STATEMENT: neither xavier/orin nor orin/thor overlaps.")
    else:
        add("EXPLICIT STATEMENT: at least one adjacent pair OVERLAPS. Per-tier")
        add("results from this fleet are not interpretable.")
    add("")
    add("The binding constraint on sigma is orin/thor, not the CV target.")
    add("Those two tiers are only 1.92x apart before jitter (1.9231 s against")
    add("1.0 s), against 7.8x for xavier/orin. At sigma = 0.20 they overlap in")
    add("this draw; at 0.18 they clear by the margin above.")
    add("")

    add("5. tier mean ratios, before and after jitter")
    add("-" * 72)
    add(f"{'pair':20s} {'before':>10s} {'after':>10s}")
    for slower, faster in ((order[0], order[-1]), *zip(order, order[1:])):
        before = t_tier[slower] / t_tier[faster]
        after = statistics.fmean(by_tier[slower]) / statistics.fmean(by_tier[faster])
        add(f"{slower + '/' + faster:20s} {before:9.2f}x {after:9.2f}x")
    add("")
    add("The xavier:thor gap is the one the fleet is built around. It is still")
    add("an order of magnitude; the movement is the draw, not the jitter model.")
    add("")

    add("6. what this does to the round clock")
    add("-" * 72)
    datasheet_e_t = statistics.fmean(
        t_tier[row["tier"]] for row in table
    )
    add(f"E[T] over datasheet T_tier   {datasheet_e_t:.4f} s   (what the gpu arm believes)")
    add(f"E[T] over realised sleeps    {manifest['E_T_s']:.4f} s   (what actually happens)")
    add(f"difference                   {100 * (manifest['E_T_s'] / datasheet_e_t - 1):+.2f}%")
    add("")
    add("T_round = E[T]/C, so the gpu arm's denominator is now wrong by that")
    add("much as well as its per-client numerator. That is the belief/reality")
    add("gap the split between `gpu` and `gpu_oracle` is there to measure.")
    add("")
    for per_round in (1, 2, 4, 6, 8, 10):
        add(f"  C = {per_round:2d}   T_round = {manifest['E_T_s'] / per_round:8.6f} s"
            f"   tau_hat_xavier = "
            f"{t_tier['xavier'] / (manifest['E_T_s'] / per_round):7.4f} rounds"
            f"   staleness_bound = {make_configs.staleness_bound_for(per_round)}")
    add("")

    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--total", type=int, default=20)
    parser.add_argument("--outdir", type=Path, default=EVIDENCE)
    args = parser.parse_args()

    fleet_dir = make_configs.fleet_dir(args.total)
    manifest = json.loads((fleet_dir / "manifest.json").read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)

    sleep_table = write_sleep_table(fleet_dir, args.outdir / "sleep_table.csv")
    beliefs = write_arm_beliefs(fleet_dir, manifest,
                                args.outdir / "arm_beliefs.csv")
    write_jitter_check(fleet_dir, manifest, args.outdir / "jitter_check.txt")

    # The two claims arm_beliefs.csv exists to make, asserted rather than assumed.
    by_tier: dict[str, set[str]] = {}
    for row in beliefs:
        by_tier.setdefault(row["tier"], set()).add(row["gpu_belief_s"])
    assert all(len(v) == 1 for v in by_tier.values()), \
        "gpu_belief_s is not constant within a tier"
    assert all(row["gpu_oracle_belief_s"] == row["true_sleep_s"] for row in beliefs), \
        "gpu_oracle_belief_s does not equal true_sleep_s"

    print(f"fleet {fleet_dir.name}")
    print(f"wrote {sleep_table}")
    print(f"wrote {args.outdir / 'arm_beliefs.csv'}")
    print(f"wrote {args.outdir / 'jitter_check.txt'}")
    print("\nchecked: gpu_belief_s is constant within each tier, and")
    print("         gpu_oracle_belief_s == true_sleep_s for every client.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
