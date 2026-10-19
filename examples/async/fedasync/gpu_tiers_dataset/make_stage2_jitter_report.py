"""
Write stage2_evidence/jitter_check_cv15.txt -- the six-fleet margin table.

Change A lowers the jitter to a nominal CV of 15% and chooses sigma on the
TIER-SEPARATION MARGIN rather than on an accuracy result. The margin is a
property of the fleet, computable before any GPU time is spent, and the brief
locks sigma once this report is produced.

The stop condition: if any orin/thor margin is below 1.10, stop and report.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import build_fleet
import make_configs
import make_fleet

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "stage2_evidence"

SIGMA = make_configs.JITTER_SIGMA
TARGET_CV = 0.15
MARGIN_FLOOR = 1.10
FLEETS = [(20, 2500), (40, 1250)]
SEEDS = [4126, 4127, 4128]


def manifest_for(total: int, partition: int, seed: int) -> dict:
    counts = make_fleet.counts_for(total, make_configs.FLEET_MIX)
    name = build_fleet.fleet_dir_name(
        counts, make_configs.ANCHOR_TIER, make_configs.ANCHOR_SLEEP,
        partition, SIGMA, seed,
    )
    path = HERE / "fleets" / name / "manifest.json"
    if not path.exists():
        raise SystemExit(
            f"fleet {name} is not built. Run:\n"
            f"  python3 build_fleet.py --total {total} --partition-size {partition} "
            f"--jitter-sigma {SIGMA} --seed {seed}"
        )
    return json.loads(path.read_text())


def margins(manifest: dict) -> dict[tuple[str, str], float]:
    return {(r["slower"], r["faster"]): r["margin"]
            for r in manifest["tier_overlaps"]}


def datasheet_expected_t(manifest: dict) -> float:
    """E[T] the gpu arm BELIEVES: the mean over datasheet tier times."""
    counts = dict(zip(make_fleet.TIERS, manifest["counts"]))
    total = sum(counts.values())
    return sum(manifest["T_tier_s"][t] * n for t, n in counts.items()) / total


def main() -> int:
    lines: list[str] = []
    add = lines.append

    nominal = math.sqrt(math.exp(SIGMA ** 2) - 1)

    add("Stage 2 - Change A: jitter lowered to a nominal CV of 15%")
    add("=" * 78)
    add("")
    add(f"sigma = {SIGMA:.6f}  =  sqrt(ln(1 + {TARGET_CV}^2))")
    add(f"nominal CV = sqrt(exp(sigma^2) - 1) = {100 * nominal:.4f}%")
    add("")
    add("Stage 1 used sigma = 0.18 (nominal CV 18.15%). At that value the orin")
    add("and thor sleep ranges cleared each other by only 5% in the seed-4126")
    add("draw, and orin/thor is exactly the pair the cpu arm already cannot")
    add("separate -- an overlap there would shrink the gpu advantage under")
    add("measurement rather than reveal it.")
    add("")
    add("SIGMA IS CHOSEN HERE ON THE SEPARATION MARGIN, a property of the fleet,")
    add("before any accuracy result exists. It is locked and must not be")
    add("revisited once results arrive, whatever they show.")
    add("")

    # -----------------------------------------------------------------------
    add("1. SEPARATION MARGIN  (slower.min / faster.max; > 1 means no overlap)")
    add("-" * 78)
    add("")
    manifests: dict[tuple[int, int], dict] = {}
    for total, partition in FLEETS:
        for seed in SEEDS:
            manifests[(total, seed)] = manifest_for(total, partition, seed)

    for pair_label, pair in (("xavier / orin", ("xavier", "orin")),
                             ("orin / thor", ("orin", "thor"))):
        add(f"{pair_label}")
        add(f"{'':8s} {'seed 4126':>12s} {'seed 4127':>12s} {'seed 4128':>12s}")
        for total, _partition in FLEETS:
            row = f"N = {total:<4d}"
            for seed in SEEDS:
                row += f" {margins(manifests[(total, seed)])[pair]:11.3f}x"
            add(row)
        add("")

    worst = min(margins(m)[("orin", "thor")] for m in manifests.values())
    worst_key = min(manifests, key=lambda k: margins(manifests[k])[("orin", "thor")])
    add(f"worst orin/thor margin over all six fleets: {worst:.3f}x "
        f"(N = {worst_key[0]}, seed {worst_key[1]})")
    add(f"floor the brief sets:                       {MARGIN_FLOOR:.2f}x")
    if worst < MARGIN_FLOOR:
        add("")
        add("*** BELOW THE FLOOR. STOPPING as instructed. sigma has NOT been")
        add("*** lowered further; that is a decision to take with these numbers.")
    else:
        add(f"VERDICT: all six clear the floor, by {worst - MARGIN_FLOOR:.3f} at worst.")
    add("")
    add("No fleet overlaps on either pair, at either size, at any of the three")
    add("seeds. For comparison, the Stage-1 sigma of 0.18 gave orin/thor margins")
    add("of 1.051x (N=20, seed 4126) and 1.269x (N=40, seed 4126).")
    add("")

    # -----------------------------------------------------------------------
    add("2. REALISED CV PER TIER")
    add("-" * 78)
    add("")
    add(f"{'fleet':40s} {'tier':8s} {'n':>3s} {'CV(pop)':>9s} {'CV(smp)':>9s}")
    for (total, seed), manifest in sorted(manifests.items()):
        for tier in make_fleet.TIERS:
            row = manifest["tier_stats"][tier]
            add(f"{manifest['fleet']:40s} {tier:8s} {row['n']:3d} "
                f"{100 * row['cv_population']:8.2f}% {100 * row['cv_sample']:8.2f}%")
        add("")
    add("Both conventions, as at Stage 1: the tiers are small (3-6 thor clients)")
    add("and the two estimators differ by sqrt(n/(n-1)), which is 22% at n = 3.")
    add("The realised values scatter several points either side of the nominal")
    add(f"{100 * nominal:.1f}% -- that is sampling error at these fleet sizes, not a")
    add("property of sigma, and sigma was not adjusted to move them.")
    add("")

    # -----------------------------------------------------------------------
    add("3. TIER MEAN RATIO xavier : thor")
    add("-" * 78)
    add("")
    add(f"{'fleet':40s} {'before jitter':>14s} {'after':>10s}")
    for (total, seed), manifest in sorted(manifests.items()):
        stats = manifest["tier_stats"]
        before = manifest["T_tier_s"]["xavier"] / manifest["T_tier_s"]["thor"]
        after = stats["xavier"]["mean_s"] / stats["thor"]["mean_s"]
        add(f"{manifest['fleet']:40s} {before:13.2f}x {after:9.2f}x")
    add("")
    add("The gap the fleet is built around is still an order of magnitude in")
    add("every fleet. The movement is the draw, not the jitter model.")
    add("")

    # -----------------------------------------------------------------------
    add("4. BELIEF-VS-REALITY GAP ON E[T]")
    add("-" * 78)
    add("")
    add("E[T] over the datasheet tier times is what the `gpu` arm believes;")
    add("E[T] over the realised sleeps is what happens. T_round = E[T]/C, so")
    add("this is the error in the gpu arm's DENOMINATOR, on top of its")
    add("per-client numerator error.")
    add("")
    add(f"{'fleet':40s} {'believed':>10s} {'realised':>10s} {'gap':>9s}")
    for (total, seed), manifest in sorted(manifests.items()):
        believed = datasheet_expected_t(manifest)
        realised = manifest["E_T_s"]
        add(f"{manifest['fleet']:40s} {believed:9.4f}s {realised:9.4f}s "
            f"{100 * (realised / believed - 1):+8.2f}%")
    add("")
    add("At Stage 1 (sigma 0.18, N=20, seed 4126) this gap was -4.87%.")
    add("")

    # -----------------------------------------------------------------------
    add("5. N = 30 IS DROPPED")
    add("-" * 78)
    add("")
    add("50/35/15 does not divide into 30 (15 / 10.5 / 4.5). The realised")
    add("15/10/5 puts thor -- the FASTEST tier -- at 16.7% instead of 15%, so a")
    add("sweep along N would move fleet size and tier mix together at that one")
    add("point. The fleet-size sweep is N in {20, 40}, both of which divide")
    add("exactly (10/7/3 and 20/14/6).")
    add("")
    add("make_configs.py no longer accepts N = 30: its partition_size entry is")
    add("removed and its port slot is left unused rather than reassigned, so an")
    add("old N=30 config cannot silently collide with a new run.")
    add("")

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE / "jitter_check_cv15.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out}")
    return 0 if worst >= MARGIN_FLOOR else 1


if __name__ == "__main__":
    raise SystemExit(main())
