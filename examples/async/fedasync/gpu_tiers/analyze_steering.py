"""
Does the selector actually steer toward fast clients, and is the trend real?

The sweep's headline claim is that as the tier-speed gap G widens, the `gpu` arm
(belief == reality) shifts its priority from data quality toward speed, pulling
away from `cpu` (belief wrong) and `baseline` (no belief).  Two things have to be
true for that claim to mean anything, and this script checks both.

1. MEASURED — does it steer?
   Per tier, the share of dispatches it won against its share of the pool.
   ``lift = share_dispatched / share_of_pool``; >1 means preferred over what
   random selection would give.  Parsed from the ``selected clients [...] with
   V_i scores {...}`` lines, so it is what the selector actually did, not a
   model of it.

   G=1 is the control: all three tiers sleep 20 s there, so tau_hat is identical
   for everyone and the speed term *cannot* discriminate.  Whatever lift shows at
   G=1 is the no-signal baseline, and it should be ~1.00 for every tier.

2. COUNTERFACTUAL — is the trend an artefact of the T_round bug?
   The ``"min"`` T_round model under-estimates tau_hat by a factor that decays
   with G (816% -> 11%, HANDOFF § 4.11), while the steering effect grows with G.
   Both move together, so part of the observed trend could be the estimator
   getting more accurate rather than the hardware gap getting wider — a confound
   pointing in exactly the direction that flatters the hypothesis.

   G_i does not depend on T_round, so in ``V_i = G_i / sqrt(1 + tau_hat_i)`` the
   entire speed preference between two tiers is::

       pref(fast over slow) = sqrt(1 + tau_hat_slow) / sqrt(1 + tau_hat_fast)

   Computing that under both T_round models at every gap says whether the shape
   of the trend depends on the bug.  It does not: both models give 1.00 at G=1
   and ~3.3-3.4 at G=30, and "queue" is *stronger* in between — so the runs made
   under "min" are a conservative lower bound on the method's own steering.

   The `cpu` arm is included because its belief is ONE file shared by all gaps
   (tiers_cpu.csv, 1.30x spread), so its preference is constant in G by
   construction — a flat control the `gpu` arm is measured against.

Usage:
    python3 analyze_steering.py                    # all finished runs, both parts
    python3 analyze_steering.py --arm gpu cpu      # once the cpu arm exists
    python3 analyze_steering.py --counterfactual   # part 2 only, no logs needed
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
LOGS = _HERE / "logs"
CLIENTS_CSV = _HERE / "input" / "clients.csv"

TIER_ORDER = ("xavier", "orin", "thor")  # slowest -> fastest
GAPS = ("01", "10", "20", "30", "40")

SEL_RE = re.compile(
    r"Round (\d+) — selected clients \[([^\]]*)\] with V_i scores \{([^}]*)\}"
)

# Anchored tier sleeps (s) — must match output/client_sleep_times_xa20_gapNN.csv.
SLEEPS = {
    "01": {"xavier": 20.0, "orin": 20.0000, "thor": 20.0000},
    "10": {"xavier": 20.0, "orin": 6.3246, "thor": 2.0000},
    "20": {"xavier": 20.0, "orin": 4.4721, "thor": 1.0000},
    "30": {"xavier": 20.0, "orin": 3.6515, "thor": 0.6667},
    "40": {"xavier": 20.0, "orin": 3.1623, "thor": 0.5000},
}
# The cpu arm's belief: tiers_cpu.csv anchored on xavier = 20 s.  One file for
# every gap — xavier and orin both clock 2.00 GHz so it cannot separate them.
CPU_BELIEF = {"xavier": 20.0, "orin": 20.0, "thor": 15.3846}

TIER_N = {"xavier": 50, "orin": 35, "thor": 15}
CLIENTS_PER_ROUND = 10
CONCURRENCY = 9  # measured in-flight depth; check_run.py reports conc=9


def t_round(belief: dict, model: str) -> float:
    """T_round under either model — mirrors gpu_tier_sleep._t_round_kernel."""
    pool = [belief[t] for t in TIER_N for _ in range(TIER_N[t])]
    if model == "queue":
        return sum(pool) / len(pool) / min(CLIENTS_PER_ROUND, len(pool))
    total, cdf, out = len(pool), 0.0, 0.0
    for v in sorted(set(pool)):
        p = pool.count(v) / total
        out += v * ((1 - cdf) ** CLIENTS_PER_ROUND - (1 - cdf - p) ** CLIENTS_PER_ROUND)
        cdf += p
    return out


def speed_preference(belief: dict, model: str, fast: str, slow: str) -> float:
    """How much more the speed term of V_i favours `fast` over `slow`."""
    tr = t_round(belief, model)
    return math.sqrt(1 + belief[slow] / tr) / math.sqrt(1 + belief[fast] / tr)


def measure(log_path: Path, tier_of: dict) -> tuple[dict, dict]:
    """Return (dispatch counts per tier, V_i samples per tier) from one log."""
    disp: dict = defaultdict(int)
    v_by_tier: dict = defaultdict(list)
    with log_path.open(errors="ignore") as fh:
        for line in fh:
            m = SEL_RE.search(line)
            if not m:
                continue
            for pair in m.group(3).split(","):
                if ":" not in pair:
                    continue
                cid_s, v_s = pair.split(":")
                tier = tier_of.get(int(cid_s.strip()))
                if tier:
                    disp[tier] += 1
                    v_by_tier[tier].append(float(v_s.strip()))
    return disp, v_by_tier


def part1_measured(arms: list[str], tier_of: dict, pool_share: dict) -> None:
    print("=" * 74)
    print("1. MEASURED — dispatch lift per tier  (share dispatched / share of pool)")
    print("   >1 = preferred over random.  G=1 is the control: no speed signal there.")
    print("=" * 74)

    for arm in arms:
        rows = []
        for gap in GAPS:
            log = LOGS / f"{arm}_xa20_gap{gap}.log"
            if not log.exists():
                continue
            disp, v_by_tier = measure(log, tier_of)
            total = sum(disp.values())
            if not total:
                continue
            for tier in TIER_ORDER:
                if not disp[tier]:
                    continue
                share = disp[tier] / total
                rows.append({
                    "G": gap, "tier": tier, "disp%": share,
                    "lift": share / pool_share[tier],
                    "mean V_i": pd.Series(v_by_tier[tier]).mean(),
                })
        if not rows:
            print(f"\n[{arm}] no logs yet.")
            continue

        df = pd.DataFrame(rows)
        print(f"\n[{arm}]")
        piv = df.pivot(index="G", columns="tier", values="lift")[list(TIER_ORDER)]
        print("  dispatch lift")
        print(piv.round(2).to_string().replace("\n", "\n  "))
        vpiv = df.pivot(index="G", columns="tier", values="mean V_i")[list(TIER_ORDER)]
        print("  mean V_i")
        print(vpiv.round(4).to_string().replace("\n", "\n  "))

        # What "correct" looks like depends on what this arm can SEE.  The gpu
        # arm's belief carries the real tier spread, so it should order
        # thor > orin > xavier.  The cpu arm's belief is tiers_cpu.csv — a 1.30x
        # spread in which xavier and orin are identical (both clock 2.00 GHz) —
        # so its speed term is nearly inert and a FLAT ordering is the expected
        # result, not a failure.  Flagging it as one would invert the finding.
        expected_pref = {"gpu": None, "cpu": CPU_BELIEF}.get(arm, CPU_BELIEF)
        for gap in piv.index:
            belief = SLEEPS[gap] if arm == "gpu" else expected_pref
            if belief is None:
                belief = SLEEPS[gap]
            pref = speed_preference(belief, "min", "thor", "xavier")
            ordered = piv.loc[gap, "thor"] > piv.loc[gap, "orin"] > piv.loc[gap, "xavier"]
            if pref < 1.2:
                verdict = ("flat as expected" if not ordered
                           else "ordered, though the signal is too weak to expect it")
            else:
                verdict = "ordered" if ordered else "!! ORDERING BROKEN"
            print(f"  G={gap}: speed signal pref={pref:.2f} -> {verdict}")


def part2_counterfactual() -> None:
    print()
    print("=" * 74)
    print("2. COUNTERFACTUAL — speed preference for thor over xavier,")
    print("   under the buggy T_round ('min') and the correct one ('queue')")
    print("=" * 74)
    print(f"\n{'G':>4} {'T_rnd min':>10} {'T_rnd queue':>12} "
          f"{'pref min':>9} {'pref queue':>11} {'queue/min':>10}")
    print("-" * 62)
    for gap, sleeps in SLEEPS.items():
        pm = speed_preference(sleeps, "min", "thor", "xavier")
        pq = speed_preference(sleeps, "queue", "thor", "xavier")
        print(f"{gap:>4} {t_round(sleeps, 'min'):10.3f} {t_round(sleeps, 'queue'):12.3f}"
              f" {pm:9.2f} {pq:11.2f} {pq / pm:10.2f}")

    pm_cpu = speed_preference(CPU_BELIEF, "min", "thor", "xavier")
    pq_cpu = speed_preference(CPU_BELIEF, "queue", "thor", "xavier")
    print(f"\n cpu {t_round(CPU_BELIEF, 'min'):10.3f} {t_round(CPU_BELIEF, 'queue'):12.3f}"
          f" {pm_cpu:9.2f} {pq_cpu:11.2f} {pq_cpu / pm_cpu:10.2f}"
          "   <- same at every G: one belief file for the whole sweep")

    print("\nReading:")
    print("  Both models give 1.00 at G=1 and 3.3-3.4 at G=30, so the trend runs")
    print("  between the same endpoints either way — the bug does not manufacture it.")
    print("  In between, 'queue' steers HARDER (+18% at G=10, +10% at G=20, +3% at")
    print("  G=30), so those runs UNDERSTATE the method's own steering.  At G=40 it")
    print("  crosses over and 'min' overstates by 2% — small, but it means 'always a")
    print("  lower bound' is not quite right: it is a lower bound for G<=30 only.")
    print("  Note the saturation — pref grows 1.00 -> 2.17 -> 2.84 -> 3.31 -> 3.70,")
    print("  and the effective gap saturates too (1.0/5.5/7.3/8.25/8.8).  Expect the")
    print("  G=40 point to flatten; that is the anchor design, not a null result.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--arm", nargs="+", default=["gpu", "cpu", "baseline"],
                        help="arms to report (default: all three)")
    parser.add_argument("--counterfactual", action="store_true",
                        help="part 2 only — needs no run logs")
    args = parser.parse_args()

    if not args.counterfactual:
        if not CLIENTS_CSV.exists():
            print(f"missing {CLIENTS_CSV}", file=sys.stderr)
            return 1
        clients = pd.read_csv(CLIENTS_CSV)
        tier_of = dict(zip(clients.client_id, clients.tier_id))
        pool_share = clients.tier_id.value_counts(normalize=True).to_dict()
        part1_measured(args.arm, tier_of, pool_share)

    part2_counterfactual()
    return 0


if __name__ == "__main__":
    sys.exit(main())
