"""
Stage 8, Step 1: pre-flight for a new RUN seed with the PHYSICS seed pinned.

Runs every Stage 7b check unchanged (`preflight_stage7b.check_run`), then adds
the two Stage 8 checks that only mean something together:

  physics pinned   every dispatched vehicle's sleep equals its Stage 7b
                   (seed 4126) value -- `check_run` already recomputes the
                   expected sleep with seed 4126, so its sleep rows ARE this
                   check -- and at p_recur = 1 every slot's identity / tier /
                   sleep / partition equals Stage 7b's fleet.

  seed changed     the logged training seed equals identity_seed(RUN seed, ...)
                   and differs from identity_seed(4126, ...) for the same
                   (identity, round); and the first-round training loss
                   differs from the Stage 7b run's.

`check_run`'s own seed row compares against 4126, so for a new run seed it
must read 0%, and that row is inverted here rather than edited in place.

Usage:
    python3 preflight_stage8.py --seed 4127 gpu_open gpu_closed
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from occupant_identity import identity_seed  # noqa: E402
from preflight_stage7b import check_run  # noqa: E402

FEDASYNC = HERE.parent
RUNS = FEDASYNC / "runs" / "sumo_open_loop"
SEED_LINE = re.compile(
    r"\[ClientSeededTrainer\] client_id=\d+ identity_id=(\S+) round=(\d+) -> seed (\d+)"
)
SEEDS_LINE = re.compile(r"\[Stage8\] seeds: run_seed=(\d+) .* physics_seed=(\d+)")
FINGERPRINT = re.compile(r"\[Stage7\] client slot=\d+ .* partition=(\d+) sampler_id=\d+ fingerprint=(\S+)")
SEVENB_SEED = 4126


def first_loss(run_dir: Path) -> float | None:
    files = sorted((run_dir / "results").glob("*.csv"))
    if not files:
        return None
    with files[0].open() as fh:
        for row in csv.DictReader(fh):
            return float(row["train_loss"])
    return None


def fingerprints(log: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in log.open(errors="ignore"):
        m = FINGERPRINT.search(line)
        if m:
            out.setdefault(int(m.group(1)), m.group(2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default="preflight")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()

    prefix = f"stage8_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    failed, out = False, {}
    for name in args.runs:
        arm, regime = name.split("_")
        log = HERE / "logs" / f"{prefix}_{name}.log"
        res = check_run(log, 1.0 if regime == "closed" else 0.0,
                        proactive=arm in ("gpu", "cpu"), polaris=arm != "baseline")
        checks = res["checks"]

        # Invert the 7b seed row: it compares to 4126, which must now NEVER match.
        seed_row = checks.pop("seed == identity_seed(identity, round)")
        rows = {k: (v["ok"], v["total"]) for k, v in checks.items()}

        run_ok = run_tot = diff_ok = 0
        seeds_logged = None
        for line in log.open(errors="ignore"):
            m = SEED_LINE.search(line)
            if m:
                ident, rnd, seed = m.group(1), int(m.group(2)), int(m.group(3))
                run_tot += 1
                run_ok += seed == identity_seed(args.seed, ident, rnd)
                diff_ok += seed != identity_seed(SEVENB_SEED, ident, rnd)
                continue
            m = SEEDS_LINE.search(line)
            if m:
                seeds_logged = (int(m.group(1)), int(m.group(2)))
        rows["[8] seed == identity_seed(RUN seed, identity, round)"] = (run_ok, run_tot)
        rows["[8] seed != Stage 7b seed for same (identity, round)"] = (diff_ok, run_tot)
        rows["[8] (7b row, must be 0) seed == identity_seed(4126, ...)"] = (
            seed_row["total"] - seed_row["ok"], seed_row["total"])
        rows["[8] logged seeds: run=%d physics=4126" % args.seed] = (
            int(seeds_logged == (args.seed, SEVENB_SEED)), 1)

        loss8 = first_loss(RUNS / f"{prefix}_{name}")
        loss7b = first_loss(RUNS / f"stage7b_{name}")
        rows["[8] first-round train_loss differs from 7b"] = (
            int(loss8 is not None and loss7b is not None and loss8 != loss7b), 1)

        # Data partitions: the INDEX is pinned (checked above); the Dirichlet
        # label draw follows [data] random_seed = run seed, as in Stage 6.
        fp8 = fingerprints(log)
        fp7 = fingerprints(HERE / "logs" / f"stage7b_{name}.log")
        common = sorted(set(fp8) & set(fp7))
        same_fp = sum(fp8[k] == fp7[k] for k in common)

        print(f"\n=== {prefix}_{name}: {res['dispatches']} dispatches, "
              f"{res['recycles']} recycles, {res['state_resets']} resets, "
              f"startup T_round {res['startup_t_round']}")
        for cname, (ok, tot) in sorted(rows.items()):
            good = tot > 0 and ok == tot
            failed |= not good
            print(f"   [{'PASS' if good else 'FAIL'}] {cname:58s} {ok:4d}/{tot:<4d}")
        print(f"   first-round train_loss: 8={loss8}  7b={loss7b}")
        print(f"   partition label draws equal to 7b: {same_fp}/{len(common)} "
              f"(informational: [data] random_seed = run seed, as in Stage 6)")
        out[name] = {"rows": {k: list(v) for k, v in rows.items()},
                     "first_loss": [loss8, loss7b],
                     "fingerprints_equal_to_7b": [same_fp, len(common)],
                     "startup_t_round": res["startup_t_round"]}

    dest = FEDASYNC / "gpu_tiers_dataset" / "stage8_evidence" / f"preflight_s{args.seed}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}\n{'PRE-FLIGHT FAILED' if failed else 'PRE-FLIGHT PASSED: all checks 100%'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
