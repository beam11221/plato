"""
Regenerate the COMMUNICATION-LATENCY sweep (BRIEF_comm_sweep.md).

Third sweep alongside the H-sweep and the fleet-mix sweep.  Fleet pinned at
50/35/15, gap pinned at G=30 (xavier 20 s, orin 3.651 s, thor 0.667 s,
compute-only effective gap 8.25x).  Only the uplink bandwidth B moves:

    C = D / B     (D = 341.52 Mbit = 42.69 MB, the measured ResNet-18 payload)

added identically to REALITY (every client's sleep, via generate_sleep_times.py)
and to BELIEF (gpu_tops_predict_seconds, via [gpu_tier] comm_bandwidth_mbps).
See gpu_tier_sleep.comm_latency_seconds() / comm_in_predictor() for the code and
BRIEF_comm_sweep.md § 4 for the landmines (t_round_model MUST be "queue"; C is
added AFTER predictor_scale(), never before).

The B = inf (C = 0) control is NOT regenerated here — it is the existing
`m50_35_15` cell of the fleet-mix sweep at G=30 (runs/mix_sweep/*_m50_35_15),
which already used t_round_model="queue", rounds=5000, target_accuracy=1.01,
seed 8888.  Re-running it would cost ~6 h for nothing (BRIEF § 3).

10 ordinary cells (gpu+cpu at 4 bandwidths, baseline at the 2 lowest) plus 1
belief-vs-reality ablation: gpu at B=10 with comm_in_predictor=false, reusing
B=10's sleep CSV (reality is unchanged; only belief is miscalibrated) under a
distinct run identity ("...b010nc") so it does not collide with the calibrated
B=10 cell.

Usage:
    python3 make_comm_sweep.py             # generate sleep CSVs + configs_comm/
    python3 make_comm_sweep.py --check      # validate only, write nothing
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tomllib
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import make_configs  # noqa: E402
from generate_sleep_times import SleepTimeGenerator, write_sleep_time_csv  # noqa: E402

GAP = "30"
ANCHOR = 20
MIX = "m50_35_15"
DATASET_SIZE = 500
EPOCHS = 1
PAYLOAD_MBIT = 341.52
ROUNDS = 5000
TARGET_ACCURACY = 1.01
T_ROUND_MODEL = "queue"
# Must match [clients] max_sleep_time in the templates (BRIEF § 4.3): Config's
# simulate_client_speed() silently clips any sleep at this value, so a bandwidth
# low enough to push xavier's sleep (20 s) + C past it would corrupt reality with
# no warning in the log.
MAX_SLEEP_TIME = 120.0

# (label, Mbps).  B = inf (C = 0) is the reused mix-sweep control — not here.
BANDWIDTHS = [("b100", 100.0), ("b050", 50.0), ("b020", 20.0), ("b010", 10.0)]
# baseline only sanity-anchors the two lowest bandwidths (BRIEF § 3).
BASELINE_LABELS = {"b020", "b010"}
# The belief-vs-reality ablation reuses this bandwidth's sleep CSV.
ABLATION_BW_LABEL = "b010"
ABLATION_MBPS = 10.0
ABLATION_RUN_LABEL = "b010nc"

TIERS_CSV = _HERE / "input" / f"tiers_geo_gap{GAP}.csv"
CLIENTS_CSV = _HERE / "input" / f"clients_{MIX}.csv"
OUTDIR = _HERE / "configs_comm"
RUNS_DIR = "comm_sweep"


def make_sleep_csv(label: str, mbps: float) -> tuple[Path, float]:
    """Reality: the G=30 / 50-35-15 sleep CSV, with C = D/B added to every client."""
    out = _HERE / "output" / f"client_sleep_times_xa{ANCHOR}_gap{GAP}_{MIX}_{label}.csv"
    gen = SleepTimeGenerator(TIERS_CSV, CLIENTS_CSV)
    scale = gen.anchor_scale("xavier", ANCHOR, DATASET_SIZE, EPOCHS)
    comm_latency = PAYLOAD_MBIT / mbps
    sleeps = gen.sleep_times(DATASET_SIZE, EPOCHS, scale, comm_latency)
    write_sleep_time_csv(out, sleeps)
    worst = max(sleeps.values())
    if worst >= MAX_SLEEP_TIME:
        raise SystemExit(
            f"{label}: worst-case sleep {worst:.2f}s >= max_sleep_time="
            f"{MAX_SLEEP_TIME:g}s -- PLATO's Config.simulate_client_speed() would "
            f"silently clip it (BRIEF_comm_sweep.md § 4.3). Raise [clients] "
            f"max_sleep_time in templates/ before running B={mbps:g} Mbps."
        )
    xavier = next(v for cid, v in sleeps.items() if cid == 1)
    orin = next(v for cid, v in sleeps.items() if cid == 51)
    thor = next(v for cid, v in sleeps.items() if cid == 86)
    print(
        f"  {label:<6} B={mbps:>5g} Mbps  C={comm_latency:7.4f}s  "
        f"xavier={xavier:7.4f}s orin={orin:7.4f}s thor={thor:7.4f}s worst={worst:7.4f}s"
    )
    return out, comm_latency


def cell(
    arm: str, bw_index: int, bw_label: str, mbps: float,
    run_label: str = "", comm_in_predictor: bool = True,
) -> tuple[str, str, dict, int]:
    name, text, port = make_configs.render(
        arm, GAP, ANCHOR, root=make_configs.DEFAULT_ROOT,
        rounds=ROUNDS, runs_dir=RUNS_DIR, mix=MIX, t_round_model=T_ROUND_MODEL,
        target_accuracy=TARGET_ACCURACY, bw_index=bw_index, bw_label=bw_label,
        run_label=run_label, comm_bandwidth_mbps=mbps, payload_mbit=PAYLOAD_MBIT,
        comm_in_predictor=comm_in_predictor,
    )
    cfg = tomllib.loads(text)
    return name, text, cfg, port


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--check", action="store_true", help="Validate only; write nothing.")
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    print(f"Comm-latency sweep: G={GAP}, fleet={MIX} (queue model, rounds={ROUNDS})\n")
    print("1. Reality: one sleep CSV per bandwidth (C added on top of the G=30 tier times)")
    for label, mbps in BANDWIDTHS:
        make_sleep_csv(label, mbps)
    print()

    print("2. Configs")
    rendered: dict[str, tuple[str, dict, int]] = {}
    for bw_index, (label, mbps) in enumerate(BANDWIDTHS, start=1):
        arms = ["gpu", "cpu"] + (["baseline"] if label in BASELINE_LABELS else [])
        for arm in arms:
            name, text, cfg, port = cell(arm, bw_index, label, mbps)
            rendered[name] = (text, cfg, port)

    # The belief-vs-reality ablation (BRIEF § 3): gpu only, its own bw_index so it
    # gets its own port even though it shares B=10's sleep CSV.
    ablate_index = len(BANDWIDTHS) + 1
    name, text, cfg, port = cell(
        "gpu", ablate_index, ABLATION_BW_LABEL, ABLATION_MBPS,
        run_label=ABLATION_RUN_LABEL, comm_in_predictor=False,
    )
    rendered[name] = (text, cfg, port)

    problems = make_configs.validate(rendered, check_paths=True)
    if problems:
        print(f"Refusing to write -- {len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    print(f"Validated {len(rendered)} configs: TOML parses, all inputs exist, "
          f"ports unique, arms comparable.")
    if args.check:
        print("--check given; nothing written.")
        return 0

    if args.outdir.exists():
        shutil.rmtree(args.outdir)
    args.outdir.mkdir(parents=True)
    for name, (text, _, _) in sorted(rendered.items()):
        (args.outdir / name).write_text(text)

    print(f"\nWrote {len(rendered)} files to {args.outdir}/\n")
    for name in sorted(rendered):
        print(f"  {name}  (port {rendered[name][2]})")

    print(
        f"\nThe B=inf (C=0) control is NOT regenerated -- it is runs/mix_sweep/"
        f"{{gpu,cpu,baseline}}_xa{ANCHOR}_gap{GAP}_{MIX} (BRIEF_comm_sweep.md § 3)."
    )
    print(
        f"\nRun as two lanes (HANDOFF § 4d), split so every cell shares the same\n"
        f"contention mode for its whole run:\n\n"
        f"  # lane hi: the two higher-bandwidth cells + the ablation (gpu/cpu, 5 runs)\n"
        f"  CONFIG_DIR=$PWD/{args.outdir.name} ./run_sweep.sh gpu cpu --gaps {GAP} "
        f'--mixes "{MIX}_b100 {MIX}_b050" --keep-rounds 20 --clean-interval 3 && \\\n'
        f"  CONFIG_DIR=$PWD/{args.outdir.name} ./run_sweep.sh gpu --gaps {GAP} "
        f'--mixes "{MIX}_{ABLATION_RUN_LABEL}" --keep-rounds 20 --clean-interval 3\n\n'
        f"  # lane lo: the two lower-bandwidth cells, all three arms (6 runs)\n"
        f"  CONFIG_DIR=$PWD/{args.outdir.name} ./run_sweep.sh gpu cpu baseline --gaps {GAP} "
        f'--mixes "{MIX}_b020 {MIX}_b010" --keep-rounds 20 --clean-interval 3\n'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
