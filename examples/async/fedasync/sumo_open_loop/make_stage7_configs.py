"""
Render the 8 Stage 7 configs: {baseline, gpu, cpu, observed} x p_recur {0, 1}.

Everything outside `[sumo_open]` is held at the values every cell since Stage 3
has used, so the only thing that varies across the eight is the arm and the
identity policy.  In particular `partition_size = 2500` and
`total_clients = M = 20`, which is what keeps `M x partition_size = 50,000`.

Stage 0 is absent by construction — there is no `stage0_*` key and the open
server has no filter to enable.  That is deliberate: this cell tests the
scoring claim, and the filter is already known to dominate the selector.

Stage 7b (`--stage 7b`) renders the same eight into `configs/stage7b/`, with
run directories `stage7b_*`, its own port range, and the Stage 3-6 fleet files
that `p_recur = 1` binds each slot's identity to.  `--rounds N --tag preflight`
renders the short smoke-run set for the Step 2 pre-flight.

Usage:
    python3 make_stage7_configs.py                       # Stage 7 (as run)
    python3 make_stage7_configs.py --stage 7b            # Stage 7b, 4000 rounds
    python3 make_stage7_configs.py --stage 7b --rounds 100 --tag preflight
    python3 make_stage7_configs.py --stage 8 --seed 4127          # Stage 8
    python3 make_stage7_configs.py --stage 8 --seed 4127 --rounds 100 --tag preflight

Stage 8 varies only the RUN seed ([clients]/[server]/[data] random_seed, all
equal, as in Stage 6) and pins `[sumo_open] physics_seed = 4126`, so every
trace vehicle's sleep is byte-identical to Stage 7b.  Configs go to
`configs/stage8_s<seed>[_<tag>]/`, runs to `stage8_s<seed>[_<tag>]_<arm>_<regime>`,
so two seeds can never share a directory, a log or a port.

Stage 9 (`--stage 9 --seed N`) is Stage 8's layout with the neutral priors:
`[server] polaris_alpha = 1.0` (= g_prior_alpha; the pre-existing key) and
`[server] observed_unseen_tau = "fleet_running_mean"`, for gpu / cpu /
observed; seed 4126 also gets `baseline` (the invariance run).  Stage 8's
missing `cpu` cells at alpha = 10: `--stage 8 --seed N --arms cpu`.

Stage 10 (`--stage 10 --variant V --seed N`) adds the open-population Stage 0
gate ([sumo_open] stage0_*; open_stage0.py):
    f1   filter on, alpha = 1 + unseen-tau running mean  baseline gpu cpu observed
    f10  filter on, alpha = 10 (Polaris default package)  gpu cpu observed
    nf   no filter, pure random                           baseline
T_hat: tier_spec for baseline / gpu (GPU spec) and cpu (its CPU spec);
observed_per_vehicle for observed.  Prefix stage10<V>_s<N>.

Stage 11 (`--stage 11 --variant f1|nf --p-return P --seed N`): Stage 10 Wave A's
settings (filter on, alpha = 1, running-mean unseen tau) on p_recur = 0 plus
[sumo_open] p_return = P and the commuter pool (the Stage 3-10 fleet files).
Prefix stage11f_p<PPP>_s<N> / stage11nf_p<PPP>_s<N>; run names <arm>_ret.
Cells: gpu, baseline (+F) at 0.25 / 0.5 / 0.75 / 1.0; observed, cpu (+F) and
baseline / none at 0.25 / 0.5 / 0.75.
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
GPU_TIERS_DATASET = HERE.parent / "gpu_tiers_dataset"
TRACE = HERE / "sumo_trace" / "e1_open_poisson_seed4126_range400"

SEED = 4126
SLOTS = 20               # M
PARTITION_SIZE = 2500    # M x partition_size = 50,000
PER_ROUND = 4            # C
ROUNDS = 4000
CONCENTRATION = 0.8
ANCHOR_SLEEP = 15
PORT_BASE = {"7": 18640, "7b": 18660, "7b_preflight": 18680,
             "8_s4127": 18700, "8_s4128": 18720,
             "8_s4127_preflight": 18740, "8_s4128_preflight": 18760,
             "9_s4126": 18780, "9_s4127": 18800, "9_s4128": 18820,
             "9_s4126_preflight": 18840,
             "10f1_s4126": 18900, "10f1_s4127": 18920, "10f1_s4128": 18940,
             "10f10_s4126": 18960, "10f10_s4127": 18980, "10f10_s4128": 19000,
             "10nf_s4126": 19020, "10nf_s4127": 19040, "10nf_s4128": 19060,
             "10f1_s4126_preflight": 19080, "10f10_s4126_preflight": 19100}
STAGE11_P = (0.25, 0.5, 0.75, 1.0)


def stage11_arms(variant: str, p_return: float) -> tuple:
    if variant == "nf":
        return () if p_return == 1.0 else ("baseline",)
    return ("gpu", "baseline") if p_return == 1.0 else ("baseline", "gpu", "cpu", "observed")


def stage11_port(variant: str, p_return: float, seed: int, arm: str, preflight: bool) -> int:
    base = 19600 if preflight else 19200
    return (base + 100 * (seed - 4126) + 20 * STAGE11_P.index(p_return)
            + (10 if variant == "nf" else 0) + list(ARMS).index(arm))


STAGE10_ARMS = {"f1": ("baseline", "gpu", "cpu", "observed"),
                "f10": ("gpu", "cpu", "observed"), "nf": ("baseline",)}
STAGE0_T_HAT = {"baseline": "tier_spec", "gpu": "tier_spec", "cpu": "tier_spec",
                "observed": "observed_per_vehicle"}
# Stage 9's two never-seen priors (both arms of every comparison get them).
STAGE9_KNOBS = (
    "# Stage 9 neutral priors: g_prior_alpha (= polaris_alpha) and the reactive\n"
    "# arm's tau for a never-seen vehicle.  baseline reads neither.\n"
    'observed_unseen_tau = "fleet_running_mean"\n'
)
PHYSICS_SEED = 4126  # Stage 8: pinned; sleeps must equal Stage 7b's
CLOSED_LOOP_INPUT = HERE.parent / "sumo_close_loop"
FLEET_CLIENTS_CSV = CLOSED_LOOP_INPUT / "input" / "clients_n20r400.csv"
FLEET_SLEEP_CSV = CLOSED_LOOP_INPUT / "output" / "client_sleep_times_n20r400.csv"

# arm -> (staleness_predictor, tiers_csv)
ARMS: dict[str, tuple[str | None, str]] = {
    "baseline": (None, "tiers_estimated_15x.csv"),
    "gpu": ("gpu_tops", "tiers_estimated_15x.csv"),
    "cpu": ("cpu", "tiers_cpu.csv"),
    "observed": ("observed", "tiers_estimated_15x.csv"),
}
P_RECUR = {"open": 0.0, "closed": 1.0}

TEMPLATE = """\
# =============================================================================
# Stage {stage} -- {arm} arm, p_recur={p_recur} ({regime} population)
# =============================================================================
# Generated by sumo_open_loop/make_stage7_configs.py. Do not edit by hand;
# edit the generator so all 8 configs of this stage move together.
#
# Open-population trace, M={slots} slots, C={per_round}, {rounds} rounds, seed {seed}.
# Stage 0 filter is ABSENT by construction (see the generator's docstring).
#
# Run:
#   cd examples/async/fedasync
#   PYTHONPATH=gpu_tiers:gpu_tiers_dataset:.:sumo_close_loop:sumo_open_loop \\
#       python3 sumo_open_loop/sumo_open_fedasync.py -c {config_path}
# =============================================================================

[general]
base_path = "{base_path}"

[clients]
type = "simple"
total_clients = {slots}
per_round = {per_round}
do_test = false

speed_simulation = true
comm_simulation = true
sleep_simulation = true
avg_training_time = 0.0
max_sleep_time = 300
random_seed = {seed}

[server]
address = "127.0.0.1"
port = {port}
synchronous = false
simulate_wall_time = true
periodic_interval = 1
ping_interval = 30
ping_timeout = 3600

staleness_bound = 25
minimum_clients_aggregated = 1
mixing_hyperparameter = 0.5
adaptive_mixing = true
eval_interval = 10
random_seed = {seed}
{polaris_block}
[server.staleness_weighting_function]
type = "polynomial"
a = 0.5

# p_recur is the identity policy: 0 wipes a slot's server-side state on every
# recycle (open population), 1 retains it (closed control on the same trace).
[sumo_open]
coverage_csv = "{coverage_csv}"
p_recur = {p_recur}
slots = {slots}
{fleet_block}
[data]
datasource = "Torchvision"
dataset_name = "CIFAR10"
download = true
data_path = "{data_path}"
partition_size = {partition_size}
sampler = "noniid"
concentration = {concentration}
random_seed = {seed}

[trainer]
type = "basic"
rounds = {rounds}
max_concurrency = {per_round}
target_accuracy = 0.9
model_name = "resnet_18"
epochs = 1
batch_size = 64
optimizer = "SGD"

[algorithm]
type = "fedavg"

[parameters]

[parameters.optimizer]
lr = 0.01
momentum = 0.9
weight_decay = 0.0005

[results]
types = "round, elapsed_time, accuracy, train_loss, valid_loss"
record_clients_accuracy = false

[gpu_tier]
tiers_csv   = "{tiers_csv}"
clients_csv = "{clients_csv}"
anchor_tier  = "xavier"
anchor_sleep = {anchor_sleep}
t_round_model = "queue"
"""


def render(
    arm: str, regime: str, outdir: Path, port: int,
    stage: str = "7", rounds: int = ROUNDS, run_prefix: str = "stage7",
    seed: int = SEED, variant: str = "", p_return: float | None = None,
    cell_suffix: str = "",
) -> Path:
    predictor, tiers_csv = ARMS[arm]
    p_recur = P_RECUR[regime]
    name = f"{arm}_{cell_suffix or regime}"
    config_path = outdir / f"{name}.toml"

    # Stage 7b: at p_recur = 1 each slot IS one Stage 3-6 fleet vehicle.
    fleet_block = ""
    if stage != "7" and p_recur == 1.0:
        fleet_block = (
            "# p_recur = 1 binds slot k to Stage 3-6 fleet vehicle k: its tier\n"
            "# and its sleep.  The trace supplies only when it is in coverage.\n"
            f'fleet_clients_csv = "{FLEET_CLIENTS_CSV}"\n'
            f'fleet_sleep_csv = "{FLEET_SLEEP_CSV}"\n'
        )
    if stage == "11":
        fleet_block += (
            "# Stage 11: partial recurrence. p_return = share of allocations that\n"
            "# go to a returning commuter (the Stage 3-10 fleet, K = 20).\n"
            f"p_return = {p_return}\n"
            f'fleet_clients_csv = "{FLEET_CLIENTS_CSV}"\n'
            f'fleet_sleep_csv = "{FLEET_SLEEP_CSV}"\n'
        )
    if stage in ("10", "11") and variant in ("f1", "f10"):
        fleet_block += (
            "# Stage 10: the open-population Stage 0 gate (open_stage0.py).\n"
            f'stage0_t_hat = "{STAGE0_T_HAT[arm]}"\n'
            'stage0_r_i = "vmax_distance"\n'
            "coverage_radius_m = 400.0\n"
            "v_max_mps = 19.444444\n"
            "rsu_x_m = 600.0\n"
            f'positions_csv = "{TRACE / "trace.csv"}"\n'
        )
    if stage in ("8", "9", "10", "11"):
        fleet_block += (
            f"# Stage {stage}: the RUN seed above varies; the PHYSICS seed (vehicle\n"
            "# sleeps, allocator) is pinned to Stage 7b's.\n"
            f"physics_seed = {PHYSICS_SEED}\n"
        )

    neutral = stage == "9" or (stage in ("10", "11") and variant == "f1")
    alpha = "1.0" if neutral else "10.0"
    polaris_block = (
        f'staleness_predictor = "{predictor}"\n'
        f"polaris_alpha = {alpha}\n"
        "polaris_initial_gradient_bound = 0.5\n"
        "polaris_initial_staleness = 0.01\n"
        if predictor
        else ""
    )
    if stage == "9":
        polaris_block += STAGE9_KNOBS if predictor else (
            "polaris_alpha = 1.0\n" + STAGE9_KNOBS)
    if stage in ("10", "11") and variant == "f1" and predictor:
        polaris_block += STAGE9_KNOBS

    text = TEMPLATE.format(
        stage=stage, arm=arm, regime=regime, p_recur=p_recur,
        config_path=config_path.relative_to(HERE.parent),
        base_path=f"./runs/sumo_open_loop/{run_prefix}_{name}",
        slots=SLOTS, per_round=PER_ROUND, rounds=rounds, seed=seed,
        port=port, polaris_block=polaris_block, fleet_block=fleet_block,
        coverage_csv=TRACE / "coverage_windows.csv",
        data_path=HERE.parent / "data",
        partition_size=PARTITION_SIZE, concentration=CONCENTRATION,
        tiers_csv=GPU_TIERS_DATASET / "tiers" / tiers_csv,
        # The open cell has no fixed fleet, but gpu_tier_sleep still wants a
        # clients.csv to resolve the anchor from; the closed cell's 20-client
        # file has the right tier ids and the same anchor tier.
        clients_csv=HERE.parent / "sumo_close_loop" / "input" / "clients_n20r400.csv",
        anchor_sleep=ANCHOR_SLEEP,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text)
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["7", "7b", "8", "9", "10", "11"], default="7")
    parser.add_argument("--p-return", type=float, default=None, help="Stage 11 only")
    parser.add_argument("--variant", choices=["f1", "f10", "nf"], default=None,
                        help="Stage 10 only")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="restrict to these arms (e.g. Stage 8's missing cpu)")
    parser.add_argument("--seed", type=int, default=SEED, help="run seed (Stage 8)")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--tag", default="", help='e.g. "preflight"')
    args = parser.parse_args()

    if args.stage == "8" and args.seed == SEED:
        raise SystemExit("Stage 8 re-uses 7b for seed 4126; pass --seed 4127 or 4128.")
    if args.stage not in ("8", "9", "10", "11") and args.seed != SEED:
        raise SystemExit("--seed is only for Stages 8-11.")
    if (args.stage in ("10", "11")) != (args.variant is not None):
        raise SystemExit("--variant is required for Stages 10-11 and only for them.")
    if args.stage == "11":
        if args.p_return not in STAGE11_P or args.variant not in ("f1", "nf"):
            raise SystemExit(f"Stage 11: --p-return in {STAGE11_P}, --variant f1|nf.")
        v = "f" if args.variant == "f1" else "nf"
        key = f"11{v}_p{int(round(args.p_return * 100)):03d}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
        outdir = HERE / "configs" / f"stage{key}"
        paths = [render(arm, "open", outdir, stage11_port(args.variant, args.p_return, args.seed, arm, bool(args.tag)),
                        stage="11", rounds=args.rounds, run_prefix=f"stage{key}", seed=args.seed,
                        variant=args.variant, p_return=args.p_return, cell_suffix="ret")
                 for arm in stage11_arms(args.variant, args.p_return)]
        print(f"Wrote {len(paths)} configs:")
        for pth in paths:
            print(f"  {pth}")
        return
    if args.stage == "10":
        stage_key = f"10{args.variant}_s{args.seed}"
    elif args.stage in ("8", "9"):
        stage_key = f"{args.stage}_s{args.seed}"
    else:
        stage_key = args.stage
    key = stage_key + (f"_{args.tag}" if args.tag else "")
    run_prefix = "stage" + key
    outdir = HERE / "configs" / run_prefix
    paths, port = [], PORT_BASE[key]
    for regime in P_RECUR:
        for arm in ARMS:
            wanted = args.arms or (
                list(STAGE10_ARMS[args.variant]) if args.stage == "10" else
                [a for a in ARMS if a != "cpu"] if args.stage == "8"
                else [a for a in ARMS if a != "baseline" or args.seed == SEED]
                if args.stage == "9" else list(ARMS))
            if arm not in wanted:
                continue
            # A fixed per-arm offset: rendering a subset never reuses a port.
            arm_port = (PORT_BASE[key] + len(ARMS) * list(P_RECUR).index(regime)
                        + list(ARMS).index(arm) + (50 if args.arms else 0))
            paths.append(render(arm, regime, outdir, arm_port, stage=args.stage,
                                rounds=args.rounds, run_prefix=run_prefix,
                                seed=args.seed, variant=args.variant or ""))
            port += 1

    if not (TRACE / "coverage_windows.csv").exists():
        raise SystemExit(f"missing {TRACE}/coverage_windows.csv")

    print(f"Wrote {len(paths)} configs:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
