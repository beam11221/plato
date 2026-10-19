"""
Render the arm configs for the gpu_tiers_dataset programme.

What is different here from `../separability/` and `../separability_n100/`
-------------------------------------------------------------------------
1. **Five arms, not four.** `gpu` was an oracle by construction -- it read the
   same tier file the sleep times came from, so its belief equalled reality
   exactly. The generator now applies a persistent per-client `exp(eps_i)` that
   no datasheet can know, which leaves `gpu` holding the TIER MEAN (the
   deployable method, what the thesis claims) and adds `gpu_oracle`, which holds
   the per-client truth (the upper bound, not deployable).

2. **Inputs come from a fleet DIRECTORY, not from two loose CSVs.** A cell names
   one directory under `fleets/`; the fleet, the sleep times, the audit table and
   the manifest are inside it and cannot be crossed with another build. See
   `build_fleet.py`.

3. **`staleness_bound` is derived from C**, by the rule
   `../separability_n100/make_configs.py` established: 6.25 bound-units per slot.
   The C sweep here runs to C = 10, and a flat 25 would sit close to the routine
   staleness of the xavier tier there -- the aggregator would start censoring
   exactly the tier under measurement (gpu_tiers/HANDOFF.md 4.9).

4. **One entry point, `fedasync_seeded.py`**, selected by `--arm`, and it seeds
   every RNG from the config's own `random_seed`. The older folders run two
   unseeded entry scripts, which is where the 6.29 % run-to-run clock spread in
   `../separability/PARALLELISM_CHECK.md` came from.

What is PINNED, and is not yours to change (ADR-0003, ADR-0004)
--------------------------------------------------------------
fleet mix 50/35/15 in contiguous ID blocks, CIFAR-10 / ResNet-18,
SGD lr 0.01 / momentum 0.9 / wd 5e-4, epochs 1, batch 64,
`tiers_estimated_15x.csv` (3 tiers, 15x nominal gap), `anchor_sleep = 15`,
`comm_simulation = true`, `avg_training_time = 0.0`, `t_round_model = "queue"`,
`eval_interval = 10`, `target_accuracy = 0.9` (= no early stop in practice).

Usage:
    python3 make_configs.py --alphas 0.8 --seeds 4126 --concurrency 4
    python3 make_configs.py --alphas 0.8 --clients 30 --concurrency 4
    python3 make_configs.py --arms gpu --rounds 200 --tag _det1
"""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path

import build_fleet
import make_fleet

HERE = Path(__file__).resolve().parent
FEDASYNC = HERE.parent
TIERS_DIR = HERE / "tiers"
FLEETS_DIR = HERE / "fleets"

# arm -> (staleness_predictor or None, tiers_csv filename, holds_oracle)
#
# The tier file is the arm's BELIEF about hardware; `holds_oracle` is whether it
# additionally gets per-client ground truth. Exactly one arm may.
ARMS: dict[str, tuple[str | None, str, bool]] = {
    "baseline":   (None,         "tiers_estimated_15x.csv", False),
    "gpu":        ("gpu_tops",   "tiers_estimated_15x.csv", False),
    "cpu":        ("cpu",        "tiers_cpu.csv",           False),
    "observed":   ("observed",   "tiers_estimated_15x.csv", False),
    "gpu_oracle": ("gpu_oracle", "tiers_estimated_15x.csv", True),
}

# Keys the arms of one cell are allowed to differ in. Anything else that differs
# is a bug: one axis moves per phase, and the arms differ only in where tau_hat
# comes from.
ARM_SPECIFIC = {
    ("general", "base_path"),
    ("server", "port"),
    ("server", "staleness_predictor"),
    ("gpu_tier", "tiers_csv"),
    ("gpu_tier", "oracle_sleep_csv"),
}

# The baseline runs fedasync_server.Server, which reads none of these, so writing
# them there would be a dead key implying an effect that does not exist
# (gpu_tiers/HANDOFF.md section 4.7).
BASELINE_MAY_LACK = {
    ("server", "staleness_predictor"),
    ("server", "polaris_alpha"),
    ("server", "polaris_initial_gradient_bound"),
    ("server", "polaris_initial_staleness"),
}

# ---------------------------------------------------------------------------
# The port plan
# ---------------------------------------------------------------------------
# Two servers on one port is a silent hang, not a crash, so every dimension gets
# its own digit and unknown values are refused rather than folded in. Blocks
# already in use elsewhere in this project: gap sweep 8100-8440, multi-seed
# 9000-10440, mix 11000+, comm 14000+, surgical 17115-17415 / 21015-23015 /
# 31015, SUMO 18100-18400 / 19100-19400, separability n20 25000-28325 and
# 34000-37325, separability n100 41000-44323 and 47000-50323.
#
# This programme sweeps N as well, so N takes a digit of its own:
#
#     real runs   51000 + 2000*alpha + 200*arm + 20*C + 5*N + seed_index
#     --tag runs  56000 + 1000*tag   + 200*arm + 20*C + 5*N + seed_index
#
# The tag block is 1000 wide, not 2000: the offset below it maxes out at
# 200*4 + 20*5 + 5*2 + 3 = 913, and six registered tags at 2000 apiece would run
# past the 16-bit port ceiling. Caught by
# tests/test_make_configs.py::test_no_port_exceeds_the_16_bit_maximum when _det3
# and _det4 were registered.
#
# Arm and C need more room than the older plans gave them: five arms and six
# concurrencies, against four and three.
#
# The TAG takes the digit alpha takes in a real run, because two probes of the
# same cell under different tags is the normal case -- `--tag _det1` and
# `--tag _det2` are the SAME config run twice, which is exactly the pair that
# must not share a port -- while two probes at different alphas is not. A tag is
# therefore registered like every other dimension, and within one tag alpha is
# not distinguished: use a different tag per probe cell.
PORT_BASE = 51000
PROBE_PORT_BASE = 56000
ALPHA_SLOT = {0.8: 0, 1.0: 1, 0.1: 2}
TAG_SLOT = {"_det1": 0, "_det2": 1, "_det3": 2, "_det4": 3, "_det5": 4, "_det6": 5,
           "_probe": 6, "_smoke": 7, "_rep2": 8}

# Throwaway determinism/probe tags, excluded from a bare (no --tag) invocation
# of `preflight.py` and `collect.py` by default -- they are verification runs
# already accounted for in stage1/stage2/step0 evidence, not part of the real
# campaign. `_rep2` is deliberately NOT in this set: it is a real, permanent
# measurement (Stage 3's own repeat of the C=4 gpu cell) and must be
# preflighted and collected exactly like every other real run.
THROWAWAY_TAGS = frozenset(TAG_SLOT) - {"_rep2"}
ARM_SLOT = {"baseline": 0, "gpu": 1, "cpu": 2, "observed": 3, "gpu_oracle": 4}
C_SLOT = {1: 0, 2: 1, 4: 2, 6: 3, 8: 4, 10: 5}
N_SLOT = {20: 0, 40: 2}   # 30 dropped; its slot is left unused, not reused
SEEDS = {4126: 1, 4127: 2, 4128: 3}

# The fleet each N is built with. N x partition_size is meant to hold CIFAR-10's
# 50,000 training images constant across the sweep; at N = 30 the brief's 1667
# gives 50,010, which is 10 images of overlap and is kept as specified.
# N = 30 is DROPPED at Stage 2: 50/35/15 does not divide into 30, and the
# realised 15/10/5 puts thor -- the fastest tier -- at 16.7% instead of 15%, so a
# sweep along N would move fleet size and tier mix together at that point.
PARTITION_SIZE = {20: 2500, 40: 1250}

ANCHOR_SLEEP = 15
ANCHOR_TIER = "xavier"
FLEET_MIX = (50, 35, 15)
# Locked at Stage 2: sigma for a NOMINAL CV of 15%,
# sigma = sqrt(ln(1 + 0.15^2)) = 0.149166. Chosen on the tier-separation margin --
# a property of the fleet -- before any accuracy result existed, and not to be
# revisited once results arrive. At the previous 0.18 the orin/thor ranges cleared
# by only 5% in the seed-4126 draw; orin/thor is exactly the pair the cpu arm
# already cannot separate. Worst margin over the six {N=20,40} x {4126,4127,4128}
# fleets is now 1.121x. See stage2_evidence/jitter_check_cv15.txt.
JITTER_SIGMA = 0.149166
FLEET_SEED = 4126
EPOCHS = 1

CONCURRENCY = 4
ROUNDS = 4000
EVAL_INTERVAL = 10
RUNS_DIR = "gpu_tiers_dataset"

# `staleness_bound` at C = 4 was a flat 25, and the template called it "inert".
# Inert is a property of the RATIO, not of the number: a report's staleness is
# roughly tau_hat = T_i / T_round and T_round = E[T]/C, so every staleness in the
# run scales with C. Crossing the bound does not drop a report -- PLATO's step 3
# flushes every pending client into ONE aggregation and jumps the simulated clock
# -- so a round would stop meaning one update and the staleness distribution
# under measurement would be truncated. Keeping the C=4 headroom exactly gives
# 25/4 = 6.25 bound-units per slot (../separability_n100/make_configs.py).
STALENESS_BOUND_PER_SLOT = 6.25

# Stage 3 Step 0: keep the host's processing_time out of finish_time, so the
# merge order is decided by the experiment and not by the machine. See
# runtime_patches.py. Written into every config so a run's own copy records it.
# This is the DEFAULT for render() / --exclude-processing-time; it is not the
# only source of truth -- a config's own `config_used.toml` is, always. Flip it
# here (or override per invocation) only after Step 0's own verification run
# passes; a config rendered before that verification must not silently claim
# the patch is on.
EXCLUDE_PROCESSING_TIME = True

TEMPLATE = """\
# =============================================================================
# gpu_tiers_dataset -- {arm} arm, alpha {alpha:g}, N {total_clients}, C {per_round}, seed {seed}
# =============================================================================
# Generated by gpu_tiers_dataset/make_configs.py. Do not edit by hand; edit the
# generator so every arm of every cell moves together.
#
# Fleet: {fleet_name}
#   sigma = {jitter_sigma:g} jitter, drawn once per client from seed {fleet_seed}.
#   Realised E[T] = {expected_t:.4f} s, so T_round = E[T]/C = {t_round:.6f} s and
#   tau_hat_xavier = {tau_hat_xavier:.4f} rounds. The server log must reproduce these.
#
# Run:
#   cd examples/async/fedasync
#   PYTHONPATH=<repo root>:gpu_tiers:. python3 {entry} --arm {arm} -c {config_path}
# =============================================================================

[general]
base_path = "{base_path}"

[clients]
type = "simple"
total_clients = {total_clients}
per_round = {per_round}
do_test = false

speed_simulation = true
# true means "charge nothing for communication" -- it routes the payload through
# a file and sets comm_time = 0 (plato/servers/base.py). With false, comm_time is
# a time.time() delta measured on whatever host ran the cell, which lands in the
# simulated clock and makes cells incomparable across machines (HANDOFF 4.8).
comm_simulation = true
sleep_simulation = true
# 0.0 so belief and reality coincide: the tier sleep is the whole training time.
# The key must be PRESENT and zero -- clients/strategies/defaults.py reads it
# unguarded, so deleting it is an AttributeError on the first report.
avg_training_time = 0.0
# Config.simulate_client_speed() clips every sleep at this value silently. The
# slowest client in any fleet built here is under 21 s, so this is inert.
max_sleep_time = 300
# THE seed. fedasync_seeded.py reads THIS key and derives every RNG from it --
# python random, numpy, torch CPU and CUDA, and cuDNN's algorithm choice -- in
# the server process and in every spawned client process. It also seeds the numpy
# stream PLATO itself uses for the Polaris arms' np.random.choice sampling
# (plato/config.py:456).
random_seed = {seed}

# REALITY -- what the clients actually sleep. T_tier(tier) * exp(eps_i), with
# eps_i fixed per client for the whole run. Identical in all five arms; what
# differs is what each arm BELIEVES about it.
[clients.simulation_distribution]
distribution = "custom"
csv_path = "{sleep_csv}"

[server]
address = "127.0.0.1"
port = {port}
synchronous = false
simulate_wall_time = true
periodic_interval = 1
ping_interval = 30
ping_timeout = 3600
# Seeds the `random` module the baseline arm selects with. Absent, its first
# selection draws from OS entropy and the arm is not reproducible
# (plato/servers/base.py:359).
random_seed = {seed}

# DERIVED FROM C: {staleness_bound_per_slot:g} bound-units per slot, the ratio that was inert at
# C = 4 with a flat 25. tau_hat_xavier here is {tau_hat_xavier:.4f} rounds, so the bound
# sits {bound_headroom:.2f}x above the routine staleness of the slowest tier.
staleness_bound = {staleness_bound}
# Stage 3 Step 0. PLATO adds the host's measured `processing_time` into every
# report's finish_time (plato/servers/base.py:1025-1030) and merges by popping
# the smallest one off a heap, so on a loaded machine the HOST decides which
# client merges first. Measured: two identical runs merged clients #7/#10/#20 in
# different orders when they finished within 0.592 s of each other
# (stage2_evidence/determinism_after.txt section 6).
#
# true makes gpu_tiers_dataset/runtime_patches.py zero that term at run time, so
# finish_time is training_time + comm_time + start_time -- exactly the quantity
# clock_reconstruction.py rebuilds. PLATO is not edited; the patch is installed
# from the example folder. The measured overhead is still logged.
exclude_processing_time_from_clock = {exclude_processing_time}
minimum_clients_aggregated = 1
mixing_hyperparameter = 0.5
adaptive_mixing = true
eval_interval = {eval_interval}
{polaris_block}
# KEEP every bare [server] key ABOVE this header. In TOML a bare key after a
# sub-table header belongs to that sub-table, so a key written below would be
# filed under server.staleness_weighting_function and the code defaults used
# instead (HANDOFF 4.2).
[server.staleness_weighting_function]
type = "polynomial"
a = 0.5

[data]
datasource = "Torchvision"
dataset_name = "CIFAR10"
download = true
# Absolute, because data_path is joined onto base_path (plato/config.py:360).
data_path = "{data_path}"
# N x partition_size holds CIFAR-10's 50,000 training images roughly constant
# across the fleet-size sweep, so N moves concurrency structure and not how much
# data the federation has in total.
partition_size = {partition_size}
sampler = "noniid"
# Dirichlet concentration. Held at {alpha:g} across this programme; the axes that
# move are C and N.
concentration = {alpha:g}
random_seed = {seed}

[trainer]
type = "basic"
rounds = {rounds}
# Must equal [clients] per_round.
max_concurrency = {per_round}
# 0.9 is unreachable for FedAsync + non-IID CIFAR-10/ResNet-18 here, i.e. NO
# early stop, which is the intent: every arm runs exactly `rounds` rounds and
# stays comparable. Read time-to-target off the curve afterwards (HANDOFF 4.13).
target_accuracy = 0.9
model_name = "resnet_18"
epochs = {epochs}
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
# BELIEF about hardware. Both files live in gpu_tiers_dataset/tiers/, which is
# the single source of truth for tier specs.
tiers_csv   = "{tiers_csv}"
clients_csv = "{clients_csv}"
# MUST match what the fleet was built with, or the predictor compares a duration
# against elapsed wall time in different units (HANDOFF 4.1).
anchor_tier  = "{anchor_tier}"
anchor_sleep = {anchor_sleep}
# T_round = E[T] / C, a closed queue of C slots. The historical "min" model is
# wrong and its error tracks whatever axis a sweep moves along (HANDOFF 4.11).
t_round_model = "queue"
{oracle_block}"""


def alpha_tag(alpha: float) -> str:
    """`0.8` -> `a0p8`. No dots -- they read as file extensions in run dirs."""
    return "a" + str(float(alpha)).replace(".", "p")


def run_name(arm: str, alpha: float, total_clients: int, per_round: int,
             seed: int, tag: str = "") -> str:
    return (f"{arm}_{alpha_tag(alpha)}_N{total_clients}_C{per_round}_s{seed}{tag}")


def cell_of(path: Path) -> str:
    """Everything about a run's identity except which arm it is.

    Arm names contain underscores (`gpu_oracle`), so the cell cannot be found by
    splitting on the first one. It is the part from `a<alpha>` onwards, which is
    the first field the run name is guaranteed to have.
    """
    stem = path.stem
    for i, part in enumerate(stem.split("_")):
        if part.startswith("a") and "p" in part:
            return "_".join(stem.split("_")[i:])
    raise SystemExit(f"cannot parse a cell out of the run name {stem!r}")


def arm_of(path: Path) -> str:
    cell = cell_of(path)
    return path.stem[: -len(cell) - 1]


def staleness_bound_for(per_round: int) -> int:
    """The bound, as a pure function of C. See STALENESS_BOUND_PER_SLOT."""
    return math.ceil(STALENESS_BOUND_PER_SLOT * per_round)


def entry_point() -> str:
    """One entry script for every arm; `--arm` selects the server class.

    The older folders use two scripts and pick between them by arm name. Seeding
    has to live at module level of whichever script actually runs, so two scripts
    means two places it can be missing from.
    """
    return "gpu_tiers_dataset/fedasync_seeded.py"


def port_for(arm: str, alpha: float, total_clients: int, per_round: int,
             seed: int, tag: str = "") -> int:
    for name, value, table in (
        ("alpha", alpha, ALPHA_SLOT),
        ("arm", arm, ARM_SLOT),
        ("concurrency", per_round, C_SLOT),
        ("clients", total_clients, N_SLOT),
        ("seed", seed, SEEDS),
    ):
        if value not in table:
            raise SystemExit(
                f"{name}={value!r} has no slot in the port plan. Add it to the "
                f"table in make_configs.py (allowed: {sorted(table)}) rather "
                f"than letting two runs land on one port."
            )
    if tag and tag not in TAG_SLOT:
        raise SystemExit(
            f"tag={tag!r} has no slot in the port plan (allowed: "
            f"{sorted(TAG_SLOT)}). Two tagged runs sharing a port hang silently, "
            f"and two tags is the normal case -- _det1 and _det2 are one config "
            f"run twice. Register the tag rather than letting them collide."
        )
    base = PROBE_PORT_BASE if tag else PORT_BASE
    leading = TAG_SLOT[tag] if tag else ALPHA_SLOT[alpha]
    port = (
        base
        + (1000 if tag else 2000) * leading
        + 200 * ARM_SLOT[arm]
        + 20 * C_SLOT[per_round]
        + 5 * N_SLOT[total_clients]
        + SEEDS[seed]
    )
    if port > 65535:
        raise SystemExit(f"port {port} is above the 16-bit maximum")
    return port


def fleet_dir(total_clients: int) -> Path:
    """The fleet directory for this N, and a clear error if it is not built."""
    counts = make_fleet.counts_for(total_clients, FLEET_MIX)
    name = build_fleet.fleet_dir_name(
        counts, ANCHOR_TIER, ANCHOR_SLEEP,
        PARTITION_SIZE[total_clients], JITTER_SIGMA, FLEET_SEED,
    )
    directory = FLEETS_DIR / name
    if not (directory / "manifest.json").exists():
        raise SystemExit(
            f"fleet {name} is not built. Run:\n"
            f"  python3 build_fleet.py --total {total_clients} "
            f"--partition-size {PARTITION_SIZE[total_clients]} "
            f"--jitter-sigma {JITTER_SIGMA} --seed {FLEET_SEED}"
        )
    return directory


def polaris_block(predictor: str | None) -> str:
    """The BELIEF block -- the one semantic difference between the arms."""
    if predictor is None:
        return (
            "\n# NOTE: no staleness_predictor and no polaris_* keys. This arm runs\n"
            "# fedasync_server.Server (random over the pool, skipping busy clients),\n"
            "# which reads none of them (HANDOFF 4.7).\n"
        )
    return f"""
# BELIEF -- THE ONE KEY THAT MAKES THIS ARM THIS ARM.
#   "gpu_tops"   tau_hat from the TIER MEAN in tiers_estimated_15x.csv. The
#                clients sleep T_tier * exp(eps_i), so this is now an ESTIMATE
#                with a real error -- the deployable method the thesis claims.
#   "cpu"        same code path, tiers_cpu.csv instead -- xavier and orin both
#                clock 2.00 GHz, so this belief cannot tell them apart.
#   "observed"   the client's own last measured staleness in ROUNDS, Polaris's
#                reactive estimator minus its +0.1 offset (ADR-0002). The frozen
#                denominator is the TREATMENT, not a bug (CODE_AUDIT 4.1).
#   "gpu_oracle" the per-client TRUE sleep, jitter included, via
#                [gpu_tier] oracle_sleep_csv below. NOT deployable: it is the
#                upper bound that says how much "gpu_tops" gives up.
staleness_predictor = "{predictor}"
polaris_alpha = 10.0
polaris_initial_gradient_bound = 0.5
polaris_initial_staleness = 0.01
"""


def oracle_block(holds_oracle: bool, sleep_csv: Path) -> str:
    """The gpu_oracle arm's extra key, and nothing at all for every other arm."""
    if not holds_oracle:
        return ""
    return f"""
# GROUND TRUTH, for the gpu_oracle arm ONLY. This is the SAME file as
# [clients.simulation_distribution] csv_path above -- the arm believes exactly
# what the clients do. fedasync_polaris_vi_server.check_oracle_wiring refuses to
# start if this key appears on any other arm, or is missing from this one, and
# preflight.py checks the two paths are the same file.
oracle_sleep_csv = "{sleep_csv}"
"""


def render(
    arm: str,
    alpha: float,
    outdir: Path,
    seed: int = 4126,
    total_clients: int = 20,
    per_round: int = CONCURRENCY,
    rounds: int = ROUNDS,
    tag: str = "",
    exclude_processing_time: bool = EXCLUDE_PROCESSING_TIME,
) -> Path:
    if total_clients not in PARTITION_SIZE:
        raise SystemExit(
            f"N={total_clients} has no partition_size in this programme "
            f"(have: {sorted(PARTITION_SIZE)})."
        )

    predictor, tiers_csv, holds_oracle = ARMS[arm]
    directory = fleet_dir(total_clients)
    manifest = json.loads((directory / "manifest.json").read_text())

    expected_t = manifest["E_T_s"]
    t_round = expected_t / per_round
    tau_hat_xavier = manifest["tier_stats"]["xavier"]["T_tier_s"] / t_round
    bound = staleness_bound_for(per_round)

    name = run_name(arm, alpha, total_clients, per_round, seed, tag)
    config_path = outdir / f"{name}.toml"
    sleep_csv = directory / "sleep_times.csv"

    text = TEMPLATE.format(
        arm=arm,
        alpha=alpha,
        seed=seed,
        entry=entry_point(),
        config_path=config_path.relative_to(FEDASYNC)
        if config_path.is_relative_to(FEDASYNC)
        else config_path,
        base_path=f"./runs/{RUNS_DIR}/{name}",
        fleet_name=directory.name,
        jitter_sigma=manifest["jitter_sigma"],
        fleet_seed=manifest["seed"],
        expected_t=expected_t,
        t_round=t_round,
        tau_hat_xavier=tau_hat_xavier,
        staleness_bound=bound,
        exclude_processing_time=str(bool(exclude_processing_time)).lower(),
        staleness_bound_per_slot=STALENESS_BOUND_PER_SLOT,
        bound_headroom=bound / tau_hat_xavier,
        total_clients=total_clients,
        per_round=per_round,
        port=port_for(arm, alpha, total_clients, per_round, seed, tag),
        polaris_block=polaris_block(predictor),
        oracle_block=oracle_block(holds_oracle, sleep_csv),
        sleep_csv=sleep_csv,
        clients_csv=directory / "clients.csv",
        tiers_csv=TIERS_DIR / tiers_csv,
        data_path=FEDASYNC / "data",
        partition_size=PARTITION_SIZE[total_clients],
        eval_interval=EVAL_INTERVAL,
        rounds=rounds,
        epochs=EPOCHS,
        anchor_tier=ANCHOR_TIER,
        anchor_sleep=ANCHOR_SLEEP,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text)
    return config_path


def flatten(config: dict) -> dict[tuple[str, str], object]:
    """{(section, key): value} for the top two levels, for comparison."""
    flat: dict[tuple[str, str], object] = {}
    for section, body in config.items():
        if isinstance(body, dict):
            for key, value in body.items():
                flat[(section, key)] = value
        else:
            flat[("", section)] = body
    return flat


def validate(paths: list[Path]) -> None:
    """Fail if two arms of one cell differ outside the permitted keys.

    Arms are compared WITHIN a cell only: alpha, N, C, seed and rounds are what
    the programme moves ACROSS cells, so a cross-cell diff is not an error.
    """
    loaded = {path: flatten(tomllib.loads(path.read_text())) for path in paths}

    cells: dict[str, list[Path]] = {}
    for path in paths:
        cells.setdefault(cell_of(path), []).append(path)

    for cell, members in sorted(cells.items()):
        reference = next((p for p in members if arm_of(p) == "gpu"), None)
        if reference is None:
            continue  # no gpu arm rendered for this cell; nothing to compare to
        for path in members:
            if path == reference:
                continue
            allowed = ARM_SPECIFIC
            if arm_of(path) == "baseline":
                allowed = allowed | BASELINE_MAY_LACK
            differing = {
                key
                for key in set(loaded[reference]) | set(loaded[path])
                if loaded[reference].get(key) != loaded[path].get(key)
            }
            unexpected = differing - allowed
            if unexpected:
                raise SystemExit(
                    f"{path.name} differs from {reference.name} in keys that "
                    f"must be shared inside cell {cell}: {sorted(unexpected)}"
                )

        baseline = next((p for p in members if arm_of(p) == "baseline"), None)
        if baseline:
            present = BASELINE_MAY_LACK & set(loaded[baseline])
            if present:
                raise SystemExit(
                    f"{baseline.name} carries keys its server never reads: "
                    f"{sorted(present)}"
                )

    # Exactly one arm may hold ground truth, and it must be the one named for it.
    for path, flat in loaded.items():
        has_key = ("gpu_tier", "oracle_sleep_csv") in flat
        is_oracle = arm_of(path) == "gpu_oracle"
        if has_key != is_oracle:
            raise SystemExit(
                f"{path.name}: oracle_sleep_csv is "
                f"{'present' if has_key else 'absent'} but the arm is "
                f"{arm_of(path)!r}. Only gpu_oracle may hold per-client truth; "
                f"either half alone silently produces a different arm."
            )
        if is_oracle:
            # An oracle pointed at another fleet is not an oracle, and nothing at
            # run time would say so: it would predict plausible times for the
            # wrong clients. `flatten` keeps sub-tables whole, so the simulation
            # distribution arrives here as a dict.
            simulation = flat.get(("clients", "simulation_distribution"), {})
            truth = simulation.get("csv_path") if isinstance(simulation, dict) else None
            believed = flat[("gpu_tier", "oracle_sleep_csv")]
            if truth != believed:
                raise SystemExit(
                    f"{path.name}: oracle_sleep_csv is {believed!r} but the "
                    f"clients sleep from {truth!r}. The oracle arm must believe "
                    f"exactly what the clients do."
                )

    ports: dict[int, Path] = {}
    for path, flat in loaded.items():
        port = flat[("server", "port")]
        if port in ports:
            raise SystemExit(
                f"{path.name} and {ports[port].name} both use port {port} -- "
                "two servers on one port hang silently."
            )
        ports[port] = path


def build_parser() -> argparse.ArgumentParser:
    """The CLI, factored out so tests can inspect parsed args without ever
    calling `parser.parse_args()` against the real `sys.argv`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=[0.8],
        help=f"Dirichlet concentration values. Registered: {sorted(ALPHA_SLOT)}.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[4126],
        help=f"Seeds to render. Registered: {sorted(SEEDS)} (seed 1 is 4126).",
    )
    parser.add_argument(
        "--clients", nargs="+", type=int, default=[20],
        help=f"Fleet sizes N. Registered: {sorted(N_SLOT)}.",
    )
    parser.add_argument(
        "--concurrency", nargs="+", type=int, default=[CONCURRENCY],
        help=f"Slots in flight C; sets both per_round and max_concurrency, and "
             f"staleness_bound follows it. Registered: {sorted(C_SLOT)}.",
    )
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=sorted(ARMS))
    parser.add_argument("--outdir", type=Path, default=HERE / "configs")
    parser.add_argument(
        "--tag", default="",
        help="Suffix for a short test build (e.g. _det1). Gets its own config "
             "names, run directories and port block, so it can never collide "
             "with a real run.",
    )
    parser.add_argument(
        "--exclude-processing-time", dest="exclude_processing_time",
        action="store_true", default=EXCLUDE_PROCESSING_TIME,
        help="Stage 3 Step 0: keep the host's measured processing_time out of "
             "PLATO's finish_time, via runtime_patches.py. "
             f"Default: {EXCLUDE_PROCESSING_TIME}.",
    )
    parser.add_argument(
        "--no-exclude-processing-time", dest="exclude_processing_time",
        action="store_false",
        help="Render configs WITHOUT the Step 0 patch, whatever the default is "
             "-- e.g. to reproduce a pre-Step-0 config, or if Step 0's own "
             "verification run fails and the sweep must proceed unpatched.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    paths = [
        render(arm, alpha, args.outdir, seed, total_clients, per_round,
               args.rounds, args.tag, args.exclude_processing_time)
        for alpha in args.alphas
        for total_clients in args.clients
        for per_round in args.concurrency
        for seed in args.seeds
        for arm in args.arms
    ]
    validate(paths)

    print(
        f"Wrote {len(paths)} configs (alphas {args.alphas}, N {args.clients}, "
        f"C {args.concurrency}, seeds {args.seeds}, rounds={args.rounds}, "
        f"tag={args.tag or 'none'}, "
        f"exclude_processing_time={args.exclude_processing_time}):"
    )
    for path in sorted(paths):
        config = tomllib.loads(path.read_text())
        print(f"  {path.name:46s} port {config['server']['port']}  "
              f"staleness_bound {config['server']['staleness_bound']}")
    print(f"\nNext: python3 {HERE.name}/preflight.py"
          + (f" --tag {args.tag}" if args.tag else ""))


if __name__ == "__main__":
    main()
