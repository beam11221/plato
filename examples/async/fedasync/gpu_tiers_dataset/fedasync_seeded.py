"""
The entry points for this folder's runs: PLATO's FedAsync arms, with every RNG seeded.

One file, two entry points, selected by `--arm`:

    --arm baseline                    fedasync_server.Server
    --arm gpu|cpu|observed|gpu_oracle fedasync_polaris_vi_server.Server

`../separability/make_configs.py` picks between two SEPARATE scripts for this
(`fedasync_gpu_tier.py` and `fedasync_gpu_tier_polaris_vi.py`, which differ only
in the server class and in one stray `import logging`). Keeping one file here
means the seeding, which must run at module level, cannot be present in one entry
point and missing from the other.

WHY NEW ENTRY POINTS RATHER THAN EDITING THE OLD ONES
-----------------------------------------------------
`../fedasync_gpu_tier.py` and `../fedasync_gpu_tier_polaris_vi.py` are shared with
`../surgical/`, `../separability/`, `../separability_n100/` and
`../sumo_close_loop/`, all of which hold finished results that are being kept as
the oracle branch for reference. Seeding torch changes what a run does. Putting
that change in a file those folders execute would mean their results could no
longer be reproduced by re-running them, and nothing in a log would say why.

So this file exists alongside them, imports their pieces rather than copying
them, and only the configs generated in THIS folder point at it.

WHY THE SEEDING CALL IS AT MODULE LEVEL
---------------------------------------
PLATO spawns client processes with the `spawn` start method, and a spawned child
re-imports this module as `__mp_main__` before unpickling its target. Module-level
code therefore runs in the server process and in every client process; code inside
`main()` runs only in the server. See `seeding.py` for the verification.

Run (run.sh does this for you):
    cd examples/async/fedasync
    PYTHONPATH=<repo root>:gpu_tiers:. python3 gpu_tiers_dataset/fedasync_seeded.py \
        --arm gpu -c gpu_tiers_dataset/configs/gpu_a0p8_N20_C4_s4126.toml

The repo root must be on PYTHONPATH: PLATO is imported from the source tree, not
installed (`../separability/run.sh:153` does the same).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# --- path setup, before anything from this project is imported ---------------
# `gpu_tier_sleep` and the fedasync_* modules are top-level modules living in
# `../` and `../gpu_tiers/`; the run command puts them on PYTHONPATH, and these
# lines make the script work when it does not.
_HERE = Path(__file__).resolve().parent
_FEDASYNC = _HERE.parent
for _path in (str(_FEDASYNC), str(_FEDASYNC / "gpu_tiers"), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import seeding  # noqa: E402

# Arms that run the Polaris V_i selector; `baseline` runs plain FedAsync.
POLARIS_ARMS = ("gpu", "cpu", "observed", "gpu_oracle")
ARMS = ("baseline",) + POLARIS_ARMS


def strip_arm(argv: list[str]) -> tuple[str, list[str]]:
    """Return (arm, argv WITHOUT the --arm flag). Does not mutate `argv`.

    Parsed by hand, and stripped before any PLATO import, for a reason that is
    easy to trip over: `plato/servers/base.py:382` declares

        def start(self, port=Config().server.port):

    and a default argument is evaluated at class-definition time, i.e. when the
    module is IMPORTED. So `Config()` -- and with it `parser.parse_args()` over
    the whole of `sys.argv` -- runs during `import fedasync_server`, long before
    `main()`. An unknown `--arm` still in argv at that moment is an argparse
    "unrecognized arguments" exit with a message that points at PLATO rather than
    at this script.

    It returns a cleaned COPY rather than editing sys.argv in place because the
    original has to come back afterwards; see the module-level block below.

    An absent or unrecognised value RAISES. It selects the server class, so a
    silent fallback would produce a run that is behaviourally another arm of the
    same experiment with nothing in the results to distinguish the duplicate --
    the same trap `fedasync_polaris_vi_server.resolve_staleness_predictor`
    documents.
    """
    cleaned = list(argv)
    arm = None
    for i, token in enumerate(cleaned):
        if token == "--arm" and i + 1 < len(cleaned):
            arm = cleaned[i + 1]
            del cleaned[i : i + 2]
            break
        if token.startswith("--arm="):
            arm = token.split("=", 1)[1]
            del cleaned[i]
            break

    if arm is None:
        raise SystemExit(
            f"[fedasync_seeded] --arm is required; expected one of {list(ARMS)}."
        )
    if arm not in ARMS:
        raise SystemExit(
            f"[fedasync_seeded] --arm {arm!r} is not recognised; expected one of "
            f"{list(ARMS)}."
        )
    return arm, cleaned


# =============================================================================
# Module level, so all of this runs in every spawned client process as well as in
# the server. ORDER MATTERS, in both directions.
#
# Seeding first, so that everything afterwards -- including torch importing and
# initialising its own RNG state -- happens with the RNGs already pinned.
#
# Then `--arm` has to LEAVE sys.argv before the first PLATO import, because that
# import runs `Config()` and therefore `parse_args()` over the whole of argv (see
# strip_arm). And then it has to be PUT BACK.
#
# The restore is not tidiness. PLATO starts each client with
# `mp.Process(target=plato.client.run, ...)` under spawn, and spawn snapshots
# `sys.argv` at `proc.start()` -- which happens inside `server.run(client)`, long
# after this module was imported. A `--arm` deleted here and never restored is
# absent from that snapshot, so every spawned child re-imports this module, finds
# no `--arm`, and exits. The server then sits waiting for clients that will never
# connect: no traceback, no failed run, just a hang. Seen and fixed here; the
# spawn test in tests/test_fedasync_seeded.py is what keeps it fixed.
# =============================================================================
SEED_RECORD = seeding.seed_everything()

ARM, _ARGV_WITHOUT_ARM = strip_arm(sys.argv)
_ARGV_WITH_ARM = list(sys.argv)
sys.argv = _ARGV_WITHOUT_ARM

import client_seeded_trainer  # noqa: E402
import fedasync_algorithm  # noqa: E402
import fedasync_polaris_vi_server  # noqa: E402
import fedasync_server  # noqa: E402
import gpu_tier_sleep  # noqa: E402

from plato.callbacks.trainer import TrainerCallback, resolve_num_samples  # noqa: E402
from plato.clients import simple  # noqa: E402
from plato.config import Config  # noqa: E402

import runtime_patches  # noqa: E402  (imports plato.servers, so it goes here)

# Config is now built and cached; nothing downstream parses argv again, so the
# original can come back for the spawned children to inherit.
sys.argv = _ARGV_WITH_ARM


class GpuTierLogCallback(TrainerCallback):
    """Logs tier, dataset size and scheduled sleep after each epoch.

    Identical in behaviour to the callback in `../fedasync_gpu_tier.py`, with one
    addition: the first call in a process emits the seeding record. Seeding
    happens at import time, before `Config()` installs the run's FileHandler, so
    this is the first moment a log line from a client process actually reaches
    `client_<id>.log` -- and that line is the evidence that the spawned process
    really was seeded.
    """

    def on_train_epoch_end(self, trainer, config, **kwargs):
        seeding.log_seeding()

        client_id = getattr(trainer, "client_id", None)
        if client_id is None or client_id == 0:
            return

        sleep_times = Config.client_sleep_times
        if sleep_times is None or client_id - 1 >= len(sleep_times):
            return

        scheduled_sleep = float(sleep_times[client_id - 1])
        dataset_size = resolve_num_samples(trainer) or 0

        try:
            tier_id = gpu_tier_sleep.tier_for(client_id).tier_id
        except KeyError:
            tier_id = "unknown"

        logging.info(
            "[GpuTierSleep] round=%d client_id=%d tier_id=%s "
            "dataset_size=%d sleep_time=%.6f s",
            getattr(trainer, "current_round", -1),
            client_id,
            tier_id,
            dataset_size,
            scheduled_sleep,
        )


def build_server(arm: str):
    import os  # noqa: PLC0415

    algorithm = fedasync_algorithm.Algorithm
    if arm == "baseline":
        server = fedasync_server.Server(algorithm=algorithm)
    else:
        server = fedasync_polaris_vi_server.Server(algorithm=algorithm)

    # Stage 4 Task 2 instrumentation (prompts/STAGE4_verify_scoring.md), opt-in
    # only: unset by default, so every other run through this entry point is
    # byte-identical to before this was added. See vi_term_logger.py — it
    # wraps the already-built strategy and does not alter its decisions.
    vi_log_path = os.environ.get("STAGE4_VI_LOG_PATH")
    if vi_log_path:
        import vi_term_logger  # noqa: PLC0415

        server.client_selection_strategy = vi_term_logger.ViTermLogger(
            server.client_selection_strategy, vi_log_path
        )
        logging.info("[fedasync_seeded] Stage4 Task2: logging every scored "
                     "candidate to %s", vi_log_path)

    return algorithm, server


def save_config_used() -> None:
    """Copy this run's TOML into its own output directory.

    `configs/` is regenerated whenever a generator constant moves, so the file a
    run was launched with may not be the file sitting at that path a week later.
    Anything analysing the run afterwards -- `clock_reconstruction.py` needs C,
    the sleep CSV and comm_time -- must read the copy, not the live one.
    """
    import shutil                                     # noqa: PLC0415

    source = Path(seeding.config_path_from_argv())
    destination = Path(Config().params["base_path"]) / "config_used.toml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    logging.info("[fedasync_seeded] saved the launch config to %s", destination)


def main() -> None:
    logging.info("[fedasync_seeded] arm=%s", ARM)
    save_config_used()
    # Must run before the Server is constructed: it replaces a method on the
    # class, and Server.__init__ is what binds the instance that will use it.
    runtime_patches.apply_if_requested()
    seeding.log_seeding()

    algorithm, server = build_server(ARM)
    # ClientSeededTrainer re-seeds from (base_seed, client_id, round) at the top
    # of every local training run, so a client's randomness follows the CLIENT
    # rather than whichever worker process happened to pick it up. See
    # client_seeded_trainer.py and stage1_evidence/determinism.txt section 4.
    client = simple.Client(
        algorithm=algorithm,
        trainer=client_seeded_trainer.ClientSeededTrainer,
        trainer_callbacks=[GpuTierLogCallback],
    )
    try:
        server.run(client)
    finally:
        runtime_patches.log_summary()


if __name__ == "__main__":
    main()
