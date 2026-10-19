"""
Entry point for the SUMO closed-loop FedAsync experiment, with every RNG seeded.

Stage 6 (prompts/STAGE6_seeds.md) is the first thing in this folder that varies a
"run seed" (lottery draws / torch / dispatch order) while holding a fleet fixed,
and asks whether the result is stable across it. That question is meaningless
unless a run seed actually controls all of that -- and, checked before writing
anything else in this stage, **`sumo_fedasync.py` did not**: it sets
`[data] random_seed` (the data sampler only) and nothing else. Torch's global RNG
(model init, `RandomHorizontalFlip` + `RandomCrop` augmentation) has been
completely unseeded in every Task 4 / Stage 5 / Stage 5b run. `../gpu_tiers_dataset
/seeding.py`'s own docstring already measured what that alone is worth: two runs
of one otherwise-identical config differed by 6.29% of simulated clock, from torch
alone. For Stage 5/5b that noise sat safely inside a much larger filter effect;
for Stage 6, whose entire point is to measure the SIZE of run-to-run noise, it
would be exactly the thing being measured, uncontrolled.

This file is `sumo_fedasync.py` plus the same two fixes
`gpu_tiers_dataset/fedasync_seeded.py` already made for the (non-SUMO) FedAsync
arms, applied here for the same reasons:

  1. `seeding.seed_everything()` at MODULE level (not in main()) -- PLATO spawns
     client processes which re-import this module before unpickling their target,
     so module-level code is what reaches every spawned process; see seeding.py.
  2. `client_seeded_trainer.ClientSeededTrainer` instead of the default trainer --
     re-seeds from (base_seed, client_id, round) at the top of every local
     training run, so a client's randomness follows the CLIENT rather than
     whichever worker process happened to be free (stage1_evidence/
     determinism.txt section 4; the same fix fedasync_seeded.py uses).

A NEW file rather than editing sumo_fedasync.py, for the reason
fedasync_seeded.py's own docstring gives for the same choice: Task 4, Stage 5 and
Stage 5b hold finished, accepted results produced by the unseeded entry point.
Changing what it does would make those results not reproducible by re-running them
with nothing in a log to say why. sumo_fedasync.py stays exactly as it is; only
configs generated for Stage 6 point at this file instead.

The one thing this does NOT change: the fleet. `[clients.simulation_distribution]
csv_path` still points at the fixed `sleep_times.csv` for fleet
`n20_x10-o7-t3_xa15_p2500_j0p149_s4126`, read as a plain file, never regenerated --
seeding numpy/torch has no effect on it. Verify this with an MD5 across every
Stage 6 config before trusting a comparison, per the prompt: "The per-client
sleep_i values must be byte-identical across every run in this stage."

Required config section: identical to sumo_fedasync.py's (see its docstring),
plus `[clients] random_seed` / `[server] random_seed` / `[data] random_seed` all
set to the SAME value -- the one "run seed" this file's seeding is keyed to
(seeding.py reads `[clients] random_seed`; client_seeded_trainer.py does too, via
Config(), for the same value).

Run:
    cd examples/async/fedasync
    PYTHONPATH=<repo root>:gpu_tiers:gpu_tiers_dataset:.:sumo_close_loop \
        python3 sumo_close_loop/sumo_fedasync_seeded.py \
        -c sumo_close_loop/configs/stage6/<name>.toml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# --- path setup, before anything from this project is imported ---------------
_HERE = Path(__file__).resolve().parent
_FEDASYNC = _HERE.parent
for _path in (
    str(_FEDASYNC),
    str(_FEDASYNC / "gpu_tiers"),
    str(_FEDASYNC / "gpu_tiers_dataset"),
    str(_HERE),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import seeding  # noqa: E402

# =============================================================================
# Module level, so this runs in every spawned client process as well as in the
# server -- see seeding.py and fedasync_seeded.py's own module-level block for
# why this has to happen before torch (or anything that imports it) is touched.
# =============================================================================
SEED_RECORD = seeding.seed_everything()

import client_seeded_trainer  # noqa: E402
import fedasync_algorithm  # noqa: E402
import sumo_server  # noqa: E402
from fedasync_gpu_tier import GpuTierLogCallback  # noqa: E402
from stage0_filter import resolve_stage0_config  # noqa: E402
from sumo_mobility import CoverageOracle  # noqa: E402
from sumo_position import PositionOracle  # noqa: E402

from plato.clients import simple  # noqa: E402
from plato.config import Config  # noqa: E402


def main() -> None:
    seeding.log_seeding()

    sumo_config = getattr(Config().config, "sumo", None)
    if sumo_config is None or not hasattr(sumo_config, "coverage_csv"):
        raise ValueError(
            "This experiment needs a [sumo] section with coverage_csv pointing "
            "at the table written by remap_fleet.py."
        )

    oracle = CoverageOracle.from_csv(sumo_config.coverage_csv)
    upload_policy = sumo_server.resolve_upload_policy(
        getattr(sumo_config, "upload_policy", None)
    )
    stage0 = resolve_stage0_config(sumo_config)

    position_oracle = None
    if stage0.enabled and stage0.r_i_mode == "vmax_distance":
        positions_csv = getattr(sumo_config, "positions_csv", None)
        if positions_csv is None:
            raise ValueError(
                "[SumoRsu] stage0_r_i = 'vmax_distance' needs [sumo] "
                "positions_csv pointing at the table written by "
                "remap_positions.py."
            )
        position_oracle = PositionOracle.from_csv(positions_csv)

    algorithm = fedasync_algorithm.Algorithm
    client = simple.Client(
        algorithm=algorithm,
        trainer=client_seeded_trainer.ClientSeededTrainer,
        trainer_callbacks=[GpuTierLogCallback],
    )

    if hasattr(Config().server, "staleness_predictor"):
        server_class = sumo_server.SumoPolarisServer
    else:
        server_class = sumo_server.SumoBaselineServer
    logging.info(
        "[SumoRsu] Server: %s  upload_policy=%s  stage0_t_hat=%s  stage0_r_i=%s  "
        "run_seed=%s",
        server_class.__name__, upload_policy, stage0.t_hat_source, stage0.r_i_mode,
        SEED_RECORD.get("base_seed"),
    )

    server = server_class(
        oracle, upload_policy=upload_policy, stage0=stage0,
        position_oracle=position_oracle, algorithm=algorithm,
    )
    server.run(client)


if __name__ == "__main__":
    main()
