"""
Entry point for the Stage 7 open-population cell.

Which arm runs is decided by the TOML, exactly as in `sumo_close_loop`:

    [server] staleness_predictor absent                  -> baseline
    [server] staleness_predictor = gpu_tops | cpu | observed -> the Polaris arm

Required `[sumo_open]` section:

    coverage_csv      = "/abs/path/to/sumo_trace/.../coverage_windows.csv"
    p_recur           = 0.0   # 0 = identity is the trace vehicle
                              # 1 = identity is bound to the slot (Stage 7b)
    slots             = 20    # M, fixed for this stage
    fleet_clients_csv = ...   # p_recur = 1 only: slot k's tier (Stage 3-6 fleet)
    fleet_sleep_csv   = ...   # p_recur = 1 only: slot k's sleep (Stage 3-6 fleet)
    physics_seed      = 4126  # optional; see "Two seeds" below

Two seeds (Stage 8)
-------------------
`[clients] random_seed` is the RUN seed: process seeding (seeding.py), the
per-(identity, round) training seed, and the server's dispatch randomness.
`[sumo_open] physics_seed` is the PHYSICS seed: each trace vehicle's sleep
(`vehicle_sleep_s`) and the slot allocator's RNG.  Until Stage 8 both came
from `[clients] random_seed`, so changing the run seed would have silently
redrawn every vehicle's sleep -- a different population, not a replication.
Absent, `physics_seed` falls back to the run seed, so every Stage 7 / 7b
config (no key, run seed 4126) behaves exactly as it did.

Seeding is at MODULE level, before torch is imported, for the reason
`gpu_tiers_dataset/seeding.py` documents: PLATO spawns client processes which
re-import this module, so module-level code is what reaches every process.
Stage 6 established that without it torch is free-running and two otherwise
identical runs differ by ~6% of simulated clock.

Run:
    cd examples/async/fedasync
    PYTHONPATH=<repo>:gpu_tiers:gpu_tiers_dataset:.:sumo_close_loop:sumo_open_loop \\
        python3 sumo_open_loop/sumo_open_fedasync.py -c <config>.toml
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FEDASYNC = _HERE.parent
for _path in (
    str(_FEDASYNC),
    str(_FEDASYNC / "gpu_tiers"),
    str(_FEDASYNC / "gpu_tiers_dataset"),
    str(_FEDASYNC / "sumo_close_loop"),
    str(_HERE),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import seeding  # noqa: E402

SEED_RECORD = seeding.seed_everything()

import client_seeded_trainer  # noqa: E402
import fedasync_algorithm  # noqa: E402
import numpy as np  # noqa: E402
import sumo_open_server  # noqa: E402
import torch  # noqa: E402
from fedasync_gpu_tier import GpuTierLogCallback  # noqa: E402
from open_stage0 import load_trace_positions, resolve_open_stage0  # noqa: E402
from occupant_identity import (  # noqa: E402
    IdentityResolver,
    identity_seed,
    load_fleet,
    partition_fingerprint,
)
from sumo_open_mobility import OpenCoverageTimeline  # noqa: E402

from plato.clients import simple  # noqa: E402
from plato.clients.strategies.defaults import (  # noqa: E402
    DefaultCommunicationStrategy,
    DefaultLifecycleStrategy,
    DefaultPayloadStrategy,
    DefaultReportingStrategy,
    DefaultTrainingStrategy,
)
from plato.config import Config  # noqa: E402
from plato.samplers import registry as samplers_registry  # noqa: E402


class OccupantLifecycle(DefaultLifecycleStrategy):
    """Train on the OCCUPANT IDENTITY's partition, seeded by its identity.

    Stage 7 swapped `client_id` for the partition around `load_data`.  That
    had no effect: `load_data` only loads the whole datasource (once per
    worker), and the sampler -- the thing that picks the partition -- is built
    later, in `configure` (`plato/clients/strategies/defaults.py:121`), after
    the swap had been undone.  Every Stage 7 dispatch trained on the SLOT's
    partition (STAGE7B_step1_siblings.md, section 1).

    So the sampler is rebuilt here, after `configure`, from the identity's
    partition.  `client_id` itself is never touched: the trainer and algorithm
    are bound to it in `configure` and name their per-client model files by
    it, and two vehicles with the same partition can be in flight at once.
    """

    def process_server_response(self, context, server_response) -> None:
        super().process_server_response(context, server_response)
        context.stage7_veh_id = server_response.get("stage7_veh_id")
        context.stage7_identity = server_response.get("stage7_identity")
        context.stage7_tier = server_response.get("stage7_tier")
        context.stage7_partition = server_response.get("stage7_partition")

    def configure(self, context) -> None:
        super().configure(context)
        partition = getattr(context, "stage7_partition", None)
        identity = getattr(context, "stage7_identity", None)
        if partition is None or identity is None:
            # `client.configure()` also runs once at process start, before any
            # server response.  No sampler is used until a dispatch, and a
            # dispatch without an identity cannot train silently on the slot's
            # data: OccupantSeededTrainer refuses to run without one, and the
            # pre-flight requires this method's log line on every dispatch.
            return
        if context.trainer is not None:
            context.trainer.stage7_identity = identity
        if context.datasource is None:
            return

        sampler_id = int(partition) + 1
        context.sampler = samplers_registry.get(context.datasource, sampler_id)
        logging.info(
            "[Stage7] client slot=%s veh_id=%s identity_id=%s tier=%s "
            "partition=%d sampler_id=%d fingerprint=%s",
            context.client_id, getattr(context, "stage7_veh_id", None), identity,
            getattr(context, "stage7_tier", None), int(partition), sampler_id,
            partition_fingerprint(context.sampler, context.datasource),
        )


class OccupantSeededTrainer(client_seeded_trainer.ClientSeededTrainer):
    """Re-seed every training run from (base_seed, IDENTITY, round).

    `ClientSeededTrainer` keys on `client_id`, which here is the slot.  The key
    format is shared (`identity_seed`), so at p_recur = 1 -- identity == slot
    number -- the seed is exactly the closed-loop cell's.
    """

    def train_model(self, config: dict, trainset, sampler, **kwargs):
        identity = getattr(self, "stage7_identity", None)
        if identity is None:
            raise RuntimeError("[Stage7b] trainer has no occupant identity to seed from.")
        seed = identity_seed(client_seeded_trainer.base_seed(), identity, self.current_round)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logging.info(
            "[ClientSeededTrainer] client_id=%s identity_id=%s round=%s -> seed %d "
            "(derived from [clients] random_seed and the occupant identity)",
            self.client_id, identity, self.current_round, seed,
        )
        # Skip ClientSeededTrainer's own slot-keyed reseed.
        return super(client_seeded_trainer.ClientSeededTrainer, self).train_model(
            config, trainset, sampler, **kwargs
        )


class OpenPopulationClient(simple.Client):
    """`simple.Client` with the occupant-keyed lifecycle strategy installed.

    `_configure_composable` (plato/clients/base.py:104) is the supported way to
    swap a strategy: it tears the previous stack down and rebuilds the
    composable runtime.  Replacing `self.lifecycle_strategy` directly would
    work for the call sites but would skip the new strategy's `setup()`, which
    `ComposableClient.__init__` only ever runs over the stack it was given.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._configure_composable(
            lifecycle_strategy=OccupantLifecycle(),
            payload_strategy=DefaultPayloadStrategy(),
            training_strategy=DefaultTrainingStrategy(),
            reporting_strategy=DefaultReportingStrategy(),
            communication_strategy=DefaultCommunicationStrategy(),
        )


def main() -> None:
    seeding.log_seeding()

    open_cfg = getattr(Config().config, "sumo_open", None)
    if open_cfg is None or not hasattr(open_cfg, "coverage_csv"):
        raise ValueError(
            "Stage 7 needs a [sumo_open] section with coverage_csv pointing at "
            "the open trace's coverage_windows.csv."
        )

    p_recur = float(getattr(open_cfg, "p_recur"))
    slots = int(getattr(open_cfg, "slots", 20))
    base_seed = int(Config().clients.random_seed)          # RUN seed
    physics_seed = int(getattr(open_cfg, "physics_seed", base_seed))

    if slots != Config().clients.total_clients:
        raise ValueError(
            f"[Stage7] slots ({slots}) must equal [clients] total_clients "
            f"({Config().clients.total_clients}); M is the client-id space."
        )

    timeline = OpenCoverageTimeline.from_csv(
        open_cfg.coverage_csv, slots=slots, p_recur=p_recur, base_seed=physics_seed
    )
    fleet = None
    p_return = getattr(open_cfg, "p_return", None)
    p_return = None if p_return is None else float(p_return)
    if p_recur == 1.0 or p_return is not None:
        fleet = load_fleet(open_cfg.fleet_clients_csv, open_cfg.fleet_sleep_csv)
    resolver = IdentityResolver(timeline, physics_seed, p_recur, slots, fleet, p_return=p_return)
    logging.info(
        "[Stage8] seeds: run_seed=%d ([clients] random_seed: training, process, "
        "dispatch) physics_seed=%d ([sumo_open]: vehicle sleeps, allocator)",
        base_seed, physics_seed,
    )
    logging.info(
        "[Stage7b] occupant identity: %s",
        "slot-bound Stage 3-6 fleet (trace supplies coverage timing only)"
        if p_recur == 1.0 else "the trace vehicle",
    )

    algorithm = fedasync_algorithm.Algorithm
    client = OpenPopulationClient(
        algorithm=algorithm,
        trainer=OccupantSeededTrainer,
        trainer_callbacks=[GpuTierLogCallback],
    )

    if hasattr(Config().server, "staleness_predictor"):
        server_class = sumo_open_server.OpenPolarisServer
    else:
        server_class = sumo_open_server.OpenBaselineServer

    logging.info(
        "[Stage7] Server: %s  p_recur=%g  M=%d  seed=%s",
        server_class.__name__, p_recur, slots, SEED_RECORD.get("base_seed"),
    )

    # Stage 10: the open-population Stage 0 gate (absent unless stage0_t_hat set).
    stage0 = resolve_open_stage0(open_cfg)
    positions = load_trace_positions(stage0.positions_csv) if stage0.enabled else None

    server = server_class(
        timeline, base_seed=physics_seed, p_recur=p_recur, resolver=resolver,
        stage0=stage0, positions=positions, algorithm=algorithm,
    )
    server.run(client)


if __name__ == "__main__":
    main()
