"""
Make training randomness a property of the CLIENT, not of the worker process.

The problem this fixes (stage1_evidence/determinism.txt section 4)
------------------------------------------------------------------
PLATO runs a pool of `max_concurrency` worker processes and hands each round's
selected client to whichever worker happens to be free
(`plato/servers/base.py:662-682`). Which worker that is depends on the order the
client processes connected over socket.io, which is a real-time race at startup
that nothing seeds. Two runs of one config showed it directly:

    worker   run A            run B
    2        client 14        client 15
    4        client 15        client 14

`seeding.py` seeds torch once per PROCESS, so a client served by worker 2 in one
run and worker 4 in the other draws a different augmentation sequence
(`RandomHorizontalFlip` + `RandomCrop`, `plato/datasources/torchvision.py:141`),
computes different gradients, gets a different `G_i`, and the selector then ranks
`V_i = G_i / sqrt(1 + tau_hat_i)` differently. `train_loss` diverged at round 3
and the dispatch sequence at round 18.

The fix: re-seed at the start of every local training run from
`(base_seed, client_id, current_round)`.

This is not a change to the model being simulated. A worker process is an
implementation detail of the simulator; a client is not. Before this, the
parallelism strategy leaked into the experiment -- the same federation run with
`max_concurrency = 2` instead of 4 would have drawn different augmentations for
the same clients in the same rounds. Tying the stream to `client_id` puts the
randomness where the model already says it belongs.

What it does NOT do
-------------------
It does not make `sim_time_s` reproducible: PLATO adds the host's
`processing_time` to every report's finish time
(`plato/servers/base.py:1025-1030`). That is what `clock_reconstruction.py`
addresses, by recomputing the clock from the dispatch order instead.

Relationship to `seeding.py`
----------------------------
`seeding.py` still runs, and is still needed: it seeds each process at import
time, which covers model construction and anything outside a training run, and it
is what makes the numpy stream the selector samples from reproducible. This
subclass adds a second, finer re-seed on top, scoped to one client's one round.
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Any

import numpy as np
import torch

from plato.config import Config
from plato.trainers import basic

# Kept below 2**31 so it is a valid seed for every RNG here: numpy's legacy
# `np.random.seed` rejects anything larger.
_SEED_MODULUS = 2**31 - 1


def derive_seed(base_seed: int, client_id: int, current_round: int) -> int:
    """A stable, well-spread seed for one client's one round.

    blake2b rather than Python's `hash()`: `hash()` of a str is randomised per
    process by PYTHONHASHSEED, so it would reintroduce exactly the cross-process
    variation this module exists to remove. blake2b is stable across processes,
    machines and interpreter versions.

    Multiplying or adding the three numbers would work too, but would make
    neighbouring (client, round) pairs produce neighbouring seeds -- adjacent
    Mersenne Twister states are correlated in their first draws, and every round
    here consumes only a few hundred of them.
    """
    if current_round is None:
        current_round = 0
    key = f"{int(base_seed)}:{int(client_id)}:{int(current_round)}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SEED_MODULUS


def base_seed() -> int:
    """`[clients] random_seed`, the one seed everything is derived from.

    Read through `Config()` rather than from the TOML directly, unlike
    `seeding.py`: by the time a trainer exists the Config singleton is long since
    built, so there is no init-ordering hazard to avoid here.

    Absent is fatal, for the same reason as in `seeding.py`: a run that quietly
    fell back to a constant would look seeded in the log and not be reproducible
    per configuration.
    """
    seed = getattr(Config().clients, "random_seed", None)
    if seed is None:
        raise ValueError(
            "[ClientSeededTrainer] [clients] random_seed is missing. Every run in "
            "this folder must carry it: it is the single seed every RNG is "
            "derived from."
        )
    return int(seed)


def seed_for_training_run(client_id: int, current_round: int) -> int:
    """Seed python `random`, numpy and torch for one client's one round."""
    seed = derive_seed(base_seed(), client_id, current_round)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


class ClientSeededTrainer(basic.Trainer):
    """`basic.Trainer`, re-seeded per (client, round) before each training run.

    `train_model` is the hook rather than the `on_train_run_start` callback,
    because the DataLoader -- and therefore every augmentation draw -- is created
    inside `train_model` AFTER that callback fires
    (`plato/trainers/composable.py:412-470`). Seeding at the top of `train_model`
    is upstream of the loader, the sampler, the augmentation and any dropout.
    """

    def train_model(self, config: dict, trainset: Any, sampler: Any, **kwargs):
        seed = seed_for_training_run(self.client_id, self.current_round)
        logging.info(
            "[ClientSeededTrainer] client_id=%s round=%s -> seed %d "
            "(derived from [clients] random_seed and NOT from the worker process)",
            self.client_id, self.current_round, seed,
        )
        return super().train_model(config, trainset, sampler, **kwargs)
