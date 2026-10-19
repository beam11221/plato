"""
Client side of the Wen environment: train on this client's disjoint partition.

PLATO's `configure` builds the sampler from `[data] sampler`
(plato/clients/strategies/defaults.py:121). This lifecycle rebuilds it
afterwards from `wen_partition.build(...)`, the same supported override point
`sumo_open_loop/sumo_open_fedasync.OccupantLifecycle` uses. The partition is a
pure function of the CIFAR-10 targets and `[data] random_seed`, so every
client process derives the identical split and the fingerprint logged here can
be compared across processes and runs.
"""

from __future__ import annotations

import logging

import torch
import wen_partition
from torch.utils.data import SubsetRandomSampler

from plato.clients import simple
from plato.clients.strategies.defaults import (
    DefaultCommunicationStrategy,
    DefaultLifecycleStrategy,
    DefaultPayloadStrategy,
    DefaultReportingStrategy,
    DefaultTrainingStrategy,
)
from plato.config import Config
from plato.samplers import base

_PARTITION: wen_partition.Partition | None = None


def partition_for(datasource) -> wen_partition.Partition:
    global _PARTITION
    if _PARTITION is None:
        _PARTITION = wen_partition.build(datasource.targets(),
                                         int(Config().data.random_seed))
    return _PARTITION


class WenPartitionSampler(base.Sampler):
    def __init__(self, indices) -> None:
        super().__init__()
        self.indices = [int(i) for i in indices]

    def get(self):
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)
        return SubsetRandomSampler(self.indices, generator=gen)

    def num_samples(self) -> int:
        return len(self.indices)


class WenLifecycle(DefaultLifecycleStrategy):
    def configure(self, context) -> None:
        super().configure(context)
        if context.datasource is None or not context.client_id:
            return
        partition = partition_for(context.datasource)
        context.sampler = WenPartitionSampler(partition.indices[context.client_id])
        logging.info("[WenClient] client=%d partition=%d images fingerprint=%s",
                     context.client_id, context.sampler.num_samples(),
                     partition.fingerprint(context.client_id))


class WenClient(simple.Client):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._configure_composable(
            lifecycle_strategy=WenLifecycle(),
            payload_strategy=DefaultPayloadStrategy(),
            training_strategy=DefaultTrainingStrategy(),
            reporting_strategy=DefaultReportingStrategy(),
            communication_strategy=DefaultCommunicationStrategy(),
        )
