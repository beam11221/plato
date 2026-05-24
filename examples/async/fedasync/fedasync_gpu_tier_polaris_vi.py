"""
FedAsync + GPU-tier speed simulation + Polaris V_i client selection.

Combines:
  * GpuTierLogCallback from fedasync_gpu_tier.py — logs per-epoch tier/sleep
    info to each client's log file (grep "[GpuTierSleep]" across client logs).
  * fedasync_polaris_vi_server.Server — tracks G_i from conv-layer weight
    deltas and selects clients via V_i = G_i / sqrt(1 + τ̂_i).

Run:
    cd plato/examples/async/fedasync
    uv run fedasync_gpu_tier_polaris_vi.py \\
        -c configs/fedasync_CIFAR10_resnet18_100client_gpu_tier_polaris_vi.toml

Grep server log for "[PolarisVi]" to extract per-round V_i scores and G_i stats.
Grep client logs for "[GpuTierSleep]" to extract per-round tier/sleep info.
"""

from __future__ import annotations

import fedasync_algorithm
import fedasync_polaris_vi_server
import gpu_tier_sleep

from plato.callbacks.trainer import TrainerCallback, resolve_num_samples
from plato.clients import simple
from plato.config import Config


class GpuTierLogCallback(TrainerCallback):
    """Logs tier_id, dataset_size, and scheduled sleep_time after each epoch."""

    def on_train_epoch_end(self, trainer, config, **kwargs):
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

        import logging
        logging.info(
            "[GpuTierSleep] round=%d client_id=%d tier_id=%s "
            "dataset_size=%d sleep_time=%.6f s",
            getattr(trainer, "current_round", -1),
            client_id,
            tier_id,
            dataset_size,
            scheduled_sleep,
        )


def main():
    algorithm = fedasync_algorithm.Algorithm
    client = simple.Client(
        algorithm=algorithm,
        trainer_callbacks=[GpuTierLogCallback],
    )
    server = fedasync_polaris_vi_server.Server(algorithm=algorithm)
    server.run(client)


if __name__ == "__main__":
    main()
