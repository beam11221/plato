"""
FedAsync training session with GPU-tier-aware sleep times.

How sleep times reach client processes
---------------------------------------
PLATO spawns each client as a separate OS subprocess with its own Config
singleton.  Writing to Config.client_sleep_times in the server process has
no effect on those subprocesses.

The correct path:
  1. A static CSV (gpu_tier_client_sleep_times_generated.csv) holds the
     tier-computed sleep time for every client.  It is generated once by
     gpu_tier_sleep.generate_client_sleep_csv() and checked in alongside
     the TOML.  Regenerate it whenever tiers.csv, partition_size, or epochs
     change (see README_gpu_tier.md).

  2. The TOML declares:
       [clients.simulation_distribution]
       distribution = "custom"
       csv_path     = ".../gpu_tier_client_sleep_times_generated.csv"
     PLATO's simulate_client_speed() reads this CSV inside every client
     process during Config init — no cross-process injection needed.

  3. ComposableTrainer.simulate_sleep_time() then calls
     time.sleep(Config.client_sleep_times[client_id - 1]) each epoch.

Logging
-------
GpuTierLogCallback.on_train_epoch_end fires in each client process and logs:
    [GpuTierSleep] round=N client_id=M tier_id=X dataset_size=D sleep_time=S s
to client_<id>.log inside the run directory.

Grep all client logs for "[GpuTierSleep]" to build the per-round table.

Run:
    cd plato/examples/async/fedasync
    uv run fedasync_gpu_tier.py -c configs/fedasync_CIFAR10_resnet18_100client_gpu_tier.toml
"""

from __future__ import annotations

import logging

import fedasync_algorithm
import fedasync_server
import gpu_tier_sleep

from plato.callbacks.trainer import TrainerCallback, resolve_num_samples
from plato.clients import simple
from plato.config import Config


# ---------------------------------------------------------------------------
# Per-round logging callback
# ---------------------------------------------------------------------------

class GpuTierLogCallback(TrainerCallback):
    """
    Logs tier_id, dataset_size, and scheduled sleep_time after each epoch.

    Reads Config.client_sleep_times[client_id-1], which is populated from
    the pre-generated CSV by PLATO's simulate_client_speed() in this client
    process.  Output goes to the client's own log file (client_<id>.log).
    """

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

        logging.info(
            "[GpuTierSleep] round=%d client_id=%d tier_id=%s "
            "dataset_size=%d sleep_time=%.6f s",
            getattr(trainer, "current_round", -1),
            client_id,
            tier_id,
            dataset_size,
            scheduled_sleep,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    algorithm = fedasync_algorithm.Algorithm
    client = simple.Client(
        algorithm=algorithm,
        trainer_callbacks=[GpuTierLogCallback],
    )
    server = fedasync_server.Server(algorithm=algorithm)
    server.run(client)


if __name__ == "__main__":
    main()
