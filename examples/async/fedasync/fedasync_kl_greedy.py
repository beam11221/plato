"""
FedAsync training session with KL-greedy oracle client selection.

Selects the top-K clients each round whose local label distribution is
most similar to the server's global test distribution (lowest KL divergence).
This is a diversity-only oracle selector — a controlled baseline before
combining speed and diversity into the full V_i selector.

How oracle data is generated
------------------------------
KLGreedyServer.configure() calls client_label_distribution.ensure_csv_current()
after the datasource is initialised.  That function:

  1. Computes a fingerprint of the config params that determine partition
     content (total_clients, partition_size, concentration, random_seed,
     partition_distribution, dataset_name).
  2. Compares the fingerprint to the sidecar <csv>.meta.json written
     alongside the existing CSV.
  3. Regenerates the CSV (using PLATO's own dirichlet.Sampler) if the CSV
     is missing or if any fingerprinted param has changed.

Because Config() is already loaded and datasource is live, the generation
always uses the exact same settings as the training run — no separate script
to remember to re-run, no risk of stale data.

Run
---
    cd plato/examples/async/fedasync
    uv run fedasync_kl_greedy.py -c configs/fedasync_CIFAR10_resnet18_100client_kl_greedy.toml

Logging
-------
Grep "[KLGreedy]" on the server log for per-round KL diagnostics.
Grep "[ClientLabelDist]" to see whether the CSV was regenerated or reused.
The round-1 sanity block (top-3 balanced / top-3 skewed clients) is
logged under "[KLGreedy] Round 1 Sanity Check".
"""

from __future__ import annotations

import logging

import fedasync_aggregation
import fedasync_algorithm
import fedasync_kl_greedy_selection

import client_label_distribution

from plato.clients import simple
from plato.servers import fedavg


# ---------------------------------------------------------------------------
# Custom server — wires in KL-greedy selection and auto-generates oracle CSV
# ---------------------------------------------------------------------------

class KLGreedyServer(fedavg.Server):
    """FedAsync server with KL-greedy oracle client selection.

    configure() is overridden to call ensure_csv_current() after the datasource
    is ready, so the oracle CSV is always consistent with the active TOML config.
    """

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        aggregation_strategy = (
            fedasync_aggregation.SequentialFedAsyncAggregationStrategy()
        )
        client_selection_strategy = (
            fedasync_kl_greedy_selection.KLGreedyClientSelectionStrategy()
        )

        logging.info(
            "[KLGreedy] Server initialised with KL-greedy oracle client selection."
        )
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=aggregation_strategy,
            client_selection_strategy=client_selection_strategy,
        )

    def configure(self) -> None:
        """Configure server, then ensure the label-distribution CSV is current.

        Ordering matters:
          1. super().configure() — sets up self.datasource, self.testset,
             and calls client_selection_strategy.setup().  The KLGreedy selector
             does NOT load the oracle CSV in setup() (it's lazy), so this is safe.
          2. ensure_csv_current() — uses self.datasource (now live) to run
             PLATO's dirichlet.Sampler for each client.  Regenerates only if
             the config fingerprint has changed since the last run.
          3. From this point, the selector's lazy oracle load on the first
             select_clients() call will find the correct CSV.
        """
        super().configure()

        logging.info("[KLGreedy] Checking label-distribution oracle CSV …")
        datasource = self.require_datasource()
        client_label_distribution.ensure_csv_current(datasource)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    algorithm = fedasync_algorithm.Algorithm
    client = simple.Client(algorithm=algorithm)
    server = KLGreedyServer(algorithm=algorithm)
    server.run(client)


if __name__ == "__main__":
    main()
