"""
FedAsync + Polaris V_i selection training session.

Uses FedAsyncPolarisAggregationStrategy to track G_i (conv-layer gradient
bounds) and PolarisViClientSelectionStrategy to score clients as

    V_i = G_i / sqrt(1 + τ̂_i)

selecting the top-K per round.

Run (sanity-check config):
    cd plato/examples/async/fedasync
    uv run fedasync_polaris_vi.py \\
        -c configs/fedasync_CIFAR10_resnet18_100client_polaris_vi.toml

Grep server log for "[PolarisVi]" to extract per-round V_i scores and G_i stats.
"""

from __future__ import annotations

import fedasync_algorithm
import fedasync_polaris_vi_server

from plato.clients import simple


def main():
    algorithm = fedasync_algorithm.Algorithm
    client = simple.Client(algorithm=algorithm)
    server = fedasync_polaris_vi_server.Server(algorithm=algorithm)
    server.run(client)


if __name__ == "__main__":
    main()
