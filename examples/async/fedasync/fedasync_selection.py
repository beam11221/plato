"""
Client selection strategy for the FedAsync example.

Filters out clients that are still training (present in ``training_clients``)
before delegating to random selection. This prevents the KeyError observed in
``process_client_info`` when the same client is picked again for a new round
while its previous round's report has not yet arrived.
"""

from __future__ import annotations

import logging

from plato.servers.strategies.client_selection.random_selection import (
    RandomSelectionStrategy,
)
from plato.servers.strategies.base import ServerContext


class FedAsyncClientSelectionStrategy(RandomSelectionStrategy):
    """Random selection that skips clients still training and caps the count
    at the number of currently-free process slots.

    Background: base.py always asks for ``clients_per_round`` new clients, even
    in async mode when most slots are still busy. Returning that many would
    drive the downstream loop in base.py:655-674 into ``UnboundLocalError``
    (no free process to assign). We cap the selection here so async
    replacement is 1-for-1 with freed slots, which is the correct semantics
    for asynchronous FL.
    """

    def select_clients(
        self,
        clients_pool: list[int],
        clients_count: int,
        context: ServerContext,
    ) -> list[int]:
        server = context.server
        busy = set(getattr(server, "training_clients", {}) or {})

        # Free slots = clients_per_round - currently-training. This matches
        # the number of client processes that just freed up and can accept a
        # new assignment.
        clients_per_round = getattr(server, "clients_per_round", clients_count)
        open_slots = max(0, clients_per_round - len(busy))

        filtered_pool = [c for c in clients_pool if c not in busy]
        effective_count = min(clients_count, open_slots, len(filtered_pool))

        if effective_count < clients_count:
            logging.info(
                "[FedAsync] %d client(s) still training, %d slot(s) open; "
                "selecting %d of %d requested.",
                len(busy),
                open_slots,
                effective_count,
                clients_count,
            )

        if effective_count == 0:
            logging.info(
                "[FedAsync] No free clients available for selection this tick."
            )
            return []

        return super().select_clients(filtered_pool, effective_count, context)
