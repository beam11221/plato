"""
Sequential staleness-aware aggregation for the FedAsync example.

When ``periodic_interval`` is short and server-side testing is slow, multiple
client reports accumulate in ``self.updates`` before the periodic task fires.
The stock FedAsync strategy keeps only ``updates[0]`` and discards the rest.

This variant treats the batch as if the clients had arrived one-per-tick:
each update is mixed into the running weights in arrival order, and the
staleness used for the mix is recomputed against an ``effective_round`` that
advances by one for each step. Earlier arrivals therefore get a lower
staleness; later arrivals in the same batch are penalised, which matches the
"as-if sequential" semantics we're after.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast

from plato.servers.strategies.aggregation.fedasync import (
    FedAsyncAggregationStrategy,
)
from plato.servers.strategies.base import ServerContext


class SequentialFedAsyncAggregationStrategy(FedAsyncAggregationStrategy):
    """Apply every pending update sequentially with its own staleness."""

    async def aggregate_weights(
        self,
        updates: list[SimpleNamespace],
        baseline_weights: dict,
        weights_received: list[dict],
        context: ServerContext,
    ) -> dict:
        if not updates:
            return baseline_weights

        algorithm = getattr(context, "algorithm", None)
        if algorithm is None or not hasattr(algorithm, "aggregate_weights"):
            raise AttributeError(
                "FedAsync requires an algorithm with 'aggregate_weights'."
            )
        algorithm = cast(Any, algorithm)

        current_round = context.current_round
        running_weights = baseline_weights

        pool = [
            (getattr(u, "client_id", "?"), getattr(u, "staleness", 0))
            for u in updates
        ]
        logging.info(
            "FedAsync: %d client(s) in aggregation pool: %s",
            len(pool),
            ", ".join(f"#{cid}(staleness={s})" for cid, s in pool),
        )
        logging.info(
            "FedAsync: sequentially applying %d update(s) in arrival order.",
            len(updates),
        )

        for idx, (update, weights) in enumerate(zip(updates, weights_received)):
            # update.staleness was set at arrival time as
            # current_round_at_arrival - starting_round. Since self.current_round
            # does not advance until wrap_up(), we simulate per-step version
            # bumps locally by adding idx.
            staleness = getattr(update, "staleness", 0) + idx
            effective_round = current_round + idx

            mixing = self.mixing_hyperparam
            if self.adaptive_mixing:
                mixing *= self._staleness_function(staleness)

            logging.info(
                "FedAsync: step %d - client #%s, effective_round=%d, "
                "staleness=%d, mixing=%.4f",
                idx + 1,
                getattr(update, "client_id", "?"),
                effective_round,
                staleness,
                mixing,
            )

            running_weights = await algorithm.aggregate_weights(
                running_weights, [weights], mixing=mixing
            )
            await asyncio.sleep(0)

        return running_weights
