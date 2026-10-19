"""
The two places RSU coverage changes FedAsync's behaviour.

Both are written as *wrappers* around whatever strategy an arm already uses, so
the four arms of the benchmark (baseline / gpu / cpu / observed) keep their own
selection and aggregation code untouched.  Coverage is the only thing added.

    RsuSelectionGate    a vehicle can only be dispatched while it is in range,
                        and only reports that actually reached the RSU are
                        allowed to update the selector's state.

    UploadGate          a vehicle that finishes training out of range cannot
                        upload; its update is discarded and the round's compute
                        is counted as waste.

The score itself is deliberately unchanged: V_i = G_i / sqrt(1 + tau_hat_i),
with no mobility term.  Mobility restricts *who may be asked*, not *who looks
best* -- that stays for a later step.

The uploaded/not-uploaded flags are written by the server into
`context.state[UPLOADED_KEY]` as {client_id: bool}; both gates read them from
there.  A client can only have one report in flight at a time, so one slot per
client is enough and entries are simply overwritten.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from plato.servers.strategies.base import (
    AggregationStrategy,
    ClientSelectionStrategy,
    ServerContext,
)

# context.state key holding {client_id: bool} — did this client's last report
# reach the RSU?
UPLOADED_KEY = "sumo_uploaded"


def _uploaded(update: SimpleNamespace, context: ServerContext) -> bool:
    """Whether this update reached the RSU. Unknown clients count as uploaded."""
    flags = context.state.get(UPLOADED_KEY, {})
    return bool(flags.get(getattr(update, "client_id", None), True))


def delivered(
    updates: list[SimpleNamespace], context: ServerContext
) -> list[SimpleNamespace]:
    """The subset of *updates* that actually reached the RSU."""
    return [update for update in updates if _uploaded(update, context)]


class RsuSelectionGate(ClientSelectionStrategy):
    """Restrict an inner selection strategy to vehicles inside RSU range."""

    def __init__(self, inner: ClientSelectionStrategy, oracle) -> None:
        self.inner = inner
        self.oracle = oracle

    # -- lifecycle: pure delegation ------------------------------------

    def setup(self, context: ServerContext) -> None:
        self.inner.setup(context)

    def teardown(self, context: ServerContext) -> None:
        self.inner.teardown(context)

    def on_clients_selected(
        self, selected_clients: list[int], context: ServerContext
    ) -> None:
        self.inner.on_clients_selected(selected_clients, context)

    # -- the two gated hooks -------------------------------------------

    @staticmethod
    def _open_slots(server) -> int:
        """How many clients it takes to bring the pipeline back to full depth.

        A client occupies a slot from the moment it is dispatched until its
        update is merged, so the depth is `training_clients` (dispatched, no
        report yet) plus `reported_clients` (report in, not yet merged). Under
        `simulate_wall_time` a report arrives long before its *simulated* finish
        time, so counting only `training_clients` would treat a vehicle that is
        still training in simulated time as idle -- dispatching it again and
        inflating concurrency, which is the denominator of every staleness in
        the experiment.

        Asking for the deficit rather than for the `clients_count` PLATO passed
        also makes the pipeline self-healing. PLATO refills with exactly as many
        clients as it just merged (servers/base.py:591), so a round where
        coverage blocked a refill would otherwise lose that slot for the rest of
        the run -- the drain of gpu_tiers/HANDOFF.md section 4.9. When nothing
        was missed the two numbers are equal, so normal rounds are unaffected.
        """
        training = len(getattr(server, "training_clients", {}) or {})
        awaiting_merge = len(getattr(server, "reported_clients", []) or [])
        return max(0, server.clients_per_round - training - awaiting_merge)

    def select_clients(
        self,
        clients_pool: list[int],
        clients_count: int,
        context: ServerContext,
    ) -> list[int]:
        server = context.server
        t = server.sumo_time()
        eligible = [cid for cid in clients_pool if self.oracle.in_coverage(cid, t)]

        open_slots = self._open_slots(server)

        # Unconditional (not just when coverage actually filtered someone) so
        # Stage 5's per-round pool-size accounting (prompts/STAGE5_stage0_
        # filter.md, Step 3) has an eligible_after_busy figure -- len(clients_
        # pool), already busy-filtered by PLATO before this gate ever sees it
        # -- for every round, joinable by round= with stage0_filter.py's own
        # per-round line.
        logging.info(
            "[SumoRsu] round=%d t=%.1f s: %d of %d client(s) in RSU range; "
            "%d slot(s) open.",
            getattr(server, "current_round", -1),
            t,
            len(eligible),
            len(clients_pool),
            open_slots,
        )

        return self.inner.select_clients(eligible, open_slots, context)

    def on_reports_received(
        self, updates: list[SimpleNamespace], context: ServerContext
    ) -> None:
        # A report that never reached the RSU teaches the server nothing. This
        # matters most for the `observed` arm, whose whole estimator is the last
        # staleness a client reported: letting a lost report through would give
        # it knowledge the RSU never actually received.
        delivered = [u for u in updates if _uploaded(u, context)]
        self.inner.on_reports_received(delivered, context)


class UploadGate(AggregationStrategy):
    """Drop updates from vehicles that were out of range when training ended."""

    def __init__(self, inner: AggregationStrategy) -> None:
        self.inner = inner

    # -- lifecycle: pure delegation ------------------------------------

    def setup(self, context: ServerContext) -> None:
        self.inner.setup(context)

    def teardown(self, context: ServerContext) -> None:
        self.inner.teardown(context)

    async def aggregate_deltas(
        self,
        updates: list[SimpleNamespace],
        deltas_received: list[Any],
        context: ServerContext,
    ) -> Any:
        return await self.inner.aggregate_deltas(updates, deltas_received, context)

    # -- the gated hook -------------------------------------------------

    async def aggregate_weights(
        self,
        updates: list[SimpleNamespace],
        baseline_weights: dict,
        weights_received: list[dict],
        context: ServerContext,
    ) -> dict:
        delivered = [
            (update, weights)
            for update, weights in zip(updates, weights_received)
            if _uploaded(update, context)
        ]

        dropped = len(updates) - len(delivered)
        if dropped:
            logging.info(
                "[SumoRsu] Discarding %d of %d update(s): the vehicle left RSU "
                "range before training finished.",
                dropped,
                len(updates),
            )

        if not delivered:
            # Nothing reached the RSU this round. The global model must not move.
            return baseline_weights

        return await self.inner.aggregate_weights(
            [update for update, _ in delivered],
            baseline_weights,
            [weights for _, weights in delivered],
            context,
        )
