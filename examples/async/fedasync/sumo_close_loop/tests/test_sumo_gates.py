"""Tests for the two coverage gates: selection and upload."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent

from sumo_gates import UPLOADED_KEY, RsuSelectionGate, UploadGate  # noqa: E402
from sumo_mobility import CoverageOracle  # noqa: E402

from plato.servers.strategies.base import ServerContext  # noqa: E402


class RecordingSelection:
    """Inner selection strategy that records what the gate handed it."""

    def __init__(self):
        self.calls: list[tuple[list[int], int]] = []
        self.reports_seen: list[list[int]] = []
        self.setup_called = False

    def setup(self, context):
        self.setup_called = True

    def teardown(self, context):
        pass

    def select_clients(self, clients_pool, clients_count, context):
        self.calls.append((list(clients_pool), clients_count))
        return list(clients_pool)[:clients_count]

    def on_clients_selected(self, selected_clients, context):
        pass

    def on_reports_received(self, updates, context):
        self.reports_seen.append([u.client_id for u in updates])


class RecordingAggregation:
    """Inner aggregation strategy that records what survived the gate."""

    def __init__(self):
        self.seen: list[list[int]] = []

    def setup(self, context):
        pass

    def teardown(self, context):
        pass

    async def aggregate_deltas(self, updates, deltas_received, context):
        return {}

    async def aggregate_weights(
        self, updates, baseline_weights, weights_received, context
    ):
        self.seen.append([u.client_id for u in updates])
        return {"aggregated": [w["id"] for w in weights_received]}


def make_context(
    oracle,
    sumo_t: float,
    busy: dict | None = None,
    per_round: int = 4,
    awaiting_merge: int = 0,
):
    """A ServerContext with just enough server for the gates to work."""
    context = ServerContext()
    context.server = SimpleNamespace(
        sumo_time=lambda: sumo_t,
        training_clients=busy or {},
        reported_clients=[None] * awaiting_merge,
        clients_per_round=per_round,
        current_round=1,
    )
    context.total_clients = 5
    return context


@pytest.fixture
def oracle() -> CoverageOracle:
    # Clients 1 and 2 are in range at t = 5; 3 is not; 4 and 5 never are.
    return CoverageOracle(
        {
            1: [(0.0, 10.0)],
            2: [(4.0, 8.0)],
            3: [(20.0, 30.0)],
            4: [(40.0, 50.0)],
            5: [(60.0, 70.0)],
        }
    )


class TestSelectionGate:
    def test_only_clients_in_range_reach_the_inner_strategy(self, oracle):
        inner = RecordingSelection()
        gate = RsuSelectionGate(inner, oracle)
        context = make_context(oracle, sumo_t=5.0)

        gate.select_clients([1, 2, 3, 4, 5], 4, context)

        eligible, _ = inner.calls[0]
        assert eligible == [1, 2]

    def test_returns_empty_when_nobody_is_in_range(self, oracle):
        gate = RsuSelectionGate(RecordingSelection(), oracle)
        context = make_context(oracle, sumo_t=15.0)

        assert gate.select_clients([1, 2, 3, 4, 5], 4, context) == []

    def test_refills_the_pipeline_back_to_full_depth(self, oracle):
        """The anti-drain rule.

        PLATO refills with as many clients as it just merged. If the gate
        forwarded that number, a round where coverage blocked a refill would
        lose the slot for good. Asking for the deficit heals it.
        """
        inner = RecordingSelection()
        gate = RsuSelectionGate(inner, oracle)
        # 4 slots, 1 busy -> 3 to fill, even though PLATO only asked for 1.
        context = make_context(oracle, sumo_t=5.0, busy={9: {}}, per_round=4)

        gate.select_clients([1, 2, 3, 4, 5], 1, context)

        _, requested = inner.calls[0]
        assert requested == 3

    def test_reports_awaiting_merge_still_hold_their_slot(self, oracle):
        """Under simulate_wall_time a report arrives long before its simulated
        finish time. Counting only training_clients would call that vehicle idle
        and dispatch it again, inflating concurrency."""
        inner = RecordingSelection()
        gate = RsuSelectionGate(inner, oracle)
        context = make_context(
            oracle, sumo_t=5.0, busy={9: {}}, per_round=4, awaiting_merge=2
        )

        gate.select_clients([1, 2, 3, 4, 5], 1, context)

        _, requested = inner.calls[0]
        assert requested == 1

    def test_matches_platos_own_refill_count_when_the_pipeline_is_full(self, oracle):
        """The normal case: 4 slots, 1 just merged, 3 still awaiting merge.
        PLATO asks for 1 and the gate must agree, or the experiment's
        concurrency is not the one the config declares."""
        inner = RecordingSelection()
        gate = RsuSelectionGate(inner, oracle)
        context = make_context(
            oracle, sumo_t=5.0, per_round=4, awaiting_merge=3
        )

        gate.select_clients([1, 2, 3, 4, 5], 1, context)

        _, requested = inner.calls[0]
        assert requested == 1

    def test_no_free_slots_asks_for_none(self, oracle):
        inner = RecordingSelection()
        gate = RsuSelectionGate(inner, oracle)
        context = make_context(
            oracle, sumo_t=5.0, busy={7: {}, 8: {}, 9: {}, 10: {}}, per_round=4
        )

        gate.select_clients([1, 2, 3, 4, 5], 2, context)

        _, requested = inner.calls[0]
        assert requested == 0

    def test_lost_reports_do_not_update_the_selector(self, oracle):
        """Critical for the `observed` arm: the RSU cannot learn from a report
        it never received."""
        inner = RecordingSelection()
        gate = RsuSelectionGate(inner, oracle)
        context = make_context(oracle, sumo_t=5.0)
        context.state[UPLOADED_KEY] = {1: True, 2: False}

        updates = [SimpleNamespace(client_id=1), SimpleNamespace(client_id=2)]
        gate.on_reports_received(updates, context)

        assert inner.reports_seen == [[1]]

    def test_unknown_upload_status_counts_as_delivered(self, oracle):
        inner = RecordingSelection()
        gate = RsuSelectionGate(inner, oracle)
        context = make_context(oracle, sumo_t=5.0)

        gate.on_reports_received([SimpleNamespace(client_id=1)], context)

        assert inner.reports_seen == [[1]]

    def test_lifecycle_is_delegated(self, oracle):
        inner = RecordingSelection()
        gate = RsuSelectionGate(inner, oracle)
        gate.setup(make_context(oracle, sumo_t=0.0))
        assert inner.setup_called

    def test_logs_every_round_not_just_when_coverage_filters_someone(self, oracle, caplog):
        """Stage 5's per-round pool-size accounting (prompts/STAGE5_stage0_
        filter.md, Step 3) needs eligible_after_busy for EVERY round, so this
        can no longer be conditional on coverage actually removing a client."""
        gate = RsuSelectionGate(RecordingSelection(), oracle)
        context = make_context(oracle, sumo_t=5.0)
        context.server.current_round = 42

        with caplog.at_level("INFO"):
            # Both clients 1 and 2 are in range at t=5 -- nobody is filtered.
            gate.select_clients([1, 2], 4, context)

        assert "round=42" in caplog.text
        assert "2 of 2 client(s) in RSU range" in caplog.text


class TestUploadGate:
    def test_drops_updates_that_could_not_upload(self):
        inner = RecordingAggregation()
        gate = UploadGate(inner)
        context = ServerContext()
        context.state[UPLOADED_KEY] = {1: True, 2: False, 3: True}

        updates = [SimpleNamespace(client_id=cid) for cid in (1, 2, 3)]
        weights = [{"id": cid} for cid in (1, 2, 3)]

        result = asyncio.run(
            gate.aggregate_weights(updates, {"baseline": True}, weights, context)
        )

        assert inner.seen == [[1, 3]]
        # The surviving weights must stay paired with their own updates.
        assert result == {"aggregated": [1, 3]}

    def test_returns_the_baseline_untouched_when_nothing_arrived(self):
        inner = RecordingAggregation()
        gate = UploadGate(inner)
        context = ServerContext()
        context.state[UPLOADED_KEY] = {1: False}
        baseline = {"baseline": True}

        result = asyncio.run(
            gate.aggregate_weights(
                [SimpleNamespace(client_id=1)], baseline, [{"id": 1}], context
            )
        )

        assert result is baseline
        assert inner.seen == []

    def test_passes_everything_through_when_all_uploads_succeeded(self):
        inner = RecordingAggregation()
        gate = UploadGate(inner)
        context = ServerContext()
        context.state[UPLOADED_KEY] = {1: True, 2: True}

        asyncio.run(
            gate.aggregate_weights(
                [SimpleNamespace(client_id=1), SimpleNamespace(client_id=2)],
                {},
                [{"id": 1}, {"id": 2}],
                context,
            )
        )

        assert inner.seen == [[1, 2]]
