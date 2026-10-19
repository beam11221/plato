"""Tests for SumoRsuMixin: the waiting clock, the upload verdict, waste counts.

The mixin is exercised on top of a stub server rather than a real PLATO one, so
these tests state exactly what the mixin adds and nothing about what PLATO
already does.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import sumo_server
from stage0_filter import Stage0Config, Stage0FeasibilityGate
from sumo_gates import UPLOADED_KEY, RsuSelectionGate
from sumo_mobility import CoverageOracle
from sumo_server import SumoBaselineServer, SumoPolarisServer, SumoRsuMixin

from plato.servers.strategies.base import ServerContext

# Simulated seconds are measured against this arbitrary epoch, exactly as PLATO
# measures them against time.time() at boot.
EPOCH = 1_000_000.0


class StubServer:
    """The parts of plato's Server the mixin talks to, and nothing else."""

    def __init__(self):
        self.wall_time = EPOCH
        self.initial_wall_time = EPOCH
        self.training_clients: dict[int, dict] = {}
        self.reported_clients: list = []
        self.clients_per_round = 4
        self.current_round = 3
        self.reports: dict[str, SimpleNamespace] = {}
        self.context = ServerContext()
        self.client_selection_strategy = SimpleNamespace(name="inner-selection")
        self.aggregation_strategy = SimpleNamespace(name="inner-aggregation")
        self.select_calls: list[bool] = []
        self.processed: list[tuple[int, str]] = []
        self.configured = False
        self.wrapped_up = 0
        self.aggregations = 0
        self.updates: list = []
        self.closed = False

    def configure(self):
        self.configured = True

    async def _select_clients(self, for_next_batch=False):
        self.select_calls.append(for_next_batch)
        if not for_next_batch:
            # What plato's Server does at the top of _select_clients.
            self.current_round += 1

    async def _process_reports(self):
        self.aggregations += 1

    async def process_client_info(self, client_id, sid):
        self.processed.append((client_id, sid))

    async def wrap_up(self):
        self.wrapped_up += 1

    def server_will_close(self):
        self.closed = True


class GatedStubServer(SumoRsuMixin, StubServer):
    def __init__(self, oracle, stage0=None):
        super().__init__()
        self.install_sumo(oracle, stage0=stage0)


@pytest.fixture
def oracle() -> CoverageOracle:
    # Client 1 is in range for [0, 10]; client 2 for [100, 110].
    return CoverageOracle({1: [(0.0, 10.0)], 2: [(100.0, 110.0)]})


@pytest.fixture
def server(oracle, monkeypatch) -> GatedStubServer:
    # Tier lookup normally reads the gpu_tier CSVs via Config, which is not
    # initialised in a unit test.
    monkeypatch.setattr(
        sumo_server.gpu_tier_sleep,
        "tier_for",
        lambda cid: SimpleNamespace(tier_id="xavier" if cid == 1 else "thor"),
    )
    return GatedStubServer(oracle)


def add_report(server: GatedStubServer, client_id: int, start_t: float, duration: float):
    """Queue a client report that started at start_t and ran for `duration`."""
    sid = f"sid-{client_id}"
    server.reports[sid] = SimpleNamespace(
        client_id=client_id, training_time=duration, processing_time=0.0
    )
    server.training_clients[client_id] = {"start_time": EPOCH + start_t}
    return sid


class TestInstallation:
    def test_strategies_are_wrapped_in_the_gates(self, server):
        assert isinstance(server.client_selection_strategy, sumo_server.RsuSelectionGate)
        assert isinstance(server.aggregation_strategy, sumo_server.UploadGate)

    def test_the_original_strategies_are_kept_inside(self, server):
        assert server.client_selection_strategy.inner.name == "inner-selection"
        assert server.aggregation_strategy.inner.name == "inner-aggregation"

    def test_the_mixin_comes_before_the_real_servers(self):
        for cls in (SumoBaselineServer, SumoPolarisServer):
            mro = cls.__mro__
            assert mro.index(SumoRsuMixin) < mro.index(cls.__bases__[1])


class TestStage0Wiring:
    """Stage 5 (prompts/STAGE5_stage0_filter.md): the feasibility gate is
    installed BETWEEN RsuSelectionGate and the arm's own strategy, and is
    entirely opt-in -- stage0=None must leave every existing config's
    behaviour untouched."""

    def test_absent_by_default(self, server):
        assert server.stage0_gate is None
        assert server.client_selection_strategy.inner.name == "inner-selection"

    def test_none_source_is_also_absent(self, oracle):
        stub = GatedStubServer(oracle, stage0=Stage0Config("none"))
        assert stub.stage0_gate is None
        assert stub.client_selection_strategy.inner.name == "inner-selection"

    def test_enabled_sits_between_the_range_gate_and_the_arms_own_strategy(self, oracle):
        stage0 = Stage0Config("fleet_mean", radius_m=400.0, v_max_mps=19.444444)
        stub = GatedStubServer(oracle, stage0=stage0)

        assert isinstance(stub.client_selection_strategy, RsuSelectionGate)
        assert isinstance(stub.client_selection_strategy.inner, Stage0FeasibilityGate)
        assert stub.client_selection_strategy.inner.inner.name == "inner-selection"
        assert stub.stage0_gate is stub.client_selection_strategy.inner

    def test_final_summary_logs_the_stage0_tally_when_installed(self, oracle, caplog):
        stage0 = Stage0Config("fleet_mean", radius_m=400.0, v_max_mps=19.444444)
        stub = GatedStubServer(oracle, stage0=stage0)

        with caplog.at_level("INFO"):
            stub.server_will_close()

        assert "[Stage0] summary" in caplog.text

    def test_final_summary_omits_stage0_when_not_installed(self, server, caplog):
        with caplog.at_level("INFO"):
            server.server_will_close()

        assert "[Stage0] summary" not in caplog.text


class TestSumoTime:
    def test_starts_at_zero(self, server):
        assert server.sumo_time() == 0.0

    def test_tracks_the_simulated_clock(self, server):
        server.wall_time = EPOCH + 42.0
        assert server.sumo_time() == 42.0


class TestWaitingClock:
    def test_does_not_move_when_a_vehicle_is_in_range(self, server):
        asyncio.run(server._select_clients())
        assert server.sumo_time() == 0.0

    def test_does_not_move_while_clients_are_training(self, server):
        # Nobody is in range at t = 50, but an in-flight client will report and
        # move the clock by itself.
        server.wall_time = EPOCH + 50.0
        server.training_clients = {1: {"start_time": EPOCH}}
        asyncio.run(server._select_clients())
        assert server.sumo_time() == 50.0

    def test_idles_forward_to_the_next_vehicle(self, server):
        # t = 50: the gap between client 1 leaving (10) and client 2 arriving
        # (100). With nothing training, no report can ever arrive, so without
        # this the run would hang.
        server.wall_time = EPOCH + 50.0
        asyncio.run(server._select_clients())
        assert server.sumo_time() == pytest.approx(100.0)

    def test_the_clock_never_runs_backwards(self, server):
        server.wall_time = EPOCH + 50.0
        before = server.wall_time
        asyncio.run(server._select_clients())
        assert server.wall_time > before

    def test_a_mid_round_batch_does_not_move_the_clock(self, server):
        server.wall_time = EPOCH + 50.0
        asyncio.run(server._select_clients(for_next_batch=True))
        assert server.sumo_time() == 50.0

    def test_selection_still_happens(self, server):
        asyncio.run(server._select_clients())
        assert server.select_calls == [False]


class TestWaitingClockUsesTheSelectableSet:
    """Regression: the cpu arm hung at round 719 on 2026-09-06.

    PLATO withholds clients whose report arrived but has not been merged
    (servers/base.py:567-585). The guard used to ask "is ANY vehicle in range",
    so three in-range-but-withheld vehicles made it skip the idle while the
    selector got an empty pool -- nothing dispatched, so no report could ever
    arrive to wake the server.
    """

    def _pending(self, client_id: int):
        """One entry of PLATO's reported_clients heap, shaped as base.py builds it."""
        return (0.0, client_id, {"client_id": client_id})

    def test_idles_when_the_only_in_range_vehicle_is_awaiting_merge(self, server):
        # Client 1 is in range at t = 5, client 2 is not. Client 1 has reported
        # and is queued for merge, so it cannot be dispatched.
        server.wall_time = EPOCH + 5.0
        server.reported_clients = [self._pending(1)]

        asyncio.run(server._select_clients())

        # Must jump to client 2's window rather than sit at t = 5 forever.
        assert server.sumo_time() == pytest.approx(100.0)

    def test_does_not_idle_when_an_in_range_vehicle_is_dispatchable(self, server):
        server.wall_time = EPOCH + 5.0
        asyncio.run(server._select_clients())
        assert server.sumo_time() == 5.0

    def test_selectable_excludes_both_withheld_groups(self, server):
        server.training_clients = {1: {"start_time": EPOCH}}
        server.reported_clients = [self._pending(2)]
        assert server._selectable_clients() == set()


class TestUploadVerdict:
    def test_finishing_in_range_uploads(self, server):
        # Client 1 is in range for [0, 10]; dispatched at 2, done at 7.
        sid = add_report(server, client_id=1, start_t=2.0, duration=5.0)
        asyncio.run(server.process_client_info(1, sid))
        assert server.context.state[UPLOADED_KEY][1] is True

    def test_finishing_out_of_range_is_waste(self, server):
        # Dispatched at 8, done at 18 -- the vehicle left range at 10.
        sid = add_report(server, client_id=1, start_t=8.0, duration=10.0)
        asyncio.run(server.process_client_info(1, sid))
        assert server.context.state[UPLOADED_KEY][1] is False

    def test_finishing_exactly_at_the_window_edge_uploads(self, server):
        sid = add_report(server, client_id=1, start_t=0.0, duration=10.0)
        asyncio.run(server.process_client_info(1, sid))
        assert server.context.state[UPLOADED_KEY][1] is True

    def test_processing_time_counts_towards_the_deadline(self, server):
        sid = add_report(server, client_id=1, start_t=0.0, duration=10.0)
        server.reports[sid].processing_time = 0.5
        asyncio.run(server.process_client_info(1, sid))
        assert server.context.state[UPLOADED_KEY][1] is False

    def test_the_report_still_reaches_the_server(self, server):
        sid = add_report(server, client_id=1, start_t=8.0, duration=10.0)
        asyncio.run(server.process_client_info(1, sid))
        # A wasted update is dropped at aggregation, not here: PLATO's own
        # bookkeeping must still run or the client's slot is never freed.
        assert server.processed == [(1, sid)]


class TestWasteCounters:
    def test_counts_dispatches_and_waste_per_tier(self, server):
        asyncio.run(
            server.process_client_info(1, add_report(server, 1, 2.0, 5.0))
        )  # xavier, uploads
        asyncio.run(
            server.process_client_info(1, add_report(server, 1, 8.0, 10.0))
        )  # xavier, wasted
        asyncio.run(
            server.process_client_info(2, add_report(server, 2, 101.0, 2.0))
        )  # thor, uploads

        assert dict(server.dispatched_per_tier) == {"xavier": 2, "thor": 1}
        assert dict(server.wasted_per_tier) == {"xavier": 1}


class TestFinalSummary:
    """The end-of-run tally cannot live in wrap_up(): PLATO's wrap_up() calls
    _close(), which ends in os._exit(0), so nothing after it runs."""

    def test_the_summary_is_logged_before_the_server_closes(self, server, caplog):
        asyncio.run(server.process_client_info(1, add_report(server, 1, 8.0, 10.0)))

        with caplog.at_level("INFO"):
            server.server_will_close()

        assert "[SumoRsu] waste" in caplog.text
        assert "xavier=1/1" in caplog.text

    def test_closing_is_still_delegated(self, server):
        server.server_will_close()
        assert server.closed

    def test_a_run_with_no_reports_does_not_divide_by_zero(self, server):
        server.server_will_close()
        assert server.closed


class TestRoundSemantics:
    """A round is a federated-layer event: it exists only when an aggregation
    happens. A dispatch cycle whose update was lost to mobility is not a round.
    """

    def _queue(self, server, *uploaded: bool):
        server.updates = [SimpleNamespace(client_id=i) for i in range(len(uploaded))]
        server.context.state[UPLOADED_KEY] = dict(enumerate(uploaded))

    def test_a_delivered_update_aggregates_and_advances_the_round(self, server):
        self._queue(server, True)
        asyncio.run(server._process_reports())
        asyncio.run(server._select_clients())

        assert server.aggregations == 1
        assert server.current_round == 4  # started at 3

    def test_an_all_wasted_cycle_does_not_aggregate(self, server):
        self._queue(server, False)
        asyncio.run(server._process_reports())
        assert server.aggregations == 0

    def test_an_all_wasted_cycle_does_not_advance_the_round(self, server):
        self._queue(server, False)
        asyncio.run(server._process_reports())
        asyncio.run(server._select_clients())

        assert server.current_round == 3  # unchanged

    def test_a_partly_delivered_cycle_is_still_a_round(self, server):
        # The UploadGate drops the lost one; the round is real because something
        # reached the RSU.
        self._queue(server, False, True)
        asyncio.run(server._process_reports())
        asyncio.run(server._select_clients())

        assert server.aggregations == 1
        assert server.current_round == 4

    def test_clients_are_still_dispatched_after_a_wasted_cycle(self, server):
        self._queue(server, False)
        asyncio.run(server._process_reports())
        asyncio.run(server._select_clients())
        # The slot was freed and must be refilled, or concurrency drains.
        assert server.select_calls == [False]

    def test_dispatch_cycles_and_rounds_diverge_by_the_wasted_cycles(self, server):
        for uploaded in (True, False, False, True, True):
            self._queue(server, uploaded)
            asyncio.run(server._process_reports())
            asyncio.run(server._select_clients())

        assert server._dispatch_cycles == 5
        assert server.current_round == 3 + 3  # only the three delivered ones

    def test_the_flag_does_not_leak_into_the_next_cycle(self, server):
        self._queue(server, False)
        asyncio.run(server._process_reports())
        asyncio.run(server._select_clients())
        assert server._round_was_empty is False

    def test_a_mid_round_batch_never_touches_the_round_counter(self, server):
        self._queue(server, False)
        asyncio.run(server._process_reports())
        asyncio.run(server._select_clients(for_next_batch=True))
        assert server.current_round == 3
        assert server._dispatch_cycles == 0


class TestConfigureGuards:
    """The upload deadline assumes comm_time is zero; enforce it loudly."""

    def _configure_with(self, server, monkeypatch, **client_keys):
        monkeypatch.setattr(
            sumo_server,
            "Config",
            lambda: SimpleNamespace(clients=SimpleNamespace(**client_keys)),
        )
        server.configure()

    def test_accepts_the_intended_settings(self, server, monkeypatch):
        self._configure_with(server, monkeypatch, comm_simulation=True)
        assert server.configured

    def test_rejects_comm_simulation_off(self, server, monkeypatch):
        with pytest.raises(ValueError, match="comm_simulation"):
            self._configure_with(server, monkeypatch, comm_simulation=False)

    def test_rejects_compute_comm_time_on(self, server, monkeypatch):
        with pytest.raises(ValueError, match="compute_comm_time"):
            self._configure_with(
                server, monkeypatch, comm_simulation=True, compute_comm_time=True
            )
