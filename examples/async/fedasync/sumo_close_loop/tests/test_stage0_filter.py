"""Tests for the Stage 0 feasibility gate (prompts/STAGE5_stage0_filter.md)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import stage0_filter
from stage0_filter import (
    DEFAULT_RI_MODE,
    FLEET_MEAN_T_HAT_S,
    Stage0Config,
    Stage0FeasibilityGate,
    resolve_r_i_mode,
    resolve_stage0_config,
    resolve_t_hat_source,
)
from sumo_mobility import CoverageOracle
from sumo_position import PositionOracle

from plato.servers.strategies.base import ServerContext


class RecordingSelection:
    """Inner selection strategy that records what the gate handed it."""

    def __init__(self):
        self.calls: list[tuple[list[int], int]] = []

    def setup(self, context):
        pass

    def teardown(self, context):
        pass

    def select_clients(self, clients_pool, clients_count, context):
        self.calls.append((list(clients_pool), clients_count))
        return list(clients_pool)[: max(clients_count, 1) if clients_pool else 0]

    def on_clients_selected(self, selected_clients, context):
        pass

    def on_reports_received(self, updates, context):
        pass


def make_context(sumo_t: float, round_no: int = 1):
    context = ServerContext()
    context.server = SimpleNamespace(sumo_time=lambda: sumo_t, current_round=round_no)
    return context


@pytest.fixture
def oracle() -> CoverageOracle:
    # Client 1 entered at t=0; client 2 entered at t=100.
    return CoverageOracle({1: [(0.0, 999.0)], 2: [(100.0, 999.0)]})


@pytest.fixture(autouse=True)
def stub_gpu_tier_sleep(monkeypatch):
    """Tests own T_hat directly; no real [gpu_tier] CSV needed."""
    monkeypatch.setattr(
        stage0_filter.gpu_tier_sleep,
        "tier_for",
        lambda cid: SimpleNamespace(tier_id="xavier" if cid == 1 else "thor"),
    )
    monkeypatch.setattr(
        stage0_filter.gpu_tier_sleep, "gpu_tops_predict_seconds", lambda cid, ds, ep: 5.0
    )
    monkeypatch.setattr(
        stage0_filter,
        "_workload",
        lambda: (2500, 1),
    )


class TestResolveTHatSource:
    def test_defaults_to_none(self):
        assert resolve_t_hat_source(None) == "none"

    @pytest.mark.parametrize(
        "raw", ["none", "fleet_mean", "tier_spec", "oracle", "ORACLE", " tier_spec "]
    )
    def test_accepts_every_documented_value(self, raw):
        assert resolve_t_hat_source(raw) in stage0_filter.T_HAT_SOURCES

    def test_rejects_a_typo(self):
        with pytest.raises(ValueError, match="stage0_t_hat"):
            resolve_t_hat_source("fleetmean")


class TestResolveStage0Config:
    def test_none_needs_no_other_keys(self):
        config = resolve_stage0_config(SimpleNamespace())
        assert config == Stage0Config("none")
        assert not config.enabled

    def test_fleet_mean_requires_the_two_constants(self):
        with pytest.raises(ValueError, match="coverage_radius_m"):
            resolve_stage0_config(SimpleNamespace(stage0_t_hat="fleet_mean"))

    def test_reads_the_constants_when_present(self):
        config = resolve_stage0_config(
            SimpleNamespace(
                stage0_t_hat="fleet_mean", coverage_radius_m=400.0, v_max_mps=19.444444
            )
        )
        assert config.enabled
        assert config.radius_m == 400.0
        assert config.v_max_mps == 19.444444

    def test_stage0_r_i_defaults_to_vmax_elapsed(self):
        """Stage 5b (prompts/STAGE5B_fix_Ri.md): every config written before
        this stage existed, including all of Stage 5 and Task 4, omits
        stage0_r_i entirely and MUST behave exactly as it always did."""
        config = resolve_stage0_config(
            SimpleNamespace(
                stage0_t_hat="tier_spec", coverage_radius_m=400.0, v_max_mps=19.444444
            )
        )
        assert config.r_i_mode == DEFAULT_RI_MODE == "vmax_elapsed"

    def test_oracle_exit_needs_no_extra_keys(self):
        config = resolve_stage0_config(
            SimpleNamespace(
                stage0_t_hat="tier_spec", coverage_radius_m=400.0,
                v_max_mps=19.444444, stage0_r_i="oracle_exit",
            )
        )
        assert config.r_i_mode == "oracle_exit"

    def test_vmax_distance_requires_rsu_x_m(self):
        with pytest.raises(ValueError, match="rsu_x_m"):
            resolve_stage0_config(
                SimpleNamespace(
                    stage0_t_hat="tier_spec", coverage_radius_m=400.0,
                    v_max_mps=19.444444, stage0_r_i="vmax_distance",
                )
            )

    def test_vmax_distance_reads_rsu_x_m_when_present(self):
        config = resolve_stage0_config(
            SimpleNamespace(
                stage0_t_hat="tier_spec", coverage_radius_m=400.0,
                v_max_mps=19.444444, stage0_r_i="vmax_distance", rsu_x_m=600.0,
            )
        )
        assert config.r_i_mode == "vmax_distance"
        assert config.rsu_x_m == 600.0


class TestResolveRIMode:
    def test_defaults_to_vmax_elapsed(self):
        assert resolve_r_i_mode(None) == "vmax_elapsed"

    @pytest.mark.parametrize(
        "raw", ["vmax_elapsed", "vmax_distance", "oracle_exit", "ORACLE_EXIT"]
    )
    def test_accepts_every_documented_value(self, raw):
        assert resolve_r_i_mode(raw) in stage0_filter.RI_MODES

    def test_rejects_a_typo(self):
        with pytest.raises(ValueError, match="stage0_r_i"):
            resolve_r_i_mode("current_speed")  # the explicitly-forbidden ablation


class TestRemainingRangeS:
    """r_i_mode dispatch (prompts/STAGE5B_fix_Ri.md, Step 1)."""

    def test_vmax_elapsed_is_a_lower_bound_from_the_window_entry(self, oracle):
        config = Stage0Config("fleet_mean", radius_m=400.0, v_max_mps=20.0)
        gate = Stage0FeasibilityGate(RecordingSelection(), oracle, config)
        # Entered at t=0, asked at t=10: crossing_s = 2*400/20 = 40.
        assert gate.remaining_range_s(1, 10.0) == pytest.approx(40.0 - 10.0)

    def test_vmax_elapsed_is_the_default_mode(self, oracle):
        config = Stage0Config("fleet_mean", radius_m=400.0, v_max_mps=20.0)
        assert config.r_i_mode == "vmax_elapsed"

    def test_none_when_not_in_coverage(self, oracle):
        config = Stage0Config("fleet_mean", radius_m=400.0, v_max_mps=20.0)
        gate = Stage0FeasibilityGate(RecordingSelection(), oracle, config)
        assert gate.remaining_range_s(99, 10.0) is None

    def test_oracle_exit_is_the_exact_true_remaining_dwell(self):
        # Client 1's window ends at t=50 exactly (not the toy fixture's 999).
        exact_oracle = CoverageOracle({1: [(0.0, 50.0)]})
        config = Stage0Config(
            "fleet_mean", radius_m=400.0, v_max_mps=20.0, r_i_mode="oracle_exit"
        )
        gate = Stage0FeasibilityGate(RecordingSelection(), exact_oracle, config)
        assert gate.remaining_range_s(1, 30.0) == pytest.approx(20.0)

    def test_oracle_exit_none_when_not_in_coverage(self, oracle):
        config = Stage0Config(
            "fleet_mean", radius_m=400.0, v_max_mps=20.0, r_i_mode="oracle_exit"
        )
        gate = Stage0FeasibilityGate(RecordingSelection(), oracle, config)
        assert gate.remaining_range_s(99, 10.0) is None

    def test_vmax_distance_reads_the_position_oracle(self, oracle):
        # RSU at 600, radius 400 -> exit boundary at 1000 for direction +1.
        # Client at x=800, 200m from the +1 exit, v_max=20 -> 10s.
        positions = PositionOracle({1: [(10.0, 800.0, 1)]})
        config = Stage0Config(
            "fleet_mean", radius_m=400.0, v_max_mps=20.0,
            r_i_mode="vmax_distance", rsu_x_m=600.0,
        )
        gate = Stage0FeasibilityGate(RecordingSelection(), oracle, config, positions)
        assert gate.remaining_range_s(1, 10.0) == pytest.approx(10.0)

    def test_vmax_distance_uses_the_minus_x_boundary_when_travelling_that_way(self, oracle):
        # direction -1 -> exit boundary at 200. Client at x=800 -> 600m away.
        positions = PositionOracle({1: [(10.0, 800.0, -1)]})
        config = Stage0Config(
            "fleet_mean", radius_m=400.0, v_max_mps=20.0,
            r_i_mode="vmax_distance", rsu_x_m=600.0,
        )
        gate = Stage0FeasibilityGate(RecordingSelection(), oracle, config, positions)
        assert gate.remaining_range_s(1, 10.0) == pytest.approx(30.0)

    def test_vmax_distance_none_when_position_is_unknown(self, oracle):
        positions = PositionOracle({1: [(10.0, 800.0, 1)]})
        config = Stage0Config(
            "fleet_mean", radius_m=400.0, v_max_mps=20.0,
            r_i_mode="vmax_distance", rsu_x_m=600.0,
        )
        gate = Stage0FeasibilityGate(RecordingSelection(), oracle, config, positions)
        assert gate.remaining_range_s(1, 999.0) is None  # far outside the sample

    def test_vmax_distance_requires_a_position_oracle_at_construction(self, oracle):
        config = Stage0Config(
            "fleet_mean", radius_m=400.0, v_max_mps=20.0,
            r_i_mode="vmax_distance", rsu_x_m=600.0,
        )
        with pytest.raises(ValueError, match="PositionOracle"):
            Stage0FeasibilityGate(RecordingSelection(), oracle, config, position_oracle=None)

    def test_disabled_config_never_requires_a_position_oracle(self, oracle):
        # r_i_mode is irrelevant when the gate is off -- must not raise.
        config = Stage0Config("none", r_i_mode="vmax_distance")
        Stage0FeasibilityGate(RecordingSelection(), oracle, config, position_oracle=None)


class TestTHat:
    def test_fleet_mean_is_the_fixed_constant(self, oracle):
        config = Stage0Config("fleet_mean", radius_m=400.0, v_max_mps=20.0)
        gate = Stage0FeasibilityGate(RecordingSelection(), oracle, config)
        assert gate.t_hat(1, 2500, 1) == FLEET_MEAN_T_HAT_S

    def test_tier_spec_calls_gpu_tops_predict_seconds(self, oracle):
        config = Stage0Config("tier_spec", radius_m=400.0, v_max_mps=20.0)
        gate = Stage0FeasibilityGate(RecordingSelection(), oracle, config)
        assert gate.t_hat(1, 2500, 1) == 5.0  # stubbed above

    def test_oracle_uses_the_same_call_as_tier_spec(self, oracle):
        # oracle vs tier_spec is entirely gpu_tier_sleep's own oracle_sleeps()
        # config switch -- this gate calls the identical function either way.
        config = Stage0Config("oracle", radius_m=400.0, v_max_mps=20.0)
        gate = Stage0FeasibilityGate(RecordingSelection(), oracle, config)
        assert gate.t_hat(1, 2500, 1) == 5.0


class TestSelectClients:
    def test_disabled_is_a_pure_passthrough(self, oracle):
        inner = RecordingSelection()
        gate = Stage0FeasibilityGate(inner, oracle, Stage0Config("none"))
        result = gate.select_clients([1, 2], 1, make_context(sumo_t=10.0))
        assert result == [1]
        assert inner.calls == [([1, 2], 1)]

    def test_rejects_a_candidate_whose_window_cannot_cover_t_hat(self):
        # v_max=20, radius=400 -> crossing_s=40. Client 1 entered at t=0, so by
        # t=39 only 1s of its predicted 40s dwell is left -- well under the
        # stubbed T_hat of 5s. Client 2 just entered (t=39), so it has the
        # full 40s left and clears T_hat easily. Two candidates, so this
        # exercises a real rejection rather than the empty-pool fallback.
        two_client_oracle = CoverageOracle({1: [(0.0, 999.0)], 2: [(39.0, 999.0)]})
        inner = RecordingSelection()
        config = Stage0Config("tier_spec", radius_m=400.0, v_max_mps=20.0)
        gate = Stage0FeasibilityGate(inner, two_client_oracle, config)

        gate.select_clients([1, 2], 2, make_context(sumo_t=39.0))

        assert inner.calls == [([2], 2)]  # only client 2 passed the predicate

    def test_accepts_a_candidate_with_plenty_of_range_left(self, oracle):
        inner = RecordingSelection()
        config = Stage0Config("tier_spec", radius_m=400.0, v_max_mps=20.0)
        gate = Stage0FeasibilityGate(inner, oracle, config)

        gate.select_clients([1], 1, make_context(sumo_t=10.0))

        assert inner.calls == [([1], 1)]

    def test_falls_back_to_the_unfiltered_pool_when_everyone_is_rejected(self, oracle):
        inner = RecordingSelection()
        config = Stage0Config("tier_spec", radius_m=400.0, v_max_mps=20.0)
        gate = Stage0FeasibilityGate(inner, oracle, config)

        result = gate.select_clients([1], 2, make_context(sumo_t=39.0))

        # The original pool reaches the inner strategy...
        assert inner.calls == [([1], 2)]
        # ...and dispatch still happens instead of idling the round.
        assert result == [1]
        assert gate.fallback_rounds == 1

    def test_never_falls_back_when_at_least_one_candidate_passes(self, oracle):
        inner = RecordingSelection()
        config = Stage0Config("tier_spec", radius_m=400.0, v_max_mps=20.0)
        gate = Stage0FeasibilityGate(inner, oracle, config)

        gate.select_clients([1, 2], 2, make_context(sumo_t=10.0))

        assert gate.fallback_rounds == 0

    def test_empty_pool_is_a_passthrough_not_a_fallback(self, oracle):
        inner = RecordingSelection()
        config = Stage0Config("tier_spec", radius_m=400.0, v_max_mps=20.0)
        gate = Stage0FeasibilityGate(inner, oracle, config)

        result = gate.select_clients([], 1, make_context(sumo_t=10.0))

        assert result == []
        assert gate.fallback_rounds == 0

    def test_counters_accumulate_across_rounds(self, oracle):
        inner = RecordingSelection()
        config = Stage0Config("tier_spec", radius_m=400.0, v_max_mps=20.0)
        gate = Stage0FeasibilityGate(inner, oracle, config)

        gate.select_clients([1], 1, make_context(sumo_t=10.0))  # passes
        gate.select_clients([1], 1, make_context(sumo_t=39.0))  # rejected, fallback

        assert gate.rounds_seen == 2
        assert gate.candidates_seen == 2
        assert gate.rejected_total == 1
        assert gate.fallback_rounds == 1

    def test_logs_the_per_round_and_per_selection_lines(self, oracle, caplog):
        config = Stage0Config("tier_spec", radius_m=400.0, v_max_mps=20.0)
        gate = Stage0FeasibilityGate(RecordingSelection(), oracle, config)

        with caplog.at_level("INFO"):
            gate.select_clients([1], 1, make_context(sumo_t=10.0, round_no=7))

        assert "round=7" in caplog.text
        assert "eligible_after_stage0=1" in caplog.text
        assert "selected_id=1" in caplog.text
        assert "selected_tier=xavier" in caplog.text


class TestLifecycleDelegation:
    def test_setup_teardown_and_hooks_pass_through(self, oracle):
        inner = RecordingSelection()
        calls = []
        inner.setup = lambda context: calls.append("setup")
        inner.teardown = lambda context: calls.append("teardown")
        inner.on_clients_selected = lambda sel, ctx: calls.append("selected")
        inner.on_reports_received = lambda upd, ctx: calls.append("reports")

        gate = Stage0FeasibilityGate(inner, oracle, Stage0Config("none"))
        context = make_context(sumo_t=0.0)
        gate.setup(context)
        gate.teardown(context)
        gate.on_clients_selected([], context)
        gate.on_reports_received([], context)

        assert calls == ["setup", "teardown", "selected", "reports"]


class TestLogSummary:
    def test_reports_zero_when_nothing_ran(self, oracle, caplog):
        gate = Stage0FeasibilityGate(
            RecordingSelection(), oracle, Stage0Config("fleet_mean", 400.0, 20.0)
        )
        with caplog.at_level("INFO"):
            gate.log_summary()
        assert "rounds=0" in caplog.text
