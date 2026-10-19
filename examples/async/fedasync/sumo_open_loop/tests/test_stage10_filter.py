"""
Stage 10: the open-population Stage 0 gate.

Every Stage 7 defect survived against a fake that put state where the real code
does not, so these use the real PositionOracle, the real selector, the real
occupant resolver and the real mixin methods.  PLATO's Config is pointed at a
Stage 7b pre-flight config only so the predictor module can load its tiers.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
OPEN_LOOP = HERE.parent
FEDASYNC = OPEN_LOOP.parent
for _p in (FEDASYNC, FEDASYNC / "gpu_tiers", FEDASYNC / "gpu_tiers_dataset",
           FEDASYNC / "sumo_close_loop", OPEN_LOOP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_saved = sys.argv
sys.argv = ["pytest", "-c", str(OPEN_LOOP / "configs" / "stage7b_preflight" / "gpu_open.toml")]
try:
    import fedasync_polaris_vi_selection as selection  # noqa: E402
    from plato.config import Config  # noqa: E402

    Config()
finally:
    sys.argv = _saved

from occupant_identity import IdentityResolver, load_fleet  # noqa: E402
from open_stage0 import (  # noqa: E402
    FLEET_MEAN_T_HAT_S,
    OpenStage0Config,
    OpenStage0Gate,
    remaining_range_s,
    resolve_open_stage0,
)
from sumo_open_server import OpenPopulationMixin, OpenSlotSelectionGate  # noqa: E402
from sumo_position import PositionOracle  # noqa: E402

FLEET = load_fleet(FEDASYNC / "sumo_close_loop" / "input" / "clients_n20r400.csv",
                   FEDASYNC / "sumo_close_loop" / "output" / "client_sleep_times_n20r400.csv")
CFG = OpenStage0Config("tier_spec", 400.0, 19.444444, 600.0, "unused.csv")
V = 19.444444


class FakeTimeline:
    def __init__(self, occupants):
        self.occupants = dict(occupants)

    @property
    def tier_of_vehicle(self):
        return {v: t for v, t in self.occupants.values()}

    def vehicle_in_slot(self, slot):
        e = self.occupants.get(slot)
        return e[0] if e else None


def positions(**tracks):
    """veh_id -> (x at t=0, direction), 1 Hz samples for t = 0..10 at v_max."""
    samples = {}
    for veh, (x0, d) in tracks.items():
        samples[veh] = [(float(t), x0 + d * V * t, d) for t in range(11)]
    return PositionOracle(samples)


class Server(OpenPopulationMixin):
    def __init__(self, occupants, p_recur=0.0, t=0.0):
        self.timeline = FakeTimeline(occupants)
        self.resolver = IdentityResolver(self.timeline, 4126, p_recur, 20,
                                         FLEET if p_recur == 1.0 else None)
        self._t = t
        self.current_round = 7
        self._dur_sum, self._dur_n = {}, {}
        self._fleet_dur_sum, self._fleet_dur_n = 0.0, 0

    def sumo_time(self):
        return self._t

    def record(self, identity_id, duration):  # what process_client_info does on delivery
        self._dur_sum[identity_id] = self._dur_sum.get(identity_id, 0.0) + duration
        self._dur_n[identity_id] = self._dur_n.get(identity_id, 0) + 1
        self._fleet_dur_sum += duration
        self._fleet_dur_n += 1


class Capture:
    """Inner strategy that records the pool it was handed."""
    def __init__(self):
        self.pool = None

    def select_clients(self, pool, count, context):
        self.pool = list(pool)
        return list(pool)[:count]


# ---------------------------------------------------------------------------

class TestConfig:
    def test_absent_key_means_no_gate(self):
        assert not resolve_open_stage0(SimpleNamespace()).enabled

    def test_missing_geometry_raises(self):
        with pytest.raises(ValueError):
            resolve_open_stage0(SimpleNamespace(stage0_t_hat="tier_spec", stage0_r_i="vmax_distance"))

    def test_only_vmax_distance(self):
        cfg = dict(stage0_t_hat="tier_spec", coverage_radius_m=400, v_max_mps=V,
                   rsu_x_m=600, positions_csv="x")
        with pytest.raises(ValueError):
            resolve_open_stage0(SimpleNamespace(stage0_r_i="vmax_elapsed", **cfg))
        assert resolve_open_stage0(SimpleNamespace(stage0_r_i="vmax_distance", **cfg)).enabled

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError):
            resolve_open_stage0(SimpleNamespace(stage0_t_hat="fleet_mean"))


class TestRemainingRange:
    def test_eastbound_exits_at_1000(self):
        pos = positions(**{"fEB.1": (700.0, 1)})
        assert remaining_range_s(CFG, pos, "fEB.1", 0.0) == pytest.approx(300.0 / V)

    def test_westbound_exits_at_200(self):
        pos = positions(**{"fWB.1": (300.0, -1)})
        assert remaining_range_s(CFG, pos, "fWB.1", 2.0) == pytest.approx((300 - 2 * V - 200) / V)

    def test_unknown_vehicle_is_none(self):
        assert remaining_range_s(CFG, positions(), "nobody", 0.0) is None


class TestTHatFollowsTheOccupant:
    def test_tier_spec_uses_occupant_tier_not_slot_row(self):
        """Slot 1's fleet-file row is xavier; its occupant is a thor (H3 regression)."""
        srv = Server({1: ("fWB.9", "thor")})
        gate = OpenStage0Gate(Capture(), srv, CFG, positions(**{"fWB.9": (900.0, -1)}))
        t_hat, basis = gate.t_hat(srv.resolver.of_slot(1))
        assert basis == "spec"
        assert t_hat == pytest.approx(Server.predict_seconds_for_tier("thor"))
        assert t_hat != pytest.approx(Server.predict_seconds_for_tier("xavier"))

    def test_closed_regime_uses_fleet_identity_tier_but_trace_position(self):
        """p_recur = 1: T_hat from slot 1's fleet identity (xavier), R_i from
        the thor trace vehicle physically in the slot."""
        srv = Server({1: ("fWB.9", "thor")}, p_recur=1.0)
        cap = Capture()
        gate = OpenStage0Gate(cap, srv, CFG, positions(**{"fWB.9": (900.0, -1)}))
        # R_i = 700 / V = 36 s >= xavier 15 s -> passes
        assert gate.select_clients([1], 1, None) == [1]
        assert gate.t_hat(srv.resolver.of_slot(1))[0] == pytest.approx(15.0)


class TestObservedPerVehicle:
    CFG_OBS = OpenStage0Config("observed_per_vehicle", 400.0, V, 600.0, "unused.csv")

    def test_never_seen_gets_fleet_mean_start(self):
        srv = Server({1: ("fEB.1", "xavier")})
        gate = OpenStage0Gate(Capture(), srv, self.CFG_OBS, positions())
        assert gate.t_hat(srv.resolver.of_slot(1)) == (FLEET_MEAN_T_HAT_S, "fleet")

    def test_running_means(self):
        srv = Server({1: ("fEB.1", "xavier"), 2: ("fEB.2", "thor")})
        gate = OpenStage0Gate(Capture(), srv, self.CFG_OBS, positions())
        srv.record("fEB.1", 14.0)
        srv.record("fEB.1", 16.0)
        srv.record("fEB.7", 2.0)
        assert gate.t_hat(srv.resolver.of_slot(1)) == (15.0, "own")           # its own mean
        assert gate.t_hat(srv.resolver.of_slot(2)) == (pytest.approx(32 / 3), "fleet")  # fleet mean

    def test_no_spec_reaches_the_observed_filter(self, monkeypatch):
        monkeypatch.setattr(Server, "predict_seconds_for_tier",
                            staticmethod(lambda *_: (_ for _ in ()).throw(AssertionError("spec read"))))
        srv = Server({1: ("fEB.1", "xavier")})
        OpenStage0Gate(Capture(), srv, self.CFG_OBS, positions()).t_hat(srv.resolver.of_slot(1))


class TestPredicateAndFallback:
    def test_rejects_infeasible_and_passes_feasible(self):
        # slot 1: xavier near the exit (R = 50/V ~ 2.6 s < 15 s) -> reject
        # slot 2: thor with plenty of road (R = 700/V ~ 36 s >= 1 s) -> pass
        srv = Server({1: ("fEB.1", "xavier"), 2: ("fWB.2", "thor")})
        cap = Capture()
        gate = OpenStage0Gate(cap, srv, CFG, positions(**{"fEB.1": (950.0, 1), "fWB.2": (900.0, -1)}))
        gate.select_clients([1, 2], 2, None)
        assert cap.pool == [2]
        assert (gate.rejected_total, gate.fallback_rounds) == (1, 0)

    def test_empty_pool_falls_back_to_unfiltered(self):
        srv = Server({1: ("fEB.1", "xavier")})
        cap = Capture()
        gate = OpenStage0Gate(cap, srv, CFG, positions(**{"fEB.1": (950.0, 1)}))
        gate.select_clients([1], 1, None)
        assert cap.pool == [1] and gate.fallback_rounds == 1


class TestSelectorLookupThroughTheGate:
    def test_polaris_selector_found_two_levels_down(self):
        selector = selection.PolarisViClientSelectionStrategy()
        srv = Server({})
        gate = OpenStage0Gate(selector, srv, CFG, positions())
        srv.client_selection_strategy = OpenSlotSelectionGate(gate, srv)
        assert srv._polaris_selector() is selector

    def test_without_gate_unchanged(self):
        selector = selection.PolarisViClientSelectionStrategy()
        srv = Server({})
        srv.client_selection_strategy = OpenSlotSelectionGate(selector, srv)
        assert srv._polaris_selector() is selector

    def test_baseline_has_no_selector(self):
        srv = Server({})
        srv.client_selection_strategy = OpenSlotSelectionGate(
            OpenStage0Gate(Capture(), srv, CFG, positions()), srv)
        assert srv._polaris_selector() is None
