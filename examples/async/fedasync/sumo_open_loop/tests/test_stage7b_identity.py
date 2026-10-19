"""
Stage 7b: the occupant identity, and the three Stage 7 siblings it closes.

Unlike test_stage7_population.py, these tests import the REAL PLATO-side
classes -- `FedAsyncPolarisAggregationStrategy` wrapped in the real
`UploadGate`, the real `PolarisViClientSelectionStrategy`, the real
`gpu_tier_sleep` predictor -- because every defect Stage 7b fixes survived
precisely by being tested against a fake that put things where the real code
does not.  PLATO's Config singleton is pointed at a Stage 7b pre-flight config
for that; nothing here trains or opens a socket.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
OPEN_LOOP = HERE.parent
FEDASYNC = OPEN_LOOP.parent
for _p in (FEDASYNC, FEDASYNC / "gpu_tiers", FEDASYNC / "gpu_tiers_dataset",
           FEDASYNC / "sumo_close_loop", OPEN_LOOP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CONFIG = OPEN_LOOP / "configs" / "stage7b_preflight" / "gpu_open.toml"
_saved_argv = sys.argv
sys.argv = ["pytest", "-c", str(CONFIG)]
try:
    import client_seeded_trainer  # noqa: E402
    import fedasync_polaris_vi_selection as selection  # noqa: E402
    import fedasync_polaris_vi_server as polaris_server  # noqa: E402
    import gpu_tier_sleep  # noqa: E402
    from plato.config import Config  # noqa: E402

    Config()  # parse once, while argv points at the config
finally:
    sys.argv = _saved_argv

from occupant_identity import (  # noqa: E402
    IdentityResolver,
    OccupantIdentity,
    identity_seed,
    load_fleet,
    partition_fingerprint,
)
from stage7_population import (  # noqa: E402
    UNSEEN_GRADIENT_BOUND,
    partition_index,
    reset_slot_state,
    vehicle_sleep_s,
)
from sumo_gates import UploadGate  # noqa: E402
from sumo_open_server import OpenPopulationMixin  # noqa: E402

FLEET_CLIENTS = FEDASYNC / "sumo_close_loop" / "input" / "clients_n20r400.csv"
FLEET_SLEEPS = FEDASYNC / "sumo_close_loop" / "output" / "client_sleep_times_n20r400.csv"
M = 20
BASE_SEED = 4126


class FakeTimeline:
    """Only what IdentityResolver reads: who is in which slot, and tiers."""

    def __init__(self, occupants: dict[int, tuple[str, str]]) -> None:
        self.occupants = dict(occupants)

    @property
    def tier_of_vehicle(self) -> dict[str, str]:
        return {v: t for v, t in self.occupants.values()}

    def vehicle_in_slot(self, slot):
        entry = self.occupants.get(slot)
        return entry[0] if entry else None

    def occupied_slots(self):
        return sorted(self.occupants)


# ---------------------------------------------------------------------------
# The identity itself
# ---------------------------------------------------------------------------

class TestResolver:
    def test_open_identity_is_the_trace_vehicle(self):
        timeline = FakeTimeline({1: ("fWB.7", "thor")})
        resolver = IdentityResolver(timeline, BASE_SEED, 0.0, M)
        ident = resolver.of_slot(1)
        assert ident == OccupantIdentity(
            "fWB.7", "thor", vehicle_sleep_s(BASE_SEED, "fWB.7", "thor"),
            partition_index("fWB.7", M),
        )
        assert resolver.of_slot(2) is None

    def test_closed_identity_is_bound_to_the_slot(self):
        """At p_recur = 1 slot 1 is fleet vehicle 1 (xavier) whoever the trace
        puts in its window -- here a thor, then an orin."""
        fleet = load_fleet(FLEET_CLIENTS, FLEET_SLEEPS)
        timeline = FakeTimeline({1: ("fWB.7", "thor")})
        resolver = IdentityResolver(timeline, BASE_SEED, 1.0, M, fleet)
        first = resolver.of_slot(1)
        timeline.occupants[1] = ("fEB.99", "orin")
        second = resolver.of_slot(1)
        assert first == second
        assert first == OccupantIdentity("1", "xavier", fleet[1][1], 0)

    def test_closed_fleet_matches_the_stage3_to_6_fleet(self):
        """Per-tier sleep means of the fleet p_recur = 1 binds to: the
        closed-loop fleet's 14.37 / 1.78 / 1.07 s (Stage 7 prompt)."""
        fleet = load_fleet(FLEET_CLIENTS, FLEET_SLEEPS)
        by_tier: dict[str, list[float]] = {}
        for tier, sleep in fleet.values():
            by_tier.setdefault(tier, []).append(sleep)
        assert {t: len(v) for t, v in by_tier.items()} == {"xavier": 10, "orin": 7, "thor": 3}
        assert np.mean(by_tier["xavier"]) == pytest.approx(14.37, abs=0.01)
        assert np.mean(by_tier["orin"]) == pytest.approx(1.78, abs=0.01)
        assert np.mean(by_tier["thor"]) == pytest.approx(1.07, abs=0.01)

    def test_only_zero_and_one_are_defined(self):
        with pytest.raises(ValueError):
            IdentityResolver(FakeTimeline({}), BASE_SEED, 0.5, M)
        with pytest.raises(ValueError):
            IdentityResolver(FakeTimeline({}), BASE_SEED, 1.0, M, fleet=None)


class TestIdentitySeed:
    def test_slot_identity_reproduces_the_closed_loop_seed(self):
        for slot in (1, 7, 20):
            for rnd in (1, 250, 4000):
                assert identity_seed(BASE_SEED, str(slot), rnd) == \
                    client_seeded_trainer.derive_seed(BASE_SEED, slot, rnd)

    def test_vehicle_identities_get_their_own_streams(self):
        seeds = {identity_seed(BASE_SEED, v, 5) for v in ("fEB.1", "fEB.2", "fWB.1")}
        assert len(seeds) == 3


class TestPartitionFingerprint:
    def test_groups_by_label_weights_only(self):
        datasource = SimpleNamespace(targets=lambda: [0, 1, 2, 0, 1, 2])
        a = SimpleNamespace(sample_weights=[0.5, 0.3, 0.2, 0.5, 0.3, 0.2])
        b = SimpleNamespace(sample_weights=[0.5, 0.3, 0.2, 0.5, 0.3, 0.2])
        c = SimpleNamespace(sample_weights=[0.1, 0.3, 0.6, 0.1, 0.3, 0.6])
        assert partition_fingerprint(a, datasource) == partition_fingerprint(b, datasource)
        assert partition_fingerprint(a, datasource) != partition_fingerprint(c, datasource)
        assert partition_fingerprint(SimpleNamespace(), datasource) == "na"


# ---------------------------------------------------------------------------
# Sibling (c): the unexplored prior, against the REAL aggregation strategy
# ---------------------------------------------------------------------------

def real_polaris_server(slots: int = M):
    aggregation = polaris_server.FedAsyncPolarisAggregationStrategy(alpha=10.0)
    context = SimpleNamespace(total_clients=slots, state={})
    aggregation.setup(context)
    return SimpleNamespace(aggregation_strategy=UploadGate(aggregation)), aggregation, context


class TestResetAgainstRealClasses:
    def test_readds_to_the_real_unexplored_set(self):
        server, aggregation, context = real_polaris_server()
        aggregation.unexplored_clients.discard(4)
        context.state["polaris"]["local_gradient_bounds"][4] = 3.3
        reset_slot_state(context, 5, server)
        assert 4 in aggregation.unexplored_clients
        assert context.state["polaris"]["local_gradient_bounds"][4] == UNSEEN_GRADIENT_BOUND

    def test_new_identity_carries_the_current_proxy(self):
        server, aggregation, context = real_polaris_server()
        aggregation.unexplored_clients.discard(4)
        aggregation.last_proxy = 8.75
        reset_slot_state(context, 5, server)
        assert context.state["polaris"]["local_gradient_bounds"][4] == 8.75


# ---------------------------------------------------------------------------
# The selector hooks are neutral by default and honoured when set
# ---------------------------------------------------------------------------

def selection_context(training=None, wall=0.0):
    server = SimpleNamespace(
        training_clients=training or {}, clients_per_round=4, current_round=3,
        wall_time=wall,
    )
    state = {
        "polaris": {
            "local_gradient_bounds": np.ones(M),
            "local_stalenesses": np.arange(M, dtype=float),
        },
        "T_round_seconds": 2.0,
        "partition_sizes": {c: 2500 for c in range(1, M + 1)},
        "local_epochs": 1,
        "clients_per_round": 4,
    }
    return SimpleNamespace(server=server, state=state, current_round=3, total_clients=M)


class TestSelectorHooks:
    def test_default_path_exposes_the_tau_hat_used(self):
        strategy = selection.PolarisViClientSelectionStrategy()
        context = selection_context()
        strategy._total_clients = M
        strategy.select_clients([2, 5, 9], 1, context)
        assert strategy.busy_predict_seconds is None
        assert strategy.last_tau_hats == {2: 1.0, 5: 4.0, 9: 8.0}

    def test_busy_terms_use_the_hook_not_the_fleet_file(self, monkeypatch):
        calls = []

        def boom(*_a, **_k):
            raise AssertionError("busy terms read the fleet file (client_id)")

        monkeypatch.setattr(selection, "gpu_tops_predict_seconds", boom)
        strategy = selection.PolarisViClientSelectionStrategy(
            predict_staleness=lambda cid, ctx: 1.0, predict_staleness_unit="seconds"
        )
        strategy._total_clients = M
        strategy.busy_predict_seconds = lambda cid, ctx: calls.append(cid) or 15.0
        context = selection_context(training={3: {"start_time": 0.0}}, wall=5.0)
        strategy.select_clients([2, 5], 1, context)
        assert calls and set(calls) == {3}


# ---------------------------------------------------------------------------
# Sibling H3: the predictor follows the occupant, the busy pool the dispatch
# ---------------------------------------------------------------------------

class Harness(OpenPopulationMixin):
    """Just the mixin's predictor methods, over a fake timeline."""

    def __init__(self, resolver):
        self.resolver = resolver
        self._predictor_tier = {}
        self._dispatch_identity = {}


class TestPredictorFollowsTheOccupant:
    def test_slot_one_scored_as_its_thor_occupant_not_the_csv_xavier(self):
        """H3's regression test: the fleet file says slot 1 = xavier."""
        timeline = FakeTimeline({1: ("fWB.7", "thor")})
        harness = Harness(IdentityResolver(timeline, BASE_SEED, 0.0, M))
        got = harness._occupant_predict_seconds(1, None)
        assert got == pytest.approx(Harness.predict_seconds_for_tier("thor"))
        assert got != pytest.approx(gpu_tier_sleep.gpu_tops_predict_seconds(1, 2500, 1))
        assert harness._predictor_tier[1] == "thor"

    def test_tier_predictions_keep_the_15x_ratio(self):
        x = Harness.predict_seconds_for_tier("xavier")
        t = Harness.predict_seconds_for_tier("thor")
        assert x == pytest.approx(15.0)
        assert x / t == pytest.approx(15.0, rel=1e-6)

    def test_busy_term_uses_the_dispatched_identity_after_a_recycle(self):
        timeline = FakeTimeline({1: ("fWB.7", "xavier")})
        harness = Harness(IdentityResolver(timeline, BASE_SEED, 0.0, M))
        harness._dispatch_identity[1] = harness.resolver.of_slot(1)
        timeline.occupants[1] = ("fEB.3", "thor")  # slot recycled mid-training
        assert harness._dispatched_predict_seconds(1, None) == \
            pytest.approx(Harness.predict_seconds_for_tier("xavier"))
        assert harness._occupant_predict_seconds(1, None) == \
            pytest.approx(Harness.predict_seconds_for_tier("thor"))
