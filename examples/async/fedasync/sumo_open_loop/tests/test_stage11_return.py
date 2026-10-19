"""
Stage 11: partial recurrence (`p_return`) -- the identity policy and the state
that must follow the identity.  Uses the REAL trace timeline, the real slot
allocator, the real aggregation strategy (wrapped in UploadGate) and the real
Stage 10 gate, as the spec asks.
"""

from __future__ import annotations

import logging
import random
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

_saved = sys.argv
sys.argv = ["pytest", "-c", str(OPEN_LOOP / "configs" / "stage7b_preflight" / "gpu_open.toml")]
try:
    import fedasync_polaris_vi_server as polaris_server  # noqa: E402
    from plato.config import Config  # noqa: E402

    Config()
finally:
    sys.argv = _saved

from occupant_identity import IdentityResolver, load_fleet  # noqa: E402
from open_stage0 import OpenStage0Config, OpenStage0Gate  # noqa: E402
from stage7_population import UNSEEN_STALENESS, apply_recycle  # noqa: E402
from sumo_gates import UploadGate  # noqa: E402
from sumo_open_mobility import OpenCoverageTimeline  # noqa: E402
from sumo_open_server import OpenPopulationMixin  # noqa: E402
from sumo_position import PositionOracle  # noqa: E402

logging.disable(logging.INFO)
TRACE = OPEN_LOOP / "sumo_trace" / "e1_open_poisson_seed4126_range400" / "coverage_windows.csv"
FLEET = load_fleet(FEDASYNC / "sumo_close_loop" / "input" / "clients_n20r400.csv",
                   FEDASYNC / "sumo_close_loop" / "output" / "client_sleep_times_n20r400.csv")
PHYS = 4126
HORIZON = 2500.0


def replay(p_return, physics_seed=PHYS, horizon=HORIZON):
    """Real timeline + allocator + resolver, identities decided at allocation."""
    tl = OpenCoverageTimeline.from_csv(TRACE, slots=20, p_recur=0.0, base_seed=physics_seed)
    res = IdentityResolver(tl, physics_seed, 0.0, 20, FLEET, p_return=p_return)
    seq = []
    orig_assign = res.returning.assign

    def assign(slot, veh, tier, base_seed, slots):
        ident, kind, ex = orig_assign(slot, veh, tier, base_seed, slots)
        seq.append((slot, veh, ident.identity_id, kind, ex, set(res.returning.in_coverage)))
        return ident, kind, ex
    res.returning.assign = assign
    res.attach(tl.allocator)
    tl.advance_to(horizon)
    return tl, res, seq


class TestIdentityPolicy:
    def test_p0_is_the_stage10_open_sequence(self):
        tl, res, seq = replay(0.0)
        assert seq and all(kind == "transient" and ident == veh for _, veh, ident, kind, _, _ in seq)
        # every occupant's identity is exactly Stage 10's p_recur = 0 identity
        ref = IdentityResolver(OpenCoverageTimeline.from_csv(TRACE, 20, 0.0, PHYS), PHYS, 0.0, 20)
        ref.timeline.advance_to(HORIZON)
        for slot in tl.occupied_slots():
            assert res.of_slot(slot) == ref.of_slot(slot)

    def test_identity_rng_touches_no_other_stream(self):
        before = (random.getstate(), np.random.get_state()[1].copy())
        replay(0.5)
        after = (random.getstate(), np.random.get_state()[1].copy())
        assert before[0] == after[0] and (before[1] == after[1]).all()

    def test_p1_every_occupant_a_commuter_unless_exhausted(self):
        _, res, seq = replay(1.0)
        for _, _, ident, kind, exhausted, cov in seq:
            if kind == "transient":
                assert exhausted and len(cov) == 20      # only when all 20 are in coverage
            else:
                assert ident.startswith("C")

    def test_commuter_never_in_two_slots(self):
        tl, res, _ = replay(0.75)
        ids = [res.returning.slot_identity[s].identity_id for s in tl.occupied_slots()]
        commuters = [i for i in ids if i.startswith("C")]
        assert len(commuters) == len(set(commuters))

    def test_digest_is_a_function_of_physics_seed_and_p_only(self):
        a, b = replay(0.5)[1], replay(0.5)[1]
        assert a.returning.digest() == b.returning.digest()
        assert replay(0.5, physics_seed=4127)[1].returning.digest() != a.returning.digest()

    def test_u_sequence_is_shared_across_p(self):
        """Common random numbers: the transient/commuter decision at p=0.25 is
        a subset of p=0.5's until the first exhaustion."""
        s25, s50 = replay(0.25)[2], replay(0.5)[2]
        c25 = {i for i, x in enumerate(s25) if x[3] == "commuter"}
        c50 = {i for i, x in enumerate(s50) if x[3] == "commuter" or x[4]}
        assert c25 <= c50

    def test_realised_share_and_tier_mix(self):
        _, res, seq = replay(0.5, horizon=7000.0)
        r = res.returning
        assert abs(r.commuter_allocations / r.allocations - 0.5) < 0.02 + r.exhaustions / r.allocations
        tr = r.kind_tiers["transient"]
        n = sum(tr.values())
        assert abs(tr["xavier"] / n - 0.491) < 0.03

    def test_occupancy_stream_unchanged_by_the_policy(self):
        plain = OpenCoverageTimeline.from_csv(TRACE, 20, 0.0, PHYS)
        plain.advance_to(HORIZON)
        tl, _, _ = replay(0.75)
        assert tl.occupancy_digest() == plain.occupancy_digest()


# ---------------------------------------------------------------------------
# State follows the identity (server side, real aggregation strategy)
# ---------------------------------------------------------------------------

class Server(OpenPopulationMixin):
    def __init__(self, p_return):
        self.timeline = OpenCoverageTimeline.from_csv(TRACE, 20, 0.0, PHYS)
        self.resolver = IdentityResolver(self.timeline, PHYS, 0.0, 20, FLEET, p_return=p_return)
        agg = polaris_server.FedAsyncPolarisAggregationStrategy(alpha=1.0)
        self.context = SimpleNamespace(total_clients=20, state={})
        agg.setup(self.context)
        self.context.state["polaris"]["staleness_predictions"] = {}
        self.agg = agg
        self.aggregation_strategy = UploadGate(agg)
        self._observations = {}
        self.install_returning_population()
        self.state_resets = 0

    def advance(self, t):
        for event in self.timeline.advance_to(t):
            apply_recycle(self.context, event, self)


class TestStateFollowsIdentity:
    def test_commuter_state_restored_transient_reset(self):
        srv = Server(1.0)
        pol = srv.context.state["polaris"]
        seen_exit = {}
        # Walk the trace; whenever a commuter occupies a slot, stamp a distinctive
        # state into that slot; on its next re-entry the stamp must come back.
        t = 0.0
        checked = 0
        while t < 3000 and checked < 15:
            t += 1.0
            before = dict(srv.returning.slot_identity)
            srv.advance(t)
            for slot, ident in srv.returning.slot_identity.items():
                idx = slot - 1
                if before.get(slot) is ident:
                    continue
                if ident.identity_id in seen_exit:  # a re-entry: state must be carried
                    g, tau, unexp = seen_exit[ident.identity_id]
                    assert pol["local_gradient_bounds"][idx] == pytest.approx(g)
                    assert pol["local_stalenesses"][idx] == pytest.approx(tau)
                    assert (idx in srv.agg.unexplored_clients) is unexp
                    checked += 1
                # stamp a fresh, identity-specific state (as if it had trained)
                g, tau = 1.0 + int(ident.identity_id[1:]) / 10, float(int(ident.identity_id[1:]) % 5)
                pol["local_gradient_bounds"][idx] = g
                pol["local_stalenesses"][idx] = tau
                srv.agg.unexplored_clients.discard(idx)
                seen_exit[ident.identity_id] = (g, tau, False)
        assert checked >= 5 and srv.commuter_reentries >= checked

    def test_transient_gets_neutral_prior(self):
        srv = Server(0.0)
        pol = srv.context.state["polaris"]
        srv.advance(200.0)
        pol["local_stalenesses"][:] = 7.0
        srv.agg.unexplored_clients.clear()
        srv.advance(900.0)
        # every slot recycled since then holds a transient on the prior
        for slot, kind in srv.returning.slot_kind.items():
            assert kind == "transient"
        recycled = [e.slot_id for e in srv.timeline.allocator.recycles if e.state_reset]
        assert recycled
        for slot in set(recycled[-5:]):
            if slot in srv.returning.slot_identity:
                assert pol["local_stalenesses"][slot - 1] == UNSEEN_STALENESS
                assert (slot - 1) in srv.agg.unexplored_clients


class TestFilterReadsTheOccupant:
    def test_commuter_tier_and_trace_position(self):
        srv = Server(1.0)
        srv.advance(300.0)
        slot, ident = next((s, i) for s, i in srv.returning.slot_identity.items()
                           if i.tier != srv.timeline.tier_of_vehicle[srv.timeline.vehicle_in_slot(s)])
        veh = srv.timeline.vehicle_in_slot(slot)
        srv.sumo_time = lambda: 300.0
        srv.current_round = 1
        pos = PositionOracle({veh: [(299.0, 600.0, 1), (300.0, 600.0, 1), (301.0, 600.0, 1)]})
        cfg = OpenStage0Config("tier_spec", 400.0, 19.444444, 600.0, "x")
        gate = OpenStage0Gate(SimpleNamespace(select_clients=lambda p, c, x: p), srv, cfg, pos)
        t_hat, basis = gate.t_hat(srv.resolver.of_slot(slot))
        assert t_hat == pytest.approx(Server.predict_seconds_for_tier(ident.tier))
        assert srv.resolver.of_slot(slot) is ident
