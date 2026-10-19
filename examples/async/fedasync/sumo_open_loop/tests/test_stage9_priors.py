"""
Stage 9: the two never-seen priors, against the REAL selector and aggregation.

    g_prior_alpha        == [server] polaris_alpha (pre-existing key, default 10)
                            proxy G_i = alpha x mean norm of the latest batch
    observed_unseen_tau  == [server] observed_unseen_tau, new:
                            "const_0.01" (default, Stages 3-8) or
                            "fleet_running_mean" (mean delivered staleness so
                            far; C - 1 before the first report)

The defaults must leave every existing config exactly as it was.
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

CONFIG = OPEN_LOOP / "configs" / "stage7b_preflight" / "observed_open.toml"
_saved = sys.argv
sys.argv = ["pytest", "-c", str(CONFIG)]
try:
    import fedasync_polaris_vi_selection as selection  # noqa: E402
    import fedasync_polaris_vi_server as polaris_server  # noqa: E402
    from plato.config import Config  # noqa: E402

    Config()
finally:
    sys.argv = _saved

import sumo_open_server  # noqa: E402
from occupant_identity import IdentityResolver  # noqa: E402
from sumo_open_server import OpenPopulationMixin  # noqa: E402


class FakeTimeline:
    def __init__(self, occupants):
        self.occupants = dict(occupants)

    @property
    def tier_of_vehicle(self):
        return {v: t for v, t in self.occupants.values()}

    def vehicle_in_slot(self, slot):
        e = self.occupants.get(slot)
        return e[0] if e else None


class Harness(OpenPopulationMixin):
    def __init__(self, selector, occupants):
        self.client_selection_strategy = SimpleNamespace(inner=selector)
        self.resolver = IdentityResolver(FakeTimeline(occupants), 4126, 0.0, 20)
        self.clients_per_round = 4
        self._observations = {}
        self._tau_sum = 0.0
        self._tau_count = 0


def with_rule(monkeypatch, rule):
    server = SimpleNamespace() if rule is None else SimpleNamespace(observed_unseen_tau=rule)
    monkeypatch.setattr(sumo_open_server, "Config", lambda: SimpleNamespace(server=server))


def ctx_with_taus(values):
    return SimpleNamespace(state={"polaris": {"local_stalenesses": np.array(values, float)}})


class TestUnseenTauRule:
    def test_default_leaves_the_observed_predictor_untouched(self, monkeypatch):
        with_rule(monkeypatch, None)
        selector = selection.PolarisViClientSelectionStrategy()
        h = Harness(selector, {1: ("fEB.1", "orin")})
        h.install_unseen_tau_rule()
        assert selector._predict_staleness is selection._observed_staleness
        assert h.unseen_tau_rule == "const_0.01"

    def test_explicit_default_is_the_same(self, monkeypatch):
        with_rule(monkeypatch, "const_0.01")
        selector = selection.PolarisViClientSelectionStrategy()
        Harness(selector, {1: ("fEB.1", "orin")}).install_unseen_tau_rule()
        assert selector._predict_staleness is selection._observed_staleness

    def test_running_mean_for_unseen_observed_for_seen(self, monkeypatch):
        with_rule(monkeypatch, "fleet_running_mean")
        selector = selection.PolarisViClientSelectionStrategy()
        h = Harness(selector, {1: ("fEB.1", "orin"), 2: ("fWB.2", "thor")})
        h.install_unseen_tau_rule()
        ctx = ctx_with_taus([0.01, 4.0] + [0.01] * 18)

        assert selector._predict_staleness(1, ctx) == 3.0      # C - 1 before any report
        h.record_delivered_staleness([SimpleNamespace(staleness=1), SimpleNamespace(staleness=2)])
        assert selector._predict_staleness(1, ctx) == 1.5      # running mean
        h._observations["fWB.2"] = 1                           # slot 2's vehicle is seen
        assert selector._predict_staleness(2, ctx) == 4.0      # its own observed tau

    def test_proactive_arms_ignore_the_rule(self, monkeypatch):
        with_rule(monkeypatch, "fleet_running_mean")
        predictor = lambda cid, ctx: 15.0  # noqa: E731
        selector = selection.PolarisViClientSelectionStrategy(
            predict_staleness=predictor, predict_staleness_unit="seconds")
        Harness(selector, {1: ("fEB.1", "orin")}).install_unseen_tau_rule()
        assert selector._predict_staleness is predictor

    def test_unknown_rule_raises(self, monkeypatch):
        with_rule(monkeypatch, "optimistic")
        h = Harness(selection.PolarisViClientSelectionStrategy(), {})
        with pytest.raises(ValueError):
            h.install_unseen_tau_rule()


def aggregate_once(alpha):
    """One real _update_gradient_bounds call: clients 1 and 2 report, 3-4 unexplored."""
    agg = polaris_server.FedAsyncPolarisAggregationStrategy(alpha=alpha)
    context = SimpleNamespace(total_clients=4, state={})
    agg.setup(context)
    base = {"conv1.weight": np.zeros(4)}
    updates = [SimpleNamespace(client_id=1), SimpleNamespace(client_id=2)]
    weights = [{"conv1.weight": np.array([3.0, 4.0, 0, 0])},   # norm 5
               {"conv1.weight": np.array([0, 0, 6.0, 8.0])}]   # norm 10
    agg._update_gradient_bounds(updates, base, weights, context)
    return agg, context.state["polaris"]["local_gradient_bounds"]


class TestGPriorAlpha:
    def test_default_alpha_is_ten(self):
        assert polaris_server.FedAsyncPolarisAggregationStrategy().alpha == 10.0

    def test_alpha_ten_proxy_is_ten_times_batch_mean(self):
        agg, g = aggregate_once(10.0)
        assert agg.last_proxy_basis == pytest.approx(7.5)
        assert agg.last_proxy == pytest.approx(75.0)
        assert list(g) == pytest.approx([5.0, 10.0, 75.0, 75.0])

    def test_alpha_one_proxy_is_the_batch_mean(self):
        agg, g = aggregate_once(1.0)
        assert agg.last_proxy == pytest.approx(7.5)
        assert list(g) == pytest.approx([5.0, 10.0, 7.5, 7.5])
