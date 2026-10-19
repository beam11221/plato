"""Step 1 / Step 8 test 2: nothing on the Wen path reads an anchor parameter.

Three layers:
  * the entry point refuses any [gpu_tier] key except t_round_model, and refuses
    [server] staleness_predictor;
  * a guarded [gpu_tier] section that RAISES when anchor_tier / anchor_sleep /
    tiers_csv / clients_csv / w_model_flops is read is put behind PLATO's Config,
    and the whole selection path (T_round kernel, the Polaris selector with the
    T_hat_total predictor, Stage 0, the estimators) is run through it;
  * no wen_env module calls the anchored predictor functions of gpu_tier_sleep.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

import numpy as np
import pytest

import gpu_tier_sleep
import wen_constants as K
import wen_estimators
from conftest import WEN
from wen_reported import ReportedVehicle
from wen_selection import WenStage0Gate

from plato.servers.strategies.base import ServerContext

FORBIDDEN = {"anchor_tier", "anchor_sleep", "tiers_csv", "clients_csv",
             "w_model_flops", "oracle_sleep_csv"}
BITS = 11_173_962 * 32


class GuardedSection(dict):
    """A [gpu_tier] section that fails the test if an anchor key is touched."""

    def __getattr__(self, name):
        if name in FORBIDDEN:
            raise AssertionError(f"anchor/tier-file key {name!r} was read")
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


@pytest.fixture
def guarded_config(monkeypatch):
    section = GuardedSection(t_round_model="queue")
    fake = SimpleNamespace(
        config=SimpleNamespace(gpu_tier=section),
        data=SimpleNamespace(partition_size=1667),
        trainer=SimpleNamespace(epochs=1),
    )
    monkeypatch.setattr("plato.config.Config", lambda *a, **k: fake)
    for name in ("_t_round_model", "_predictor_scale", "_w_model_flops",
                 "_comm_latency", "_comm_in_predictor", "_oracle_sleeps", "_computer"):
        monkeypatch.setattr(gpu_tier_sleep, name, None)
    return section


def rv(cid, x, tier):
    return ReportedVehicle(cid, 5.0, x, 25.0, tier, K.TIER_TFLOPS_FP16[tier],
                           K.TIER_CPU_GHZ[tier], K.partition_size(cid))


class FakeServer:
    def __init__(self):
        self.training_clients = {1: {"start_time": 1.0}, 16: {"start_time": 2.0}}
        self.reported_clients = []
        self.clients_per_round = 3
        self.wall_time = 5.0
        self.current_round = 7
        tiers = ["xavier"] * 15 + ["orin"] * 10 + ["thor"] * 5
        self.reports = {c: rv(c, 300.0 + 40 * c, tiers[c - 1]) for c in range(1, 31)}
        self.est = wen_estimators.GpuEstimator(K.FLOPS_FWD_PER_IMAGE, BITS)

    def sumo_time(self):
        return self.wall_time

    def wen_report(self, cid, t):
        return self.reports[cid]


def test_selection_path_reads_no_anchor(guarded_config):
    import fedasync_polaris_vi_selection as sel

    server = FakeServer()
    context = ServerContext()
    context.server = server
    context.total_clients = 30
    context.state.update({
        "polaris": {"local_gradient_bounds": np.full(30, 0.5),
                    "local_stalenesses": np.full(30, 0.01)},
        "partition_sizes": {c: K.partition_size(c) for c in range(1, 31)},
        "local_epochs": 1, "clients_per_round": 3, "T_round_seconds": 3.0,
    })
    selector = sel.PolarisViClientSelectionStrategy(
        predict_staleness=lambda cid, ctx: server.est.estimate(server.reports[cid]).total,
        predict_staleness_unit="seconds",
    )
    selector.busy_predict_seconds = lambda cid, ctx: 9.0
    selector.setup(context)
    gate = WenStage0Gate(selector, server.est)

    pool = [c for c in range(1, 31) if c not in server.training_clients]
    chosen = gate.select_clients(pool, 1, context)
    assert len(chosen) == 1
    assert gpu_tier_sleep.t_round_model() == "queue"
    assert selector.last_tau_hats                     # the T_hat path really ran


def test_entry_point_refuses_anchor_keys():
    import wen_config as wen_fedasync

    def cfg(gpu_tier, **server):
        return SimpleNamespace(
            wen_env=SimpleNamespace(arm="gpu", scenario="S1"),
            gpu_tier=gpu_tier,
            server=SimpleNamespace(polaris_alpha=1.0, **server),
            clients=SimpleNamespace(total_clients=30, per_round=3),
            trainer=SimpleNamespace(epochs=1),
        )

    for bad in ({"t_round_model": "queue", "anchor_tier": "xavier"},
                {"t_round_model": "queue", "anchor_sleep": 15},
                {"t_round_model": "queue", "tiers_csv": "x.csv"}):
        with pytest.raises(ValueError, match="gpu_tier"):
            wen_fedasync.validate_config(cfg(dict(bad)))
    with pytest.raises(ValueError, match="staleness_predictor"):
        wen_fedasync.validate_config(cfg({"t_round_model": "queue"},
                                         staleness_predictor="gpu_tops"))


ANCHORED_FUNCTIONS = {"predictor_scale", "_predicted_seconds", "gpu_tops_predict_seconds",
                      "compute_pool_T_round_seconds", "compute_pool_mean_T_seconds",
                      "_client_seconds", "sleep_time_seconds", "tier_for", "get_computer"}


@pytest.mark.parametrize("path", sorted(WEN.glob("wen_*.py")), ids=lambda p: p.name)
def test_no_wen_module_calls_the_anchored_predictor(path):
    tree = ast.parse(path.read_text())
    called = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    called |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not called & ANCHORED_FUNCTIONS, called & ANCHORED_FUNCTIONS
