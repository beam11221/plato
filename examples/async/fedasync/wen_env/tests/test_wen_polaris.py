"""Prompt J Step 3: the `polaris` host (P4 solver, sampling, runs)."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

import wen_constants as K
import wen_polaris
from conftest import RUNS_DIR
from wen_polaris import PolarisSelection, solve_p4

DETERMINISTIC = ("cycle", "round", "client_id", "t_dispatch", "duration",
                 "t_finish_clean", "accepted", "score")


def _rows(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _objective(q, G, tau, p, A=1.0):
    return float(np.sum(p * p * G * G / q) + A * np.sum(p * p * q * tau * G))


def _pool(rng):
    n = int(rng.integers(20, 31))
    return (rng.uniform(0.1, 5.0, n), rng.uniform(0.01, 20.0, n), np.full(n, 1 / 30))


# -- 1. bisection == a general-purpose solve of P4 ------------------------------

def test_bisection_matches_slsqp_on_200_pools():
    from scipy.optimize import minimize

    rng = np.random.default_rng(20261021)
    for _ in range(200):
        G, tau, p = _pool(rng)
        q, _mu = solve_p4(G, tau, p)
        n = len(G)
        res = minimize(lambda x: _objective(x, G, tau, p), np.full(n, 1 / n),
                       jac=lambda x: -p * p * G * G / x**2 + p * p * tau * G,
                       method="SLSQP", bounds=[(1e-12, 1.0)] * n,
                       constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0,
                                     "jac": lambda x: np.ones_like(x)}],
                       options={"ftol": 1e-16, "maxiter": 2000})
        assert res.success, res.message
        assert np.max(np.abs(q - res.x)) < 1e-6
        ours, ref = _objective(q, G, tau, p), _objective(res.x, G, tau, p)
        assert ours <= ref * (1 + 1e-9)


# -- 2. sum q = 1, q >= 0 -------------------------------------------------------

def test_q_is_a_distribution_on_random_pools():
    rng = np.random.default_rng(7)
    for _ in range(200):
        q, _ = solve_p4(*_pool(rng))
        assert abs(q.sum() - 1.0) < 1e-12 and (q >= 0).all()


# -- 3. monotonicity, 4. uniform, edge cases -----------------------------------

def test_raising_g_raises_q_and_raising_tau_lowers_q():
    rng = np.random.default_rng(3)
    for _ in range(50):
        G, tau, p = _pool(rng)
        q0, _ = solve_p4(G, tau, p)
        j = int(rng.integers(len(G)))
        G2, tau2 = G.copy(), tau.copy()
        G2[j] *= 1.5
        tau2[j] *= 1.5
        assert solve_p4(G2, tau, p)[0][j] > q0[j]
        assert solve_p4(G, tau2, p)[0][j] < q0[j]


def test_identical_states_give_uniform_q():
    n = 27
    q, mu = solve_p4(np.full(n, 0.5), np.full(n, 0.01), np.full(n, 1 / 30))
    assert np.allclose(q, 1 / n, rtol=0, atol=1e-15)
    f1, f2 = (0.5 / 30) ** 2, 0.01 * 0.5 / 900
    assert abs(np.sqrt(f1 / (f2 + mu)) - 1 / n) < 1e-15


def test_zero_g_is_never_sampled():
    G = np.array([0.0, 1.0, 2.0, 3.0])
    q, _ = solve_p4(G, np.ones(4), np.full(4, 0.25))
    assert q[0] == 0.0 and abs(q.sum() - 1) < 1e-12
    with pytest.raises(ValueError):
        solve_p4(np.zeros(3), np.ones(3), np.full(3, 1 / 3))


# -- the host: tau bookkeeping, sampling, the side log --------------------------

class _Server:
    def __init__(self):
        self.training_clients, self.current_round = {}, 1

    def sumo_time(self):
        return 0.0


def _context(G):
    return SimpleNamespace(server=_Server(),
                           state={"polaris": {"local_gradient_bounds": np.asarray(G)}})


def test_tau_is_staleness_plus_offset_and_p_is_fleet_static(tmp_path):
    sel = PolarisSelection(seed=4126, out_dir=tmp_path)
    assert np.allclose(sel.tau, K.POLARIS_INITIAL_TAU)
    sizes = np.array([K.partition_size(c) for c in range(1, 31)], dtype=float)
    assert np.array_equal(sel.p, sizes / sizes.sum())
    ctx = _context(np.full(30, 0.5))
    u = SimpleNamespace(client_id=4, staleness=2, report=SimpleNamespace())
    sel.on_reports_received([u], ctx)
    assert sel.tau[3] == 2 + K.POLARIS_TAU_OFFSET


def test_selection_logs_a_distribution_and_is_seeded(tmp_path):
    G = np.linspace(0.5, 3.0, 30)
    runs = []
    for d in ("a", "b"):
        sel = PolarisSelection(seed=4126, out_dir=tmp_path / d)
        ctx = _context(G)
        ctx.server.training_clients = {5: {}}
        picks = sel.select_clients(list(range(1, 31)), 3, ctx)
        assert len(set(picks)) == 3 and 5 not in picks
        assert set(sel.last_q) == set(range(1, 31)) - {5}
        runs.append(picks)
        row = _rows(tmp_path / d / "polaris_decisions.csv")[0]
        q = np.array(json.loads(row["q"]))
        assert abs(q.sum() - 1) < 1e-12 and (q >= 0).all()
        assert json.loads(row["picks"]) == picks
    assert runs[0] == runs[1]


# -- 5-7: run-based checks (skipped until the runs exist) ----------------------

DET = (RUNS_DIR / "S1_polaris_s4126_det1", RUNS_DIR / "S1_polaris_s4126_det2")


def test_logged_q_is_a_distribution_in_every_call():
    run = DET[0]
    if not (run / "polaris_decisions.csv").exists():
        pytest.skip("polaris 100-round run not present")
    rows = _rows(run / "polaris_decisions.csv")
    assert len(rows) >= 100                     # one call per dispatch cycle
    for r in rows:
        q = np.array(json.loads(r["q"]))
        assert abs(q.sum() - 1) < 1e-12 and (q >= 0).all()


def test_determinism_two_100_round_polaris_runs():
    a, b = DET
    if not ((a / "dispatch_log.csv").exists() and (b / "dispatch_log.csv").exists()):
        pytest.skip("polaris determinism runs not present")
    keys = [[tuple(r[c] for c in DETERMINISTIC) for r in _rows(d / "dispatch_log.csv")]
            for d in (a, b)]
    assert len(keys[0]) > 100 and keys[0] == keys[1]
    assert _rows(a / "polaris_decisions.csv") == _rows(b / "polaris_decisions.csv")


def test_online_clock_equals_replay_on_polaris_run():
    run = DET[0]
    if not (run / "dispatch_log.csv").exists():
        pytest.skip("polaris 100-round run not present")
    import wen_clock_replay

    assert wen_clock_replay.replay(run)["max_abs_diff_s"] == 0.0


def test_polaris_constants_are_the_plato_defaults():
    assert (K.POLARIS_ALPHA, K.POLARIS_A, K.POLARIS_TAU_OFFSET,
            K.POLARIS_INITIAL_TAU) == (10.0, 1.0, 0.1, 0.01)
    assert wen_polaris.A_CONST == K.POLARIS_A


# -- Prompt L (S2): Stage 0 in front of Polaris, lost uploads, runs -------------

def test_polaris_behind_stage0_sees_only_passed_vehicles(tmp_path):
    from conftest import BUNDLE_DIR, FLEET_DIR, TIERS_CSV
    from wen_estimators import GpuEstimator
    from wen_fleet import Fleet
    from wen_reported import ReportedView
    from wen_selection import WenStage0Gate
    from wen_trace import TraceIndex, load_speeds, veh_id

    speeds = load_speeds(BUNDLE_DIR)
    trace = TraceIndex.from_bundle(BUNDLE_DIR, speeds)
    fleet = Fleet.load(FLEET_DIR, TIERS_CSV)
    view = ReportedView(position=lambda c, t: trace.x_at(veh_id(c), t),
                        speed=lambda c: speeds[veh_id(c)],
                        spec=lambda c: (fleet.spec(c).tier_id, fleet.spec(c).tflops_fp16,
                                        fleet.spec(c).cpu_ghz),
                        n_samples=K.partition_size)
    est = GpuEstimator(K.FLOPS_FWD_PER_IMAGE, 357_566_784)
    lo, hi = K.S2_COVERAGE_M
    checked = 0
    for t in np.arange(100.0, 3000.0, 97.0):
        eligible = [c for c in range(1, 31) if lo <= view.report(c, t).x <= hi]
        server = SimpleNamespace(training_clients={}, current_round=int(t),
                                 sumo_time=lambda t=t: t,
                                 wen_report=lambda c, tt: view.report(c, tt),
                                 wen_is_eligible=lambda c, tt, e=set(eligible): c in e)
        host = PolarisSelection(seed=4126, out_dir=tmp_path / f"t{int(t)}")
        gate = WenStage0Gate(host, est, scenario="S2")
        ctx = SimpleNamespace(server=server, state={"polaris": {
            "local_gradient_bounds": np.linspace(0.5, 3.0, 30)}})
        gate.select_clients(eligible, 1, ctx)
        passed = sorted(c for c, ok in gate.last_pass.items() if ok)
        row = _rows(tmp_path / f"t{int(t)}" / "polaris_decisions.csv")[0]
        pool = json.loads(row["pool"])
        assert pool == (sorted(eligible) if gate.last_stats["fallback"] else passed)
        q = np.array(json.loads(row["q"]))
        assert abs(q.sum() - 1) < 1e-12
        checked += int(len(passed) < len(eligible))
    assert checked > 0                       # some calls actually rejected vehicles


def test_lost_upload_changes_neither_g_nor_tau(tmp_path):
    import asyncio

    from sumo_gates import UPLOADED_KEY, UploadGate

    class _Recorder:
        seen = None

        async def aggregate_weights(self, updates, baseline, weights, context):
            _Recorder.seen = [u.client_id for u in updates]
            return baseline

    bounds = np.full(30, 0.5)
    ctx = SimpleNamespace(server=_Server(), state={
        "polaris": {"local_gradient_bounds": bounds}, UPLOADED_KEY: {3: False, 4: True}})
    lost = SimpleNamespace(client_id=3, staleness=2, report=SimpleNamespace())
    kept = SimpleNamespace(client_id=4, staleness=1, report=SimpleNamespace())

    sel = PolarisSelection(seed=4126, out_dir=tmp_path)
    sel.on_reports_received([lost, kept], ctx)
    assert sel.tau[2] == K.POLARIS_INITIAL_TAU               # lost: unchanged
    assert sel.tau[3] == 1 + K.POLARIS_TAU_OFFSET            # merged: updated

    gate = UploadGate(_Recorder())
    asyncio.run(gate.aggregate_weights([lost, kept], {}, [{}, {}], ctx))
    assert _Recorder.seen == [4]          # the G tracker never sees the lost update
    out = asyncio.run(gate.aggregate_weights([lost], {"w": 1}, [{}], ctx))
    assert out == {"w": 1} and np.all(bounds == 0.5)


S2_DET = (RUNS_DIR / "S2_polaris_s0_s4126_det1", RUNS_DIR / "S2_polaris_s0_s4126_det2")


def test_polaris_s0_plugin_off_reproduces_polaris_s2():
    a, b = RUNS_DIR / "S2_polaris_s0_pluginoff_s4126", RUNS_DIR / "S2_polaris_s4126_det1"
    if not ((a / "dispatch_log.csv").exists() and (b / "dispatch_log.csv").exists()):
        pytest.skip("S2 polaris plug-in-off runs not present")
    keys = [[tuple(r[c] for c in DETERMINISTIC) for r in _rows(d / "dispatch_log.csv")
             if int(r["round"]) <= 100] for d in (a, b)]
    assert len(keys[0]) >= 100 and keys[0] == keys[1]


def test_determinism_two_100_round_polaris_s0_runs():
    a, b = S2_DET
    if not ((a / "dispatch_log.csv").exists() and (b / "dispatch_log.csv").exists()):
        pytest.skip("S2 polaris_s0 determinism runs not present")
    keys = [[tuple(r[c] for c in DETERMINISTIC) for r in _rows(d / "dispatch_log.csv")]
            for d in (a, b)]
    assert len(keys[0]) >= 100 and keys[0] == keys[1]
    for f in ("polaris_decisions.csv", "stage0_log.csv"):
        assert _rows(a / f) == _rows(b / f)


@pytest.mark.parametrize("run", ["S2_polaris_s4126_det1", "S2_polaris_s0_s4126_det1"])
def test_s2_logged_q_is_a_distribution_over_the_post_filter_pool(run):
    d = RUNS_DIR / run
    if not (d / "polaris_decisions.csv").exists():
        pytest.skip(f"{run} not present")
    for r in _rows(d / "polaris_decisions.csv"):
        q, pool = np.array(json.loads(r["q"])), json.loads(r["pool"])
        assert len(q) == len(pool) and abs(q.sum() - 1) < 1e-12 and (q >= 0).all()
