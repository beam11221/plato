"""Step 8 tests 4, 5, 6, 7, plus estimator and partition checks."""

from __future__ import annotations

import ast
import dataclasses

import numpy as np
import pytest

import wen_constants as K
import wen_estimators
import wen_partition
from conftest import BUNDLE_DIR, FLEET_DIR, TIERS_CSV, WEN
from wen_fleet import Fleet
from wen_reported import ReportedVehicle, ReportedView
from wen_selection import stage0_feasible
from wen_trace import TraceIndex, _Track, load_speeds, veh_id
from wen_truth import GroundTruth

BITS = 11_173_962 * 32


@pytest.fixture(scope="module")
def speeds():
    return load_speeds(BUNDLE_DIR)


@pytest.fixture(scope="module")
def trace(speeds):
    return TraceIndex.from_bundle(BUNDLE_DIR, speeds)


@pytest.fixture(scope="module")
def fleet():
    return Fleet.load(FLEET_DIR, TIERS_CSV)


def make_view(trace, speeds, fleet):
    def spec(cid):
        s = fleet.spec(cid)
        return s.tier_id, s.tflops_fp16, s.cpu_ghz
    return ReportedView(lambda c, t: trace.x_at(veh_id(c), t),
                        lambda c: speeds[veh_id(c)], spec, K.partition_size)


# --- Test 4: coverage policy ---------------------------------------------------

def straight_trace(x0: float, speed: float, seconds: int = 400) -> TraceIndex:
    """One synthetic vehicle (veh00) moving at constant speed, no wrap."""
    t = np.arange(seconds + 1, dtype=float)
    x = x0 + speed * t
    tracks = {f"veh{i:02d}": _Track(t0=0.0, x=x % 2000, lap=(x // 2000).astype(int),
                                    speed=speed) for i in range(30)}
    return TraceIndex(tracks)


class _FixedFleet:
    """Fleet stand-in: every client is a Xavier with no jitter."""

    def __init__(self, real):
        self._real = real

    def spec(self, cid):
        return self._real.spec(1)

    def tier(self, cid):
        return "xavier"

    def veh_id(self, cid):
        return veh_id(cid)

    def jitter(self, cid):
        return 1.0


def outcome_for(fleet, x0, speed, scenario, chi_one=True):
    truth = GroundTruth(_FixedFleet(fleet), straight_trace(x0, speed),
                        K.FLOPS_FWD_PER_IMAGE, BITS, seed=1, scenario=scenario)
    if chi_one:
        truth.chi2 = lambda cid, rnd: 1.0
    return truth.dispatch(1, 0.0, 1)


def test_s1_accepts_every_upload(fleet):
    for x0 in (0.0, 300.0, 1590.0, 1999.0):
        assert outcome_for(fleet, x0, 30.0, "S1").accepted


def test_s2_rejects_an_exit_during_training(fleet):
    # 2.78 s of training at 30 m/s covers 83 m: start at 1550 -> leaves before upload.
    o = outcome_for(fleet, 1550.0, 30.0, "S2")
    assert o.x_finish > 1600 and not o.accepted


def test_s2_rejects_an_exit_during_upload(fleet):
    # Training ends inside (x ~ 1583) but the ~9.5 s upload carries it past 1600.
    o = outcome_for(fleet, 1500.0, 30.0, "S2")
    assert 400 <= o.x_finish <= 1600 and not o.accepted


def test_s2_accepts_a_vehicle_that_stays_inside(fleet):
    o = outcome_for(fleet, 800.0, 20.0, "S2")
    assert o.accepted and o.x_finish + 20.0 * o.T_up_true < 1600


def test_upload_uses_distance_at_training_finish(fleet):
    o = outcome_for(fleet, 800.0, 20.0, "S1")
    expected = BITS / (K.BANDWIDTH_HZ * np.log2(
        1 + o.d_finish ** -3 * K.TX_POWER_W / K.NOISE_W))
    assert o.T_up_true == pytest.approx(expected, rel=1e-12)
    assert o.x_finish == pytest.approx(800.0 + 20.0 * o.T_train_true, abs=1e-6)


# --- Test 5: Stage 0 with the oracle never admits a coverage failure ------------

def test_stage0_oracle_admits_no_failure_without_fading(trace, speeds, fleet):
    view = make_view(trace, speeds, fleet)
    truth = GroundTruth(fleet, trace, K.FLOPS_FWD_PER_IMAGE, BITS, seed=4126,
                        scenario="S2")
    oracle = wen_estimators.build("gpu_oracle", K.FLOPS_FWD_PER_IMAGE, BITS, [],
                                  true_train_seconds=truth.t_train_true)
    unit = GroundTruth(fleet, trace, K.FLOPS_FWD_PER_IMAGE, BITS, seed=4126,
                       scenario="S2")
    unit.chi2 = lambda cid, rnd: 1.0
    admitted = failed_unit = failed_fading = 0
    for t in np.arange(0.0, 6000.0, 7.3):
        for cid in fleet.client_ids:
            rv = view.report(cid, float(t))
            if not 400 <= rv.x <= 1600:
                continue
            if not stage0_feasible(rv, oracle.estimate(rv))[0]:
                continue
            admitted += 1
            failed_unit += not unit.dispatch(cid, float(t), int(t)).accepted
            failed_fading += not truth.dispatch(cid, float(t), int(t)).accepted
    assert admitted > 1000
    assert failed_unit == 0
    # Fading may still push a few over the boundary; counted, not forbidden.
    assert failed_fading / admitted < 0.05


# --- Test 6: determinism of the ground truth ------------------------------------

def test_same_seed_same_outcomes(trace, fleet):
    a = GroundTruth(fleet, trace, K.FLOPS_FWD_PER_IMAGE, BITS, 4126, "S2")
    b = GroundTruth(fleet, trace, K.FLOPS_FWD_PER_IMAGE, BITS, 4126, "S2")
    calls = [(c, 3.7 * c, r) for r in range(1, 6) for c in (1, 16, 26)]
    assert [a.dispatch(*x) for x in calls] == [b.dispatch(*x) for x in calls]
    c = GroundTruth(fleet, trace, K.FLOPS_FWD_PER_IMAGE, BITS, 4127, "S2")
    assert [a.chi2(1, r) for r in range(9, 12)] != [c.chi2(1, r) for r in range(9, 12)]


def test_repeat_dispatch_in_one_round_gets_a_fresh_draw(trace, fleet):
    truth = GroundTruth(fleet, trace, K.FLOPS_FWD_PER_IMAGE, BITS, 4126, "S2")
    assert truth.chi2(5, 10) != truth.chi2(5, 10)


# --- Test 7: selectors cannot reach ground truth -------------------------------

GROUND_TRUTH_MODULES = {"wen_truth", "wen_fleet", "wen_trace"}
SELECTOR_SIDE = ("wen_estimators.py", "wen_selection.py", "wen_reported.py", "wen_faithful.py",
                 "wen_formulas.py", "wen_constants.py")


@pytest.mark.parametrize("name", SELECTOR_SIDE)
def test_selector_side_does_not_import_ground_truth(name):
    tree = ast.parse((WEN / name).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & GROUND_TRUTH_MODULES, f"{name} imports {imported & GROUND_TRUTH_MODULES}"


def test_reported_record_has_no_ground_truth_field():
    fields = {f.name for f in dataclasses.fields(ReportedVehicle)}
    assert fields == {"client_id", "t", "x", "speed", "tier", "tflops_fp16",
                      "cpu_ghz", "n_samples"}


def test_polaris_selector_does_not_import_ground_truth():
    import fedasync_polaris_vi_selection as sel
    text = open(sel.__file__).read()
    assert not any(m in text for m in GROUND_TRUTH_MODULES)


# --- Estimators -----------------------------------------------------------------

def rv_at(x, tier="xavier", speed=25.0, n=1667):
    return ReportedVehicle(1, 0.0, x, speed, tier, K.TIER_TFLOPS_FP16[tier],
                           K.TIER_CPU_GHZ[tier], n)


def test_gpu_estimate_total_is_train_plus_upload():
    est = wen_estimators.GpuEstimator(K.FLOPS_FWD_PER_IMAGE, BITS).estimate(rv_at(500.0))
    assert est.T_train == pytest.approx(2.7838, abs=1e-4)
    assert est.x_finish == pytest.approx(500.0 + 25.0 * est.T_train)
    assert est.total == est.T_train + est.T_up
    assert est.E == pytest.approx(40.0 * est.T_train + 0.2 * est.T_up)


def test_estimated_finish_position_wraps():
    est = wen_estimators.CpuEstimator(K.FLOPS_FWD_PER_IMAGE, BITS).estimate(rv_at(1990.0))
    assert 0 <= est.x_finish < 2000


def test_cpu_arm_ties_xavier_and_orin():
    cpu = wen_estimators.CpuEstimator(K.FLOPS_FWD_PER_IMAGE, BITS)
    assert cpu.train_hat(rv_at(0, "xavier")) == cpu.train_hat(rv_at(0, "orin"))


def test_location_only_holds_the_gpu_fleet_mean():
    reports = [rv_at(0, t) for t in ["xavier"] * 15 + ["orin"] * 10 + ["thor"] * 5]
    est = wen_estimators.build("location_only", K.FLOPS_FWD_PER_IMAGE, BITS, reports)
    gpu = wen_estimators.GpuEstimator(K.FLOPS_FWD_PER_IMAGE, BITS)
    mean = np.mean([gpu.train_hat(r) for r in reports])
    assert est.train_hat(rv_at(0, "thor")) == pytest.approx(mean)


def test_oracle_requires_its_callable():
    with pytest.raises(ValueError):
        wen_estimators.build("gpu_oracle", K.FLOPS_FWD_PER_IMAGE, BITS, [])


# --- Partition ------------------------------------------------------------------

@pytest.fixture(scope="module")
def targets():
    return np.repeat(np.arange(10), 5000)[np.random.default_rng(0).permutation(50_000)]


def test_partition_is_disjoint_and_exhaustive(targets):
    p = wen_partition.build(targets, seed=4126)
    allidx = np.concatenate(list(p.indices.values()))
    assert len(allidx) == 50_000 and len(np.unique(allidx)) == 50_000
    assert [len(p.indices[c]) for c in range(1, 31)] == [K.partition_size(c)
                                                         for c in range(1, 31)]


def test_partition_is_deterministic_and_seeded(targets):
    a = wen_partition.build(targets, seed=4126)
    b = wen_partition.build(targets, seed=4126)
    c = wen_partition.build(targets, seed=4127)
    assert all(np.array_equal(a.indices[k], b.indices[k]) for k in a.indices)
    assert not np.array_equal(a.indices[1], c.indices[1])


@pytest.mark.parametrize("seed", range(4120, 4140))
def test_partition_rounding_completes_for_many_seeds(targets, seed):
    p = wen_partition.build(targets, seed=seed)
    assert p.counts.sum(axis=0).tolist() == [5000] * 10


def test_partition_stays_non_iid(targets):
    s = wen_partition.build(targets, seed=4126).summary()
    assert s["max_class_share_mean"] > 0.25         # iid would be 0.10
