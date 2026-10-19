"""Step 8 tests 1-2 plus the Step 2 trace / fleet checks."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

import wen_constants as K
import wen_formulas as F
from conftest import BUNDLE_DIR, FLEET_DIR, REPO_ROOT, TIERS_CSV, WEN
from wen_fleet import Fleet
from wen_trace import (
    MANIFEST_SHA256, BundleIntegrityError, TraceEndError, TraceIndex, load_speeds,
    veh_id, verify_bundle,
)


@pytest.fixture(scope="module")
def speeds():
    return load_speeds(BUNDLE_DIR)


@pytest.fixture(scope="module")
def trace(speeds):
    return TraceIndex.from_bundle(BUNDLE_DIR, speeds)


@pytest.fixture(scope="module")
def fleet():
    return Fleet.load(FLEET_DIR, TIERS_CSV)


# --- Test 1: upload model -----------------------------------------------------

@pytest.mark.parametrize("d, expected", [(10, 5.0), (100, 7.0), (250, 8.3),
                                         (500, 9.6), (1000, 11.4)])
def test_upload_table_reproduces_the_prompt(d, expected):
    assert F.upload_seconds(d, 1.0, 11.17e6 * 32) == pytest.approx(expected, abs=0.1)


def test_fading_scales_rate_not_distance():
    d, bits = 500.0, 11_173_962 * 32
    assert F.upload_seconds(d, 2.0, bits) < F.upload_seconds(d, 1.0, bits) \
        < F.upload_seconds(d, 0.5, bits)


def test_distance_includes_the_lateral_offset():
    assert F.distance_to_rsu(1000.0) == 10.0
    assert F.distance_to_rsu(400.0) == pytest.approx(math.hypot(600, 10))


# --- Test 2: training time, no anchor ------------------------------------------

@pytest.mark.parametrize("tier, expected", [("xavier", 2.7505), ("orin", 0.35263),
                                            ("thor", 0.18337)])
def test_training_times_are_near_the_prompt(tier, expected):
    """Expected = the roadmap's values at exactly 1.1 GFLOP; measured is +1.2%."""
    t = F.train_seconds(K.FLOPS_FWD_PER_IMAGE, 1667, 1, K.TIER_TFLOPS_FP16[tier])
    assert t == pytest.approx(expected, rel=0.02)


def test_training_time_formula_to_1e9(fleet):
    for cid in fleet.client_ids:
        spec = fleet.spec(cid)
        expected = (3 * K.FLOPS_FWD_PER_IMAGE * K.partition_size(cid) * 1
                    / (0.2 * spec.tflops_fp16 * 1e12))
        got = F.train_seconds(K.FLOPS_FWD_PER_IMAGE, K.partition_size(cid), 1,
                              spec.tflops_fp16)
        assert got == pytest.approx(expected, rel=1e-9)


def test_cpu_estimates_match_the_prompt():
    assert F.cpu_train_seconds(1667, 2.2) == pytest.approx(23.3, abs=0.05)
    assert F.cpu_train_seconds(1667, 2.6) == pytest.approx(19.7, abs=0.05)


def test_pinned_flops_equal_a_fresh_measurement(tmp_path):
    pytest.importorskip("fvcore")
    out = tmp_path / "flops.json"
    subprocess.run([sys.executable, str(WEN / "measure_flops.py"), "--json", str(out)],
                   check=True, capture_output=True,
                   env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"})
    measured = json.loads(out.read_text())
    assert measured["flops_fwd"] == K.FLOPS_FWD_PER_IMAGE
    assert abs(measured["deviation_from_1p1_gflop"]) < 0.20


def test_partition_sizes_sum_to_the_dataset():
    sizes = [K.partition_size(c) for c in range(1, 31)]
    assert sum(sizes) == 50_000
    assert sizes.count(1667) == 20 and sizes.count(1666) == 10


def test_jitter_sigma_is_cv_15_percent():
    assert math.sqrt(math.exp(K.JITTER_SIGMA ** 2) - 1) == pytest.approx(0.15, abs=1e-12)
    assert K.JITTER_SIGMA == pytest.approx(0.149166, abs=5e-7)


# --- Step 2: bundle, mapping, trace -------------------------------------------

def test_bundle_verifies():
    manifest = verify_bundle(BUNDLE_DIR)
    assert manifest["n_vehicles"] == 30 and manifest["motion_model"] == "kinematic"


def test_bundle_refuses_a_changed_manifest(tmp_path):
    import shutil
    copy = tmp_path / "bundle"
    shutil.copytree(BUNDLE_DIR, copy)
    (copy / "manifest.yaml").write_text((copy / "manifest.yaml").read_text() + "\n# x")
    with pytest.raises(BundleIntegrityError):
        verify_bundle(copy)
    assert MANIFEST_SHA256 != ""


def test_bundle_refuses_a_wrong_zip(tmp_path):
    fake = tmp_path / "bundle_wen_closed_v1.zip"
    fake.write_bytes(b"not the bundle")
    with pytest.raises(BundleIntegrityError):
        verify_bundle(BUNDLE_DIR, fake)


def test_speeds_are_the_seeded_draw(speeds):
    draw = np.random.default_rng(20261015).uniform(17, 34, 30)
    assert [speeds[veh_id(c)] for c in range(1, 31)] == pytest.approx(list(draw), abs=1e-6)
    assert min(speeds.values()) == pytest.approx(17.077, abs=1e-3)
    assert max(speeds.values()) == pytest.approx(32.590, abs=1e-3)


def test_client_veh_mapping_and_tiers(fleet, speeds):
    assert veh_id(1) == "veh00" and veh_id(30) == "veh29"
    expected = ["xavier"] * 15 + ["orin"] * 10 + ["thor"] * 5
    assert [fleet.tier(c) for c in range(1, 31)] == expected
    means = {t: np.mean([speeds[veh_id(c)] for c in fleet.client_ids if fleet.tier(c) == t])
             for t in ("xavier", "orin", "thor")}
    assert means["xavier"] == pytest.approx(26.8, abs=0.05)
    assert means["orin"] == pytest.approx(24.9, abs=0.05)
    assert means["thor"] == pytest.approx(28.1, abs=0.05)


def test_lookup_matches_per_lap_kinematics_within_0p2_m(trace, speeds):
    """Inside every lap, x(t) = x(t_lap_start) + V_k (t - t_lap_start).

    The prompt asks for (x0 + V_k t) mod 2000 over the WHOLE trace; that global
    form does not hold for this bundle (each wrap costs one 0.1 s step, see
    wen_trace.py) and is checked as a deviation in the next test.
    """
    worst = 0.0
    for vid, v in speeds.items():
        t0, x, lap = trace.samples(vid)
        t = t0 + np.arange(len(x))
        for lap_id in np.unique(lap):
            idx = np.flatnonzero(lap == lap_id)
            pred = x[idx[0]] + v * (t[idx] - t[idx[0]])
            worst = max(worst, float(np.abs(pred - x[idx]).max()))
            mids = t[idx[:-1]] + 0.5
            got = np.array([trace.x_at(vid, m) for m in mids[:: max(1, len(mids) // 5)]])
            want = x[idx[0]] + v * (mids[:: max(1, len(mids) // 5)] - t[idx[0]])
            worst = max(worst, float(np.abs(got - want).max()) if len(got) else 0.0)
    assert worst < 0.2


def test_global_analytic_form_drifts_by_one_step_per_lap(trace, speeds):
    """Documents the deviation: a single global x0 drifts by ~V*0.1 per lap."""
    vid = "veh04"          # slowest: ~435 m of drift, short of the 1 km wrap-around
    v = speeds[vid]
    t_end = trace.t_end
    analytic = (trace.x_at(vid, 1.0) + v * (t_end - 1.0)) % K.ROAD_LENGTH_M
    drift = abs(((trace.x_at(vid, t_end) - analytic) + 1000) % 2000 - 1000)
    laps = trace.lap_at(vid, t_end) - trace.lap_at(vid, 1.0)
    assert drift == pytest.approx(laps * v * 0.1, rel=0.05)


def test_lookup_never_interpolates_across_the_wrap(trace, speeds):
    for vid, v in speeds.items():
        t0, x, lap = trace.samples(vid)
        k = int(np.flatnonzero(np.diff(lap) != 0)[0])
        for frac in (0.1, 0.5, 0.9):
            t = t0 + k + frac
            got = trace.x_at(vid, t)
            assert 0.0 <= got < K.ROAD_LENGTH_M
            # Either continuing from the left sample, or backing off the right.
            assert min(abs(got - (x[k] + v * frac)),
                       abs(got - (x[k + 1] - v * (1 - frac)))) < 1e-9


def test_trace_end_is_a_hard_stop(trace):
    with pytest.raises(TraceEndError):
        trace.x_at("veh00", trace.t_end + 0.5)


def test_lane_is_never_read():
    for name in ("wen_trace.py", "wen_truth.py", "wen_server.py", "wen_estimators.py",
                 "wen_selection.py", "wen_reported.py"):
        text = (WEN / name).read_text()
        code = "\n".join(line.split("#")[0] for line in text.splitlines())
        assert '"lane"' not in code and "['lane']" not in code, name
