"""Prompt D, Step 2: the cpu_cal ablation (tests 2-4 here; test 1 needs runs)."""

from __future__ import annotations

import csv
import tomllib

import numpy as np
import pytest

import wen_constants as K
import wen_estimators
import wen_formulas as F
import wen_manifest
from conftest import FLEET_DIR, RUNS_DIR, TIERS_CSV, WEN
from wen_fleet import Fleet
from wen_reported import ReportedVehicle

BITS = 11_173_962 * 32


@pytest.fixture(scope="module")
def reports():
    fleet = Fleet.load(FLEET_DIR, TIERS_CSV)
    return [ReportedVehicle(c, 0.0, 500.0, 25.0, fleet.tier(c), fleet.spec(c).tflops_fp16,
                            fleet.spec(c).cpu_ghz, K.partition_size(c))
            for c in fleet.client_ids]


@pytest.fixture(scope="module")
def rho(reports):
    return wen_estimators.rho_cal(K.FLOPS_FWD_PER_IMAGE, reports)


def test_rho_cal_is_about_85(rho):
    assert rho == pytest.approx(85.0, abs=0.1)


def test_configs_carry_the_computed_rho(rho):
    for seed in (4126, 4127, 4128):
        cfg = tomllib.loads((WEN / "configs" / "runs" / f"S1_cpu_cal_s{seed}.toml").read_text())
        assert cfg["wen_env"]["cpu_rho"] == rho
        assert cfg["wen_env"]["scenario"] == "S1"
        assert cfg["trainer"]["target_accuracy"] == 0.9


# --- Test 2: fleet-mean cpu_cal T_train equals the fleet-mean datasheet T_train --

def test_fleet_mean_matches_datasheet_to_1e9(reports, rho):
    cal = wen_estimators.build("cpu_cal", K.FLOPS_FWD_PER_IMAGE, BITS, reports, rho=rho)
    gpu = wen_estimators.GpuEstimator(K.FLOPS_FWD_PER_IMAGE, BITS)
    mean_cal = np.mean([cal.train_hat(rv) for rv in reports])
    mean_gpu = np.mean([gpu.train_hat(rv) for rv in reports])
    assert mean_cal == pytest.approx(mean_gpu, rel=1e-9)


# --- Test 3: per-tier values ------------------------------------------------------

@pytest.mark.parametrize("tier, expected", [("xavier", 1.58), ("orin", 1.58), ("thor", 1.34)])
def test_per_tier_cpu_cal_train(reports, rho, tier, expected):
    cal = wen_estimators.build("cpu_cal", K.FLOPS_FWD_PER_IMAGE, BITS, reports, rho=rho)
    rv = next(r for r in reports if r.tier == tier)    # thor clients all hold 1,666
    assert cal.train_hat(rv) == pytest.approx(expected, abs=0.01)


# --- Test 4: no tier blocked at E_max ----------------------------------------------

@pytest.mark.parametrize("tier", ["xavier", "orin", "thor"])
def test_no_tier_blocked(reports, rho, tier):
    cal = wen_estimators.build("cpu_cal", K.FLOPS_FWD_PER_IMAGE, BITS, reports, rho=rho)
    base = next(r for r in reports if r.tier == tier)
    for x in np.linspace(0, 1999, 200):
        rv = ReportedVehicle(base.client_id, 0.0, float(x), 25.0, tier, base.tflops_fp16,
                             base.cpu_ghz, base.n_samples)
        est = cal.estimate(rv)
        assert est.E < K.E_MAX_J
        assert est.E == pytest.approx(F.cpu_train_energy_joules(base.n_samples, base.cpu_ghz, rho)
                                      + K.UPLOAD_POWER_W * est.T_up)


def test_cpu_cal_at_1250_is_the_cpu_estimator(reports):
    cal = wen_estimators.build("cpu_cal", K.FLOPS_FWD_PER_IMAGE, BITS, reports,
                               rho=K.CYCLES_PER_BIT_RHO)
    cpu = wen_estimators.build("cpu", K.FLOPS_FWD_PER_IMAGE, BITS, reports)
    for rv in reports:
        assert cal.estimate(rv) == cpu.estimate(rv)


# --- Step 0 guard ------------------------------------------------------------------

def test_cpu_cal_refused_without_the_addendum(tmp_path, monkeypatch):
    monkeypatch.setattr(wen_manifest, "predictions_commit",
                        lambda path=wen_manifest.PREDICTIONS: None
                        if path == wen_manifest.ADDENDUM_V3_1 else "abc")
    cfg = tmp_path / "c.toml"
    cfg.write_text("x = 1\n")
    settings = {"fleet_dir": FLEET_DIR, "bundle_dir": WEN.parent / "sumo_ver2" /
                "bundle_wen_closed_v1", "scenario": "S1", "arm": "cpu_cal", "stage0": False,
                "selection": "egreedy", "epsilon": 0.2, "cpu_rho": 85.0}
    with pytest.raises(wen_manifest.PredictionsNotLockedError):
        wen_manifest.write(tmp_path / "run", cfg, settings, 30, 4126)   # even a short run


# --- Test 1 (runs) ------------------------------------------------------------------

DETERMINISTIC = ("cycle", "round", "client_id", "t_dispatch", "duration",
                 "t_finish_clean", "accepted")


def _key(run):
    with (run / "dispatch_log.csv").open(newline="") as fh:
        return [tuple(r[c] for c in DETERMINISTIC) for r in csv.DictReader(fh)]


def test_cpu_cal_at_1250_reproduces_cpu_dispatch_log():
    a = RUNS_DIR / "S1_cpu_cal_rho1250_s4126_rhochk"
    b = RUNS_DIR / "S1_cpu_s4126_rhochk"
    if not ((a / "dispatch_log.csv").exists() and (b / "dispatch_log.csv").exists()):
        pytest.skip("rho = 1,250 check runs not present (blocked until the addendum commit)")
    assert len(_key(a)) > 30
    assert _key(a) == _key(b)
