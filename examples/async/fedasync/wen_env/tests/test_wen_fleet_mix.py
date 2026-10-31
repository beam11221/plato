"""Prompt M B2: the tier-mix sweep changes the tier counts and nothing else."""

from __future__ import annotations

import csv
import types

import pytest
import wen_constants as K
from conftest import BUNDLE_DIR, FLEET_DIR, RUNS_DIR, TIERS_CSV
from wen_fleet import TIER_ORDER, Fleet, contiguous_tiers, mix_fleet_dir

FLEETS = FLEET_DIR.parent
SWEEP = [m for m in K.FLEET_MIXES if m != "M0"]


def _load(mix):
    d = mix_fleet_dir(FLEETS, mix)
    if not d.exists():
        pytest.skip(f"{d.name} not built (wen_env/wen_fleet_mixes.py)")
    return Fleet.load(d, TIERS_CSV, mix=mix)


def test_m0_is_the_locked_fleet():
    assert mix_fleet_dir(FLEETS, "M0") == FLEET_DIR
    assert dict(zip(TIER_ORDER, K.FLEET_MIXES["M0"])) == K.TIER_COUNTS


@pytest.mark.parametrize("mix", list(K.FLEET_MIXES))
def test_counts_blocks_and_jitter(mix):
    fleet, m0 = _load(mix), _load("M0")
    assert [fleet.tier(c) for c in fleet.client_ids] == contiguous_tiers(mix)
    assert sum(K.FLEET_MIXES[mix]) == K.N_CLIENTS
    # eps_i is per id: identical to M0 for every id.
    assert all(fleet.jitter(c) == m0.jitter(c) for c in fleet.client_ids)


def test_wrong_mix_is_refused():
    with pytest.raises(ValueError):
        Fleet.load(mix_fleet_dir(FLEETS, "M0"), TIERS_CSV, mix="M1")


@pytest.mark.parametrize("mix", list(K.FLEET_MIXES))
def test_estimator_tier_equals_ground_truth_tier(mix):
    """What the selectors see (ReportedView) and what the simulator uses
    (GroundTruth) agree on every id's tier and spec."""
    from wen_reported import ReportedView
    from wen_trace import TraceIndex, load_speeds, veh_id
    from wen_truth import GroundTruth

    fleet = _load(mix)
    speeds = load_speeds(BUNDLE_DIR)
    trace = TraceIndex.from_bundle(BUNDLE_DIR, speeds)

    def spec(cid):
        s = fleet.spec(cid)
        return s.tier_id, s.tflops_fp16, s.cpu_ghz

    view = ReportedView(position=lambda c, t: trace.x_at(veh_id(c), t),
                        speed=lambda c: speeds[veh_id(c)], spec=spec,
                        n_samples=K.partition_size)
    truth = GroundTruth(fleet, trace, K.FLOPS_FWD_PER_IMAGE, 357_566_784, 4126, "S1")
    for c in fleet.client_ids:
        rv = view.report(c, 100.0)
        out = truth.dispatch(c, 100.0, 1)
        assert rv.tier == fleet.tier(c) == out.tier
        assert rv.tflops_fp16 == K.TIER_TFLOPS_FP16[rv.tier]


def test_partitions_do_not_depend_on_the_fleet():
    """wen_partition.build takes only the labels and the data seed."""
    import inspect

    import wen_partition
    assert list(inspect.signature(wen_partition.build).parameters) == \
        ["targets", "seed", "alpha"]


def test_config_refuses_mismatched_fleet_dir():
    from wen_config import validate_config

    def cfg(mix, fleet_dir, arm="gpu"):
        wen = types.SimpleNamespace(arm=arm, scenario="S1", stage0=False,
                                    selection="egreedy", epsilon=0.2,
                                    bundle_dir=str(BUNDLE_DIR), fleet_dir=str(fleet_dir),
                                    tiers_csv=str(TIERS_CSV), fleet_mix=mix)
        return types.SimpleNamespace(
            wen_env=wen, gpu_tier=None, server=types.SimpleNamespace(polaris_alpha=1.0),
            clients=types.SimpleNamespace(total_clients=30, per_round=3),
            trainer=types.SimpleNamespace(epochs=1))
    assert validate_config(cfg("M1", mix_fleet_dir(FLEETS, "M1")))["fleet_mix"] == "M1"
    with pytest.raises(ValueError):
        validate_config(cfg("M1", FLEET_DIR))
    with pytest.raises(ValueError):
        validate_config(cfg("M1", mix_fleet_dir(FLEETS, "M1"), arm="cpu"))


@pytest.mark.parametrize("mix", SWEEP)
def test_absent_tiers_never_appear_in_sweep_logs(mix):
    absent = {t for t, n in zip(TIER_ORDER, K.FLEET_MIXES[mix]) if n == 0}
    runs = sorted(RUNS_DIR.glob(f"S1_{mix}_*_s4126"))
    if not absent or not runs:
        pytest.skip("no absent tier or no sweep run yet")
    for run in runs:
        for name in ("dispatch_log.csv",):
            with (run / name).open(newline="") as fh:
                assert not {r["tier"] for r in csv.DictReader(fh)} & absent, run.name
