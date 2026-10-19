"""Tests for the unanchored (physical) fleet and the matching runtime belief.

The Wen-aligned environment (prompts/roadmap-wen-environment.md, "Anchor
dropped", 15 Oct 2026) replaces the anchor with

    T = 3 x FLOPs_fwd x |S| x E / (eta x TFLOPS),   FLOPs_fwd = 1.1 GFLOP, eta = 0.2

which the roadmap evaluates to Xavier 2.75 s / Orin 0.35 s / Thor 0.18 s at
|S| = 1,667, E = 1. These tests pin that the generator reproduces those numbers,
that the predictor computes the SAME numbers from the same inputs, and that the
anchored code path every finished run used is untouched.
"""

from __future__ import annotations

import csv
from types import SimpleNamespace

import pytest

import build_fleet

from conftest import DATASET_DIR

TIERS_WEN = DATASET_DIR / "tiers" / "tiers_wen.csv"
TIERS_15X = DATASET_DIR / "tiers" / "tiers_estimated_15x.csv"
W_PHYSICAL = 3 * 1.1e9
PARTITION = 1667

PHYSICAL = dict(
    total=30, mix=(50, 35, 15), tiers_csv=TIERS_WEN,
    anchor_tier=None, anchor_sleep=None, partition_size=PARTITION, epochs=1,
    w_model_flops=W_PHYSICAL,
)


def build_physical(tmp_path, **overrides):
    kwargs = dict(PHYSICAL, jitter_sigma=0.149, seed=4126, fleets_dir=tmp_path)
    kwargs.update(overrides)
    return build_fleet.build(**kwargs)


class TestTierFile:
    def test_cpu_frequencies_are_the_roadmap_values(self):
        with TIERS_WEN.open(newline="") as fh:
            ghz = {r["tier_id"]: float(r["cpu_freq_ghz"]) for r in csv.DictReader(fh)}
        assert ghz == {"xavier": 2.2, "orin": 2.2, "thor": 2.6}

    def test_eta_is_0p2_everywhere(self):
        tiers = build_fleet.load_tiers(TIERS_WEN)
        assert {t.eta_nominal for t in tiers.values()} == {0.2}

    def test_gpu_throughput_matches_the_anchored_tier_file(self):
        # Same datasheet TFLOPS; only efficiency and CPU clocks moved.
        wen = build_fleet.load_tiers(TIERS_WEN)
        old = build_fleet.load_tiers(TIERS_15X)
        assert {k: v.gpu_tflops_fp16 for k, v in wen.items()} == \
               {k: v.gpu_tflops_fp16 for k, v in old.items()}


class TestPhysicalFormula:
    def test_tier_times_match_the_roadmap(self, tmp_path):
        _, manifest = build_physical(tmp_path, jitter_sigma=0.0, dry_run=True)
        t = manifest["T_tier_s"]
        assert t["xavier"] == pytest.approx(2.7505, abs=1e-4)
        assert t["orin"] == pytest.approx(0.35263, abs=1e-5)
        assert t["thor"] == pytest.approx(0.18337, abs=1e-5)

    def test_fleet_mean_matches_the_roadmap(self, tmp_path):
        _, manifest = build_physical(tmp_path, jitter_sigma=0.0, dry_run=True)
        assert manifest["E_T_s"] == pytest.approx(1.52, abs=0.005)

    def test_scale_is_exactly_one(self, tmp_path):
        _, manifest = build_physical(tmp_path, dry_run=True)
        assert manifest["anchor_scale"] == 1.0

    def test_counts_are_15_10_5(self, tmp_path):
        _, manifest = build_physical(tmp_path, dry_run=True)
        assert manifest["counts"] == [15, 10, 5]

    def test_manifest_records_w_and_no_anchor(self, tmp_path):
        _, manifest = build_physical(tmp_path, dry_run=True)
        assert manifest["w_model_flops"] == W_PHYSICAL
        assert manifest["anchor_tier"] is None
        assert manifest["anchor_sleep"] is None

    def test_directory_name_says_physical(self, tmp_path):
        directory, _ = build_physical(tmp_path, dry_run=True)
        assert directory.name == "n30_x15-o10-t5_phys_p1667_j0p149_s4126"

    def test_tiers_stay_separated_at_sigma_0p149(self, tmp_path):
        _, manifest = build_physical(tmp_path, dry_run=True)
        assert not any(row["overlap"] for row in manifest["tier_overlaps"])

    def test_different_w_is_a_different_digest(self, tmp_path):
        _, a = build_physical(tmp_path, dry_run=True)
        _, b = build_physical(tmp_path, dry_run=True, w_model_flops=1.1e9)
        assert a["inputs_digest"] != b["inputs_digest"]


class TestRefusals:
    def test_half_anchor_is_refused(self, tmp_path):
        with pytest.raises(SystemExit):
            build_physical(tmp_path, anchor_tier="xavier", anchor_sleep=None)
        with pytest.raises(SystemExit):
            build_physical(tmp_path, anchor_tier=None, anchor_sleep=15.0)

    def test_w_with_an_anchor_is_refused(self, tmp_path):
        with pytest.raises(SystemExit):
            build_physical(tmp_path, anchor_tier="xavier", anchor_sleep=15.0)

    def test_non_positive_w_is_refused(self, tmp_path):
        with pytest.raises(SystemExit):
            build_physical(tmp_path, w_model_flops=0.0)


class TestAnchoredPathUnchanged:
    def test_existing_n30_fleet_rebuilds_to_the_same_digest(self, tmp_path):
        on_disk = DATASET_DIR / "fleets" / "n30_x15-o10-t5_xa15_p1667_j0p180_s4126"
        if not on_disk.exists():
            pytest.skip("anchored n30 fleet not built")
        import json
        recorded = json.loads((on_disk / "manifest.json").read_text())
        _, rebuilt = build_fleet.build(
            total=30, mix=(50, 35, 15), tiers_csv=TIERS_15X, anchor_tier="xavier",
            anchor_sleep=15.0, partition_size=1667, epochs=1, jitter_sigma=0.18,
            seed=4126, fleets_dir=tmp_path, dry_run=True,
        )
        assert rebuilt["inputs_digest"] == recorded["inputs_digest"]


# ---------------------------------------------------------------------------
# Runtime: the predictor must hold the same W as the fleet
# ---------------------------------------------------------------------------

@pytest.fixture
def gpu_tier_sleep(monkeypatch):
    module = pytest.importorskip("gpu_tier_sleep")

    def clear():
        module._computer = None
        module._predictor_scale = None
        module._w_model_flops = None
        module._comm_latency = None
        module._comm_in_predictor = None
        module._oracle_sleeps = None
    clear()
    yield module
    clear()


def fake_config(monkeypatch, clients_csv, **gpu_tier_keys):
    gpu_tier = SimpleNamespace(
        tiers_csv=str(TIERS_WEN), clients_csv=str(clients_csv), **gpu_tier_keys
    )
    fake = SimpleNamespace(
        config=SimpleNamespace(gpu_tier=gpu_tier),
        data=SimpleNamespace(partition_size=PARTITION),
        trainer=SimpleNamespace(epochs=1),
    )
    monkeypatch.setattr("plato.config.Config", lambda *a, **k: fake)


class TestRuntimeBelief:
    def test_predictor_equals_the_fleet_tier_time(self, tmp_path, monkeypatch,
                                                  gpu_tier_sleep):
        directory, manifest = build_physical(tmp_path, jitter_sigma=0.0)
        fake_config(monkeypatch, directory / "clients.csv", w_model_flops=W_PHYSICAL)
        # Clients 1, 16, 26 are the first xavier, orin and thor (contiguous blocks).
        for cid, tier in ((1, "xavier"), (16, "orin"), (26, "thor")):
            predicted = gpu_tier_sleep.gpu_tops_predict_seconds(cid, PARTITION, 1)
            assert predicted == pytest.approx(manifest["T_tier_s"][tier], rel=1e-12)

    def test_absent_key_keeps_the_historical_constant(self, tmp_path, monkeypatch,
                                                      gpu_tier_sleep):
        directory, _ = build_physical(tmp_path, jitter_sigma=0.0)
        fake_config(monkeypatch, directory / "clients.csv")
        assert gpu_tier_sleep.w_model_flops() == gpu_tier_sleep.W_MODEL_FLOPS

    def test_key_with_an_anchor_raises(self, tmp_path, monkeypatch, gpu_tier_sleep):
        directory, _ = build_physical(tmp_path, jitter_sigma=0.0)
        fake_config(monkeypatch, directory / "clients.csv", w_model_flops=W_PHYSICAL,
                    anchor_tier="xavier", anchor_sleep=15)
        with pytest.raises(ValueError):
            gpu_tier_sleep.w_model_flops()

    def test_non_positive_key_raises(self, tmp_path, monkeypatch, gpu_tier_sleep):
        directory, _ = build_physical(tmp_path, jitter_sigma=0.0)
        fake_config(monkeypatch, directory / "clients.csv", w_model_flops=-1)
        with pytest.raises(ValueError):
            gpu_tier_sleep.w_model_flops()
