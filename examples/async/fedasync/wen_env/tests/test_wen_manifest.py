"""The run manifest, and the refusal to start a full run without locked predictions."""

from __future__ import annotations

import pytest
import yaml

import wen_manifest
from conftest import BUNDLE_DIR, FLEET_DIR

SETTINGS = {"fleet_dir": FLEET_DIR, "bundle_dir": BUNDLE_DIR, "scenario": "S1",
            "arm": "gpu", "stage0": False, "selection": "egreedy", "epsilon": 0.2}


def test_smoke_run_writes_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(wen_manifest, "predictions_commit", lambda: None)
    cfg = tmp_path / "c.toml"
    cfg.write_text("x = 1\n")
    m = wen_manifest.write(tmp_path / "run", cfg, SETTINGS, 200, 4126)
    on_disk = yaml.safe_load((tmp_path / "run" / "results" / "manifest.yaml").read_text())
    assert on_disk == m
    assert m["fleet"] == FLEET_DIR.name and m["predictions_commit"] is None
    assert len(m["config_sha256"]) == 64 and len(m["bundle_manifest_sha256"]) == 64


def test_full_run_refused_without_locked_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(wen_manifest, "predictions_commit", lambda: None)
    cfg = tmp_path / "c.toml"
    cfg.write_text("x = 1\n")
    with pytest.raises(wen_manifest.PredictionsNotLockedError):
        wen_manifest.write(tmp_path / "run", cfg, SETTINGS, 4000, 4126)


def test_full_run_records_the_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(wen_manifest, "predictions_commit", lambda: "abc123")
    cfg = tmp_path / "c.toml"
    cfg.write_text("x = 1\n")
    assert wen_manifest.write(tmp_path / "run", cfg, SETTINGS, 4000, 1)[
        "predictions_commit"] == "abc123"
