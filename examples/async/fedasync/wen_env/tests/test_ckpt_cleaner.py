"""The checkpoint cleaner: what it may delete, and the 16 Oct payload race."""

from __future__ import annotations

import os
import sys
import time

from conftest import WEN

sys.path.insert(0, str(WEN / "tools"))
import ckpt_cleaner  # noqa: E402

OLD = time.time() - 3600


def make_run(root, name):
    ckpt = root / name / "checkpoints"
    ckpt.mkdir(parents=True)
    files = ["checkpoint_resnet_18_5.safetensors", "checkpoint_resnet_18_5.safetensors.pkl",
             "checkpoint_resnet_18_6.safetensors", "checkpoint_resnet_18_6.safetensors.pkl",
             "prng_state_5.pkl", "prng_state_6.pkl", "numpy_prng_state_5.pkl",
             "numpy_prng_state_6.pkl", "current_round.pkl", "resnet_18_19.pkl",
             "resnet_18_client_19.pkl", "resnet_18_3.pkl"]
    for i, f in enumerate(files):
        (ckpt / f).write_bytes(b"x")
        os.utime(ckpt / f, (OLD + i, OLD + i))
    (root / name / "dispatch_log.csv").write_text("keep me")
    (root / name / "results").mkdir()
    (root / name / "results" / "r.csv").write_text("keep me")
    return ckpt


def test_running_run_keeps_payloads_and_newest_round(tmp_path, monkeypatch):
    ckpt = make_run(tmp_path, "run_a")
    monkeypatch.setattr(ckpt_cleaner, "running_run_dirs", lambda: {(tmp_path / "run_a").resolve()})
    deleted = {p.name for p, _, _ in ckpt_cleaner.plan(tmp_path, 600, time.time())}
    assert deleted == {"checkpoint_resnet_18_5.safetensors",
                       "checkpoint_resnet_18_5.safetensors.pkl",
                       "prng_state_5.pkl", "numpy_prng_state_5.pkl"}


def test_finished_run_loses_every_checkpoint_and_nothing_else(tmp_path, monkeypatch):
    ckpt = make_run(tmp_path, "run_b")
    monkeypatch.setattr(ckpt_cleaner, "running_run_dirs", lambda: set())
    ckpt_cleaner.one_pass(tmp_path, tmp_path / "log", 600, dry_run=False)
    assert list(ckpt.iterdir()) == []
    assert (tmp_path / "run_b" / "dispatch_log.csv").exists()
    assert (tmp_path / "run_b" / "results" / "r.csv").exists()


def test_a_file_rewritten_after_planning_is_spared(tmp_path, monkeypatch):
    ckpt = make_run(tmp_path, "run_c")
    monkeypatch.setattr(ckpt_cleaner, "running_run_dirs", lambda: {(tmp_path / "run_c").resolve()})
    real_plan = ckpt_cleaner.plan

    def plan_then_rewrite(root, min_age, now):
        victims = real_plan(root, min_age, now)
        os.utime(ckpt / "prng_state_5.pkl")        # touched between plan and unlink
        return victims

    monkeypatch.setattr(ckpt_cleaner, "plan", plan_then_rewrite)
    ckpt_cleaner.one_pass(tmp_path, tmp_path / "log", 600, dry_run=False)
    assert (ckpt / "prng_state_5.pkl").exists()
    assert not (ckpt / "checkpoint_resnet_18_5.safetensors").exists()


def test_dry_run_deletes_nothing(tmp_path, monkeypatch):
    ckpt = make_run(tmp_path, "run_d")
    monkeypatch.setattr(ckpt_cleaner, "running_run_dirs", lambda: set())
    before = sorted(p.name for p in ckpt.iterdir())
    ckpt_cleaner.one_pass(tmp_path, tmp_path / "log", 600, dry_run=True)
    assert sorted(p.name for p in ckpt.iterdir()) == before
