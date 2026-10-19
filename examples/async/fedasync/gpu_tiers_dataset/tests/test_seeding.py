"""Tests for seeding.py.

The claims under test, in order of how much rests on them:

  1. the seed comes from the config's existing `random_seed` and nowhere else
  2. a missing seed is fatal, never defaulted
  3. the server and each client process get distinct, derived, reproducible seeds
  4. every RNG named in the brief is actually pinned
  5. the module-level call really does run inside a spawned child process
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import seeding

DATASET_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(textwrap.dedent("""
        [clients]
        total_clients = 20
        random_seed = 4126

        [server]
        random_seed = 4126

        [data]
        random_seed = 4126
    """))
    return path


# ---------------------------------------------------------------------------
# Finding the config
# ---------------------------------------------------------------------------

class TestConfigPath:
    @pytest.mark.parametrize("argv", [
        ["-c", "run.toml"],
        ["--config", "run.toml"],
        ["--config=run.toml"],
        ["-crun.toml"],
        ["--arm", "gpu", "-c", "run.toml"],
        ["-c", "run.toml", "--arm", "gpu"],
    ])
    def test_forms_plato_accepts(self, argv):
        assert seeding.config_path_from_argv(argv) == Path("run.toml")

    def test_defaults_the_way_plato_does(self):
        assert seeding.config_path_from_argv([]) == Path("./config.toml")

    def test_flag_without_value_is_fatal(self):
        with pytest.raises(SystemExit, match="no value"):
            seeding.config_path_from_argv(["-c"])


# ---------------------------------------------------------------------------
# Reading the seed
# ---------------------------------------------------------------------------

class TestReadSeed:
    def test_reads_clients_random_seed(self, config):
        assert seeding.read_seed(config) == 4126

    def test_missing_file_is_fatal(self, tmp_path):
        with pytest.raises(SystemExit, match="not found"):
            seeding.read_seed(tmp_path / "nope.toml")

    def test_missing_key_is_fatal_not_defaulted(self, tmp_path):
        path = tmp_path / "run.toml"
        path.write_text("[clients]\ntotal_clients = 20\n")
        with pytest.raises(SystemExit, match="random_seed"):
            seeding.read_seed(path)

    def test_non_integer_seed_is_fatal(self, tmp_path):
        path = tmp_path / "run.toml"
        path.write_text('[clients]\nrandom_seed = "abc"\n')
        with pytest.raises(SystemExit, match="random_seed"):
            seeding.read_seed(path)

    def test_no_second_seed_knob_exists(self):
        """The brief: derive everything from the config's existing seed field."""
        source = (DATASET_DIR / "seeding.py").read_text()
        assert seeding.SEED_KEY == "random_seed"
        assert "torch_seed" not in source
        assert "TORCH_SEED" not in source


# ---------------------------------------------------------------------------
# The per-process offset
# ---------------------------------------------------------------------------

class TestProcessOffset:
    def test_server_process_has_no_offset(self, monkeypatch):
        monkeypatch.delenv("PLATO_CLIENT_ID", raising=False)
        assert seeding.process_offset() == 0

    def test_client_process_uses_its_id(self, monkeypatch):
        monkeypatch.setenv("PLATO_CLIENT_ID", "3")
        assert seeding.process_offset() == 3

    def test_garbage_id_is_fatal(self, monkeypatch):
        monkeypatch.setenv("PLATO_CLIENT_ID", "three")
        with pytest.raises(SystemExit, match="PLATO_CLIENT_ID"):
            seeding.process_offset()

    def test_processes_get_distinct_seeds(self, config, monkeypatch):
        seeds = set()
        for cid in (None, "1", "2", "3", "4"):
            if cid is None:
                monkeypatch.delenv("PLATO_CLIENT_ID", raising=False)
            else:
                monkeypatch.setenv("PLATO_CLIENT_ID", cid)
            seeds.add(seeding.seed_everything(config)["seed"])
        assert len(seeds) == 5

    def test_seed_is_a_pure_function_of_config_seed_and_offset(self, config, monkeypatch):
        monkeypatch.setenv("PLATO_CLIENT_ID", "3")
        record = seeding.seed_everything(config)
        assert record["seed"] == record["base_seed"] + record["offset"] == 4129


# ---------------------------------------------------------------------------
# What actually gets pinned
# ---------------------------------------------------------------------------

class TestSeedEverything:
    def test_python_random_is_reproducible(self, config, monkeypatch):
        import random
        monkeypatch.delenv("PLATO_CLIENT_ID", raising=False)
        seeding.seed_everything(config)
        first = [random.random() for _ in range(5)]
        seeding.seed_everything(config)
        assert [random.random() for _ in range(5)] == first

    def test_numpy_is_reproducible(self, config, monkeypatch):
        np = pytest.importorskip("numpy")
        monkeypatch.delenv("PLATO_CLIENT_ID", raising=False)
        seeding.seed_everything(config)
        first = np.random.rand(5).tolist()
        seeding.seed_everything(config)
        assert np.random.rand(5).tolist() == first

    def test_torch_cpu_is_reproducible(self, config, monkeypatch):
        torch = pytest.importorskip("torch")
        monkeypatch.delenv("PLATO_CLIENT_ID", raising=False)
        seeding.seed_everything(config)
        first = torch.randn(5).tolist()
        seeding.seed_everything(config)
        assert torch.randn(5).tolist() == first

    def test_model_init_is_reproducible(self, config, monkeypatch):
        """The RNG consumer that actually matters: it is what sets G_i."""
        torch = pytest.importorskip("torch")
        monkeypatch.delenv("PLATO_CLIENT_ID", raising=False)

        def first_conv_weights():
            seeding.seed_everything(config)
            layer = torch.nn.Conv2d(3, 64, 3)
            return layer.weight.detach().flatten()[:8].tolist()

        assert first_conv_weights() == first_conv_weights()

    def test_different_offsets_give_different_streams(self, config, monkeypatch):
        torch = pytest.importorskip("torch")
        monkeypatch.setenv("PLATO_CLIENT_ID", "1")
        seeding.seed_everything(config)
        one = torch.randn(5).tolist()
        monkeypatch.setenv("PLATO_CLIENT_ID", "2")
        seeding.seed_everything(config)
        assert torch.randn(5).tolist() != one

    def test_cudnn_is_pinned(self, config, monkeypatch):
        torch = pytest.importorskip("torch")
        monkeypatch.delenv("PLATO_CLIENT_ID", raising=False)
        torch.backends.cudnn.benchmark = True          # the state to overwrite
        record = seeding.seed_everything(config)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert record["cudnn_deterministic"] is True

    def test_cuda_is_seeded_when_present(self, config, monkeypatch):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA on this box")
        monkeypatch.delenv("PLATO_CLIENT_ID", raising=False)
        record = seeding.seed_everything(config)
        assert record["cuda_devices"] == torch.cuda.device_count() >= 1

        seeding.seed_everything(config)
        first = torch.randn(5, device="cuda").tolist()
        seeding.seed_everything(config)
        assert torch.randn(5, device="cuda").tolist() == first

    def test_deterministic_algorithms_is_off_by_default(self, config, monkeypatch):
        monkeypatch.delenv("PLATO_CLIENT_ID", raising=False)
        monkeypatch.delenv("PLATO_DETERMINISTIC_ALGORITHMS", raising=False)
        assert seeding.seed_everything(config)["deterministic_algorithms"] is False

    def test_record_and_format(self, config, monkeypatch):
        monkeypatch.setenv("PLATO_CLIENT_ID", "2")
        seeding.seed_everything(config)
        text = seeding.format_record()
        assert "client_process_2" in text
        assert "seed=4128" in text
        assert seeding.record()["role"] == "client_process_2"


# ---------------------------------------------------------------------------
# The claim the whole design rests on: it runs inside spawned children
# ---------------------------------------------------------------------------

class TestReachesSpawnedProcesses:
    """`mp.Process(target=..., ...)` under spawn re-imports __main__, so a
    module-level `seed_everything()` runs in the child too. If this stops being
    true, seeding covers the server only and the client processes -- which do all
    the training -- go back to being unseeded, silently."""

    SCRIPT = textwrap.dedent("""
        import multiprocessing as mp, os, sys
        sys.path.insert(0, {dataset!r})
        import seeding
        REC = seeding.seed_everything()
        print("SEEDED", REC["role"], REC["seed"], REC["torch"], flush=True)

        def child():
            import torch
            print("DRAW", os.environ.get("PLATO_CLIENT_ID"),
                  round(torch.randn(1).item(), 8), flush=True)

        if __name__ == "__main__":
            mp.set_start_method("spawn", force=True)
            for cid in (1, 2):
                os.environ["PLATO_CLIENT_ID"] = str(cid)
                p = mp.Process(target=child); p.start(); p.join()
                del os.environ["PLATO_CLIENT_ID"]
    """)

    def run(self, tmp_path, config):
        script = tmp_path / "spawn_main.py"
        script.write_text(self.SCRIPT.format(dataset=str(DATASET_DIR)))
        proc = subprocess.run(
            [sys.executable, str(script), "-c", str(config)],
            capture_output=True, text=True, cwd=tmp_path,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout

    def test_children_are_seeded_with_their_own_offsets(self, tmp_path, config):
        pytest.importorskip("torch")
        out = self.run(tmp_path, config)
        seeded = [l.split() for l in out.splitlines() if l.startswith("SEEDED")]
        roles = {row[1]: int(row[2]) for row in seeded}
        assert roles["server"] == 4126
        assert roles["client_process_1"] == 4127
        assert roles["client_process_2"] == 4128
        assert all(row[3] == "True" for row in seeded)      # torch was seeded

    def test_children_draw_reproducibly_across_whole_runs(self, tmp_path, config):
        pytest.importorskip("torch")
        assert self.run(tmp_path, config) == self.run(tmp_path, config)

    def test_children_draw_differently_from_each_other(self, tmp_path, config):
        pytest.importorskip("torch")
        draws = [l.split()[2] for l in self.run(tmp_path, config).splitlines()
                 if l.startswith("DRAW")]
        assert len(draws) == 2 and draws[0] != draws[1]

    def test_a_different_config_seed_changes_every_draw(self, tmp_path, config):
        pytest.importorskip("torch")
        before = self.run(tmp_path, config)
        config.write_text(config.read_text().replace("4126", "4127"))
        assert self.run(tmp_path, config) != before
