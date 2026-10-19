"""Tests for Change B -- randomness that belongs to the client, not the worker.

The claim under test: two processes that serve the same (client, round) must draw
the same augmentation sequence, whichever worker they are. That is the property
whose absence made `train_loss` diverge at round 3 in
`stage1_evidence/determinism.txt`.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import client_seeded_trainer as cst

DATASET_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DATASET_DIR.parents[3]

# Subprocesses get a minimal env, so PYTHONPATH has to carry what conftest.py
# arranges in-process: PLATO is imported from the source tree, not installed.
SUBPROCESS_ENV = {
    "PATH": "/usr/bin:/bin",
    "PYTHONPATH": f"{DATASET_DIR}:{REPO_ROOT}",
}


@pytest.fixture
def fake_config(monkeypatch):
    fake = SimpleNamespace(clients=SimpleNamespace(random_seed=4126))
    # Patch the name BOUND IN THE MODULE: client_seeded_trainer does
    # `from plato.config import Config` at import time, so rebinding
    # plato.config.Config afterwards would not reach it.
    monkeypatch.setattr(cst, "Config", lambda *a, **k: fake)
    return fake


# ---------------------------------------------------------------------------
# derive_seed
# ---------------------------------------------------------------------------

class TestDeriveSeed:
    def test_is_a_pure_function_of_its_three_inputs(self):
        assert cst.derive_seed(4126, 7, 3) == cst.derive_seed(4126, 7, 3)

    def test_differs_per_client(self):
        seeds = {cst.derive_seed(4126, cid, 3) for cid in range(1, 41)}
        assert len(seeds) == 40

    def test_differs_per_round(self):
        seeds = {cst.derive_seed(4126, 7, rnd) for rnd in range(1, 201)}
        assert len(seeds) == 200

    def test_differs_per_base_seed(self):
        assert cst.derive_seed(4126, 7, 3) != cst.derive_seed(4127, 7, 3)

    def test_is_stable_across_processes(self):
        """Python's hash() of a str is randomised per process by PYTHONHASHSEED,
        which would reintroduce exactly the cross-process variation this module
        removes. blake2b is not."""
        script = (
            f"import sys; sys.path.insert(0, {str(DATASET_DIR)!r});"
            "import client_seeded_trainer as c;"
            "print(c.derive_seed(4126, 7, 3))"
        )
        outputs = set()
        for hashseed in ("0", "1", "12345"):
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True,
                env={**SUBPROCESS_ENV, "PYTHONHASHSEED": hashseed},
            )
            assert proc.returncode == 0, proc.stderr
            outputs.add(proc.stdout.strip())
        assert len(outputs) == 1
        assert outputs.pop() == str(cst.derive_seed(4126, 7, 3))

    def test_neighbouring_inputs_do_not_give_neighbouring_seeds(self):
        """Adjacent Mersenne Twister states are correlated in their first draws,
        and a round here consumes only a few hundred of them."""
        a = cst.derive_seed(4126, 7, 3)
        b = cst.derive_seed(4126, 7, 4)
        c = cst.derive_seed(4126, 8, 3)
        assert abs(a - b) > 1000
        assert abs(a - c) > 1000

    def test_is_a_valid_seed_for_numpy(self):
        """np.random.seed rejects anything >= 2**32; the modulus keeps it under."""
        for cid in range(1, 101):
            for rnd in range(1, 51):
                seed = cst.derive_seed(4126, cid, rnd)
                assert 0 <= seed < 2**31

    def test_round_none_is_tolerated(self):
        assert isinstance(cst.derive_seed(4126, 7, None), int)


# ---------------------------------------------------------------------------
# base_seed
# ---------------------------------------------------------------------------

class TestBaseSeed:
    def test_reads_clients_random_seed(self, fake_config):
        assert cst.base_seed() == 4126

    def test_missing_seed_is_fatal(self, monkeypatch):
        fake = SimpleNamespace(clients=SimpleNamespace())
        monkeypatch.setattr(cst, "Config", lambda *a, **k: fake)
        with pytest.raises(ValueError, match="random_seed is missing"):
            cst.base_seed()

    def test_no_second_seed_knob(self):
        source = (DATASET_DIR / "client_seeded_trainer.py").read_text()
        assert "random_seed" in source
        assert "torch_seed" not in source


# ---------------------------------------------------------------------------
# seed_for_training_run -- what the RNGs actually do afterwards
# ---------------------------------------------------------------------------

class TestSeedForTrainingRun:
    def test_same_client_and_round_gives_the_same_draws(self, fake_config):
        torch = pytest.importorskip("torch")
        cst.seed_for_training_run(7, 3)
        first = torch.randn(8).tolist()
        cst.seed_for_training_run(7, 3)
        assert torch.randn(8).tolist() == first

    def test_different_client_gives_different_draws(self, fake_config):
        torch = pytest.importorskip("torch")
        cst.seed_for_training_run(7, 3)
        first = torch.randn(8).tolist()
        cst.seed_for_training_run(8, 3)
        assert torch.randn(8).tolist() != first

    def test_different_round_gives_different_draws(self, fake_config):
        torch = pytest.importorskip("torch")
        cst.seed_for_training_run(7, 3)
        first = torch.randn(8).tolist()
        cst.seed_for_training_run(7, 4)
        assert torch.randn(8).tolist() != first

    def test_the_augmentation_a_client_sees_is_identical(self, fake_config):
        """The concrete thing that diverged: RandomCrop + RandomHorizontalFlip
        over the same image, seeded for the same (client, round)."""
        torch = pytest.importorskip("torch")
        torchvision = pytest.importorskip("torchvision")
        from torchvision import transforms

        augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
        ])
        image = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(3, 32, 32)

        def draws():
            cst.seed_for_training_run(14, 5)
            return [augment(image).sum().item() for _ in range(20)]

        assert draws() == draws()

    def test_prior_process_state_does_not_leak_in(self, fake_config):
        """The whole point: whatever the worker did before, the client's stream
        starts from the same place."""
        torch = pytest.importorskip("torch")

        def draws_after(noise_draws):
            torch.manual_seed(999)
            torch.randn(noise_draws)              # stand-in for other clients
            cst.seed_for_training_run(14, 5)
            return torch.randn(8).tolist()

        assert draws_after(0) == draws_after(137) == draws_after(5000)


# ---------------------------------------------------------------------------
# Two workers serving the same client must agree -- the actual regression
# ---------------------------------------------------------------------------

class TestWorkerIndependence:
    """`determinism.txt` section 4: worker 2 served client 14 in run A and worker
    4 served it in run B, and the two drew different augmentations."""

    SCRIPT = textwrap.dedent("""
        import os, sys
        sys.path.insert(0, {dataset!r})
        from types import SimpleNamespace
        import torch
        import client_seeded_trainer as cst
        cst.Config = lambda *a, **k: SimpleNamespace(
            clients=SimpleNamespace(random_seed=4126))

        # Stage 1's per-process seeding, with this worker's own offset.
        offset = int(os.environ["PLATO_CLIENT_ID"])
        torch.manual_seed(4126 + offset)
        torch.randn(int(os.environ["PRIOR_DRAWS"]))  # work this worker did before

        cst.seed_for_training_run(14, 5)
        print(" ".join(f"{{v:.10f}}" for v in torch.randn(5).tolist()))
    """)

    def draws(self, tmp_path, worker: int, prior: int) -> str:
        script = tmp_path / f"w{worker}.py"
        script.write_text(self.SCRIPT.format(dataset=str(DATASET_DIR)))
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True,
            env={**SUBPROCESS_ENV,
                 "PLATO_CLIENT_ID": str(worker),
                 "PRIOR_DRAWS": str(prior)},
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    def test_two_different_workers_draw_identically(self, tmp_path):
        pytest.importorskip("torch")
        worker2 = self.draws(tmp_path, worker=2, prior=137)
        worker4 = self.draws(tmp_path, worker=4, prior=911)
        assert worker2 == worker4, (
            "two workers serving client 14 round 5 drew differently -- the "
            "parallelism strategy is still leaking into the experiment"
        )

    def test_a_different_client_still_differs(self, tmp_path):
        """Independence from the worker must not become independence from the
        client: the streams still have to be distinct per client."""
        pytest.importorskip("torch")
        script = (
            f"import sys; sys.path.insert(0, {str(DATASET_DIR)!r});"
            "from types import SimpleNamespace;"
            "import client_seeded_trainer as c, torch;"
            "c.Config = lambda *a, **k: SimpleNamespace("
            "clients=SimpleNamespace(random_seed=4126));"
            "c.seed_for_training_run({cid}, 5);"
            "print(torch.randn(3).tolist())"
        )
        out = []
        for cid in (14, 15):
            proc = subprocess.run(
                [sys.executable, "-c", script.format(cid=cid)],
                capture_output=True, text=True, env=SUBPROCESS_ENV,
            )
            assert proc.returncode == 0, proc.stderr
            out.append(proc.stdout)
        assert out[0] != out[1]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

class TestWiring:
    def test_the_entry_point_uses_the_subclass(self):
        source = (DATASET_DIR / "fedasync_seeded.py").read_text()
        assert "trainer=client_seeded_trainer.ClientSeededTrainer" in source

    def test_the_hook_is_train_model_not_the_callback(self):
        """The DataLoader, and therefore every augmentation draw, is created
        inside train_model AFTER on_train_run_start fires
        (plato/trainers/composable.py:412-470). Seeding in the callback would be
        downstream of nothing useful."""
        source = (DATASET_DIR / "client_seeded_trainer.py").read_text()
        assert "def train_model(" in source
        assert "on_train_run_start" not in source.split('"""')[-1]

    def test_it_subclasses_platos_basic_trainer(self):
        from plato.trainers import basic
        assert issubclass(cst.ClientSeededTrainer, basic.Trainer)

    def test_plato_itself_was_not_edited(self):
        """The brief: do not touch anything under plato/."""
        import plato.trainers.basic as basic_module
        source = Path(basic_module.__file__).read_text()
        assert "client_seeded" not in source
        assert "derive_seed" not in source
