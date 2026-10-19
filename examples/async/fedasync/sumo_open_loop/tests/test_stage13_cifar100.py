"""
Stage 13 (CIFAR-100 pilot): the generated configs change only what the spec
allows, and the real datasource / sampler / model build for 100 classes.
CPU only, no training run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
OPEN_LOOP = HERE.parent
FEDASYNC = OPEN_LOOP.parent
CONFIGS = OPEN_LOOP / "configs"
for _p in (FEDASYNC, FEDASYNC / "gpu_tiers", FEDASYNC / "gpu_tiers_dataset",
           FEDASYNC / "sumo_close_loop", OPEN_LOOP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PAIRS = [("stage13nf_s4126", "baseline", "stage10nf_s4126"),
         ("stage13f_s4126", "baseline", "stage10f1_s4126"),
         ("stage13f_s4126", "gpu", "stage10f1_s4126"),
         ("stage13f_s4126", "observed", "stage10f1_s4126"),
         ("stage13f_s4126", "cpu", "stage10f1_s4126"),
         ("stage13rep_s4126", "baseline", "stage10f1_s4126")]
ALLOWED = re.compile(r'^(base_path = |port = |dataset_name = "CIFAR100"|target_accuracy = 0\.99|'
                     r'\[parameters\.model\]|num_classes = 100|# |$)')


@pytest.mark.parametrize("prefix,arm,source", PAIRS)
def test_config_differs_only_where_allowed(prefix, arm, source):
    new = (CONFIGS / prefix / f"{arm}_open.toml").read_text().splitlines()
    old = set((CONFIGS / source / f"{arm}_open.toml").read_text().splitlines())
    extra = [ln for ln in new if ln not in old and not ALLOWED.match(ln)]
    assert extra == []
    gone = old - set(new)
    assert all(re.match(r'^(base_path|port|dataset_name|target_accuracy) = ', ln) for ln in gone), gone
    assert "rounds = 4000" in new


def test_ports_and_run_dirs_unique():
    ports, dirs = [], []
    for prefix, arm, _ in PAIRS:
        t = (CONFIGS / prefix / f"{arm}_open.toml").read_text()
        ports.append(re.search(r"(?m)^port = (\d+)", t).group(1))
        dirs.append(re.search(r'(?m)^base_path = "(.*)"', t).group(1))
    assert len(set(ports)) == 6 and len(set(dirs)) == 6
    used = []
    for f in CONFIGS.rglob("*.toml"):
        if not f.parent.name.startswith("stage13"):
            m = re.search(r"(?m)^port = (\d+)", f.read_text())
            if m:
                used.append(m.group(1))
    assert not set(ports) & set(used)


CHECK = """
import sys, torch
sys.argv = ["x", "-c", sys.argv[1]]
from plato.config import Config
Config()
from plato.datasources import registry as datasources
from plato.models import registry as models
from plato.samplers import registry as samplers
ds = datasources.get()
assert len(ds.classes()) == 100, len(ds.classes())
assert ds.num_train_examples() == 50000 and ds.num_test_examples() == 10000
sampler = samplers.get(ds, client_id=7)
idx = list(sampler.get())
assert sampler.num_samples() == 2500 and len(idx) == 2500
assert max(ds.targets()[i] for i in idx) > 9          # labels drawn from all 100 classes
model = models.get()
assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 100)
x, y = ds.get_train_set()[idx[0]]
loss = torch.nn.functional.cross_entropy(model(x.unsqueeze(0)), torch.tensor([y]))
loss.backward()
assert torch.isfinite(loss)
print("OK")
"""


def test_dataset_sampler_model_build_for_100_classes(tmp_path):
    """In a fresh process: Plato's Config is a process-wide singleton.  The
    config is a copy whose base_path is a temp dir: loading it makes Plato
    write a server_<pid>.log under base_path, which must never land in a
    real run folder (the analysis reads the newest server log there)."""
    import os
    import subprocess
    cfg = (CONFIGS / "stage13f_s4126" / "gpu_open.toml").read_text()
    cfg = re.sub(r'(?m)^base_path = ".*"$', f'base_path = "{tmp_path}"', cfg, count=1)
    copy = tmp_path / "gpu_open.toml"
    copy.write_text(cfg)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join([str(FEDASYNC.parents[2]), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-c", CHECK, str(copy)],
                       capture_output=True, text=True, cwd=FEDASYNC, env=env, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stderr[-2000:]


@pytest.mark.parametrize("prefix,arm,_src", PAIRS)
def test_stage13b_is_stage13_at_10000_rounds(prefix, arm, _src):
    """Stage 13B = Stage 13 config with rounds = 10000 (and its own folder / port)."""
    s13 = (CONFIGS / prefix / f"{arm}_open.toml").read_text().splitlines()
    s12 = (CONFIGS / prefix.replace("stage13", "stage13b", 1) / f"{arm}_open.toml").read_text().splitlines()
    diff = {(a, b) for a, b in zip(s13, s12) if a != b}
    assert len(s13) == len(s12)
    assert all(a.startswith(("# ", "base_path = ", "port = ", "rounds = ")) for a, _ in diff), diff
    assert "rounds = 10000" in s12 and "target_accuracy = 0.99" in s12
    port = int(next(ln for ln in s12 if ln.startswith("port = ")).split()[-1])
    assert 19721 <= port <= 19726
