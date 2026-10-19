"""Prompt H, Step 3: the wen_fresh arm (Wen-faithful with a fresh F3).

Unit tests here: test 2 (F3 against a direct evaluation), test 3 (cache), and the
config / manifest guards. Tests 1, 4 and 5 need runs and skip until they exist.
"""

from __future__ import annotations

import ast
import csv
from types import SimpleNamespace

import pytest
import torch

import wen_constants as K
import wen_faithful as W
import wen_fresh
import wen_manifest
import wen_partition
from conftest import BUNDLE_DIR, FLEET_DIR, RUNS_DIR, WEN
from wen_estimators import CpuEstimator
from wen_reported import ReportedVehicle

BITS = 357_566_784
DATA = WEN.parent / "data"


def _need_cifar():
    if not (DATA / "cifar-10-batches-py").exists():
        pytest.skip("CIFAR-10 not on disk")


def _resnet18(seed=0):
    from plato.models import resnet

    torch.manual_seed(seed)
    return resnet.Model.get(model_name="resnet_18", num_classes=10)


# --- Test 2: F3 for 3 vehicles matches a direct evaluation to 1e-6 ------------------

def test_fresh_f3_matches_direct_evaluation():
    _need_cifar()
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    fresh = wen_fresh.FreshLoss(DATA, 4126)
    model = _resnet18().to(fresh.device)
    f3 = fresh.evaluate(model)
    assert sorted(f3) == list(range(1, K.N_CLIENTS + 1))

    # Direct path: torchvision's CIFAR10 with the test transform (ToTensor + the
    # training Normalize), a Subset of the client's partition, a DataLoader.
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(wen_fresh.CIFAR10_MEAN, wen_fresh.CIFAR10_STD)])
    train = datasets.CIFAR10(root=str(DATA), train=True, download=False, transform=tf)
    part = wen_partition.build(list(train.targets), 4126)
    model.eval()
    for cid in (1, 21, 30):                         # a Xavier, an Orin-block, a Thor
        total, n = 0.0, 0
        with torch.no_grad():
            for x, y in DataLoader(Subset(train, part.indices[cid].tolist()), batch_size=500):
                out = model(x.to(fresh.device))
                total += torch.nn.functional.cross_entropy(
                    out, y.to(fresh.device), reduction="sum").double().item()
                n += len(y)
        assert n == K.partition_size(cid)
        assert f3[cid] == pytest.approx(total / n, rel=0, abs=1e-6), cid


# --- Test 3: cache -- F3 changes only when global_version changes -------------------

class _StubFresh:
    """Deterministic stand-in for FreshLoss: F3 depends on the model's weights."""

    device = torch.device("cpu")

    def __init__(self):
        self.calls = 0

    def evaluate(self, model):
        self.calls += 1
        w = float(model.weight.sum())
        return {c: w + 0.01 * c for c in range(1, K.N_CLIENTS + 1)}


class _FakeServer:
    training_clients = {}
    current_round = 1

    def __init__(self):
        self.global_version = 0
        self.model = torch.nn.Linear(2, 1)
        torch.nn.init.constant_(self.model.weight, 1.0)

    def require_trainer(self):
        return SimpleNamespace(model=self.model)

    def sumo_time(self):
        return 0.0

    def wen_report(self, cid, t):
        return ReportedVehicle(cid, t, 300.0 + 40 * cid, 25.0, "xavier", 10.0, 2.2, 1667)


def _fresh_selection(tmp_path, fresh, stale=False):
    return wen_fresh.WenFreshSelection(
        fresh=fresh, stale_f3_test=stale,
        estimator=CpuEstimator(K.FLOPS_FWD_PER_IMAGE, BITS), t_max=34.0, e_max=36.0,
        kl={c: 0.5 for c in range(1, 31)}, out_dir=tmp_path)


def test_fresh_f3_cached_per_global_version(tmp_path):
    fresh, server = _StubFresh(), _FakeServer()
    sel = _fresh_selection(tmp_path, fresh)
    ctx = SimpleNamespace(server=server)

    sel.select_clients(list(range(1, 31)), 1, ctx)
    first = dict(sel.fresh_f3)
    with torch.no_grad():
        server.model.weight += 1.0                  # a model change without a version bump
    sel.select_clients(list(range(1, 31)), 1, ctx)
    assert fresh.calls == 1 and sel.fresh_f3 == first    # no recompute between merges

    server.global_version = 1                       # a merge
    sel.select_clients(list(range(1, 31)), 1, ctx)
    assert fresh.calls == 2 and sel.fresh_f3 != first
    assert all(sel.f3(c) == sel.fresh_f3[c] for c in range(1, 31))

    sel._eval_fh.flush(), sel._dec_fh.flush()
    evals = list(csv.DictReader((tmp_path / "wen_fresh_eval.csv").open()))
    decs = list(csv.DictReader((tmp_path / "wen_decisions.csv").open()))
    assert [e["global_version"] for e in evals] == ["0", "1"]
    assert [d["global_version"] for d in decs] == ["0", "0", "1"]
    assert all(float(e["fresh_eval_s"]) >= 0.0 for e in evals)


def test_fresh_f3_ignores_reported_losses(tmp_path):
    sel = _fresh_selection(tmp_path, _StubFresh())
    sel.select_clients(list(range(1, 31)), 1, SimpleNamespace(server=_FakeServer()))
    before = sel.f3(5)
    sel.on_reports_received([SimpleNamespace(client_id=5, report=SimpleNamespace(train_loss=9.0))],
                            SimpleNamespace(state={}))
    assert sel.f3(5) == before                      # no prior, no reported loss


def test_stale_test_mode_decides_on_the_wen_f3(tmp_path):
    sel = _fresh_selection(tmp_path, _StubFresh(), stale=True)
    sel.select_clients(list(range(1, 31)), 1, SimpleNamespace(server=_FakeServer()))
    sel.on_reports_received([SimpleNamespace(client_id=5, report=SimpleNamespace(train_loss=9.0))],
                            SimpleNamespace(state={}))
    assert sel.f3(5) == 9.0 and sel.f3(6) == 9.0    # wen's last loss / prior
    assert sel.fresh_f3                              # the fresh evaluation still ran


def test_same_selector_code_path_as_wen():
    assert issubclass(wen_fresh.WenFreshSelection, W.WenFaithfulSelection)
    overridden = {n for n in vars(wen_fresh.WenFreshSelection) if not n.startswith("_")}
    assert overridden <= {"decision_fields", "candidate_fields", "refresh", "f3",
                          "select_clients", "log_extra"}
    assert W.WenFaithfulSelection.decision_fields == W.DECISION_FIELDS   # wen logs unchanged
    assert "global_version" not in W.DECISION_FIELDS


def test_wen_fresh_does_not_import_ground_truth():
    tree = ast.parse((WEN / "wen_fresh.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"wen_truth", "wen_fleet", "wen_trace"}


# --- config and manifest guards ----------------------------------------------------

def _cfg(arm="wen_fresh", **wen):
    base = dict(arm=arm, scenario="S1", selection="wen", wen_t_max=34.3, wen_e_max=36.4,
                bundle_dir="b", fleet_dir="f", tiers_csv="t")
    base.update(wen)
    return SimpleNamespace(
        wen_env=SimpleNamespace(**base), gpu_tier=None,
        server=SimpleNamespace(polaris_alpha=1.0),
        clients=SimpleNamespace(total_clients=30, per_round=3),
        trainer=SimpleNamespace(epochs=1))


def test_config_accepts_wen_fresh_and_guards_the_stale_flag():
    from wen_config import validate_config

    s = validate_config(_cfg())
    assert s["arm"] == "wen_fresh" and s["wen_fresh_stale_f3_test"] is False
    assert validate_config(_cfg(wen_fresh_stale_f3_test=True))["wen_fresh_stale_f3_test"]
    with pytest.raises(ValueError, match="wen_fresh only"):
        validate_config(_cfg(arm="wen", wen_fresh_stale_f3_test=True))
    with pytest.raises(ValueError, match="Stage 0"):
        validate_config(_cfg(stage0=True))
    with pytest.raises(ValueError, match="no epsilon"):
        validate_config(_cfg(epsilon=0.2))


SETTINGS = {"fleet_dir": FLEET_DIR, "bundle_dir": BUNDLE_DIR, "scenario": "S1",
            "arm": "wen_fresh", "stage0": False, "selection": "wen", "epsilon": None}


def test_wen_fresh_refused_without_the_v3_3_commit(tmp_path, monkeypatch):
    commits = {wen_manifest.ADDENDUM_V3_3: None}
    monkeypatch.setattr(wen_manifest, "predictions_commit",
                        lambda path=wen_manifest.PREDICTIONS: commits.get(path, "abc"))
    cfg = tmp_path / "c.toml"
    cfg.write_text("x = 1\n")
    with pytest.raises(wen_manifest.PredictionsNotLockedError, match="wen_fresh"):
        wen_manifest.write(tmp_path / "run", cfg, SETTINGS, 100, 4126)   # any length
    commits[wen_manifest.ADDENDUM_V3_3] = "def456"
    m = wen_manifest.write(tmp_path / "run", cfg, SETTINGS, 4000, 4126)
    assert m["predictions_v3_3_addendum_commit"] == "def456"


# --- Tests 1, 4, 5: on runs ----------------------------------------------------------

DETERMINISTIC = ("cycle", "round", "client_id", "t_dispatch", "duration",
                 "t_finish_clean", "accepted")


def _rows(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _keys(run, max_round=None):
    rows = _rows(run / "dispatch_log.csv")
    return [tuple(r[c] for c in DETERMINISTIC) for r in rows
            if max_round is None or int(r["round"]) <= max_round]


def test_stale_mode_reproduces_the_wen_run_for_100_rounds():
    stale, wen = RUNS_DIR / "S1_wen_fresh_stale_s4126", RUNS_DIR / "S1_wen_s4126"
    if not (stale / "dispatch_log.csv").exists():
        pytest.skip("stale-F3 test run not present")
    a, b = _keys(stale, 100), _keys(wen, 100)
    assert len(a) >= 100 and a == b


def test_online_clock_unaffected_by_fresh_evaluation():
    run = RUNS_DIR / "S1_wen_fresh_s4126_det1"
    if not (run / "dispatch_log.csv").exists():
        pytest.skip("wen_fresh 100-round run not present")
    import wen_clock_replay

    r = wen_clock_replay.replay(run)
    assert r["max_abs_diff_s"] == 0.0
    assert sum(float(e["fresh_eval_s"]) for e in _rows(run / "wen_fresh_eval.csv")) > 0


def test_determinism_two_100_round_wen_fresh_runs():
    a, b = RUNS_DIR / "S1_wen_fresh_s4126_det1", RUNS_DIR / "S1_wen_fresh_s4126_det2"
    if not ((a / "dispatch_log.csv").exists() and (b / "dispatch_log.csv").exists()):
        pytest.skip("wen_fresh determinism runs not present")
    assert len(_keys(a)) > 100 and _keys(a) == _keys(b)
    assert _rows(a / "wen_decisions.csv") == _rows(b / "wen_decisions.csv")
    assert _rows(a / "wen_candidates.csv") == _rows(b / "wen_candidates.csv")
