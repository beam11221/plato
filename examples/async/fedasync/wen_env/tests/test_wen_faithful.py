"""Prompt E, Step 3: the Wen-faithful selector.

Tests 1-3 and 5-7 are unit tests; tests 4 and 6 also read runs when present
(skipped otherwise). Run with `-s` to see the printed tables.
"""

from __future__ import annotations

import ast
import csv
import math
from types import SimpleNamespace

import numpy as np
import pytest

import wen_constants as K
import wen_faithful as W
import wen_partition
from conftest import FLEET_DIR, RUNS_DIR, TIERS_CSV, WEN
from wen_estimators import CpuEstimator
from wen_fleet import Fleet
from wen_reported import ReportedVehicle

BITS = 357_566_784


# --- Test 1: F1, F2 against a hand calculation -------------------------------------

def hand_f1_f2(n, ghz, x, v):
    """Wen's estimates written out from the formulas, independent of the code."""
    t_com = 24_576 * n * 1_250 / (ghz * 1e9)
    x_hat = (x + t_com * v) % 2_000
    d = math.sqrt((x_hat - 1_000) ** 2 + 10 ** 2)
    p_w = 10 ** (23 / 10) / 1000
    n_w = 10 ** (-114 / 10) / 1000
    rate = 2e6 * math.log2(1 + d ** -3 * p_w / n_w)
    t_up = BITS / rate
    e_com = 1e-28 * 1_250 * 24_576 * n * (ghz * 1e9) ** 2
    return t_com + t_up, e_com + 0.2 * t_up, x_hat


@pytest.mark.parametrize("tier, n, ghz", [("xavier", 1667, 2.2), ("orin", 1667, 2.2),
                                          ("thor", 1666, 2.6)])
@pytest.mark.parametrize("x", [200.0, 1000.0, 1700.0])
def test_f1_f2_match_hand_calculation(tier, n, ghz, x):
    v = 25.0
    est = CpuEstimator(K.FLOPS_FWD_PER_IMAGE, BITS)
    rv = ReportedVehicle(1, 0.0, x, v, tier, K.TIER_TFLOPS_FP16[tier], ghz, n)
    e = est.estimate(rv)
    f1, f2, x_hat = hand_f1_f2(n, ghz, x, v)
    print(f"\n{tier:6s} x={x:6.0f}  F1={e.total:.6f} (hand {f1:.6f})  "
          f"F2={e.E:.6f} (hand {f2:.6f})  x_hat={e.x_finish:.1f}")
    assert e.total == pytest.approx(f1, rel=1e-9)
    assert e.E == pytest.approx(f2, rel=1e-9)
    assert e.x_finish == pytest.approx(x_hat, rel=1e-12)


# --- Test 2: F4 against a hand calculation ------------------------------------------

@pytest.fixture(scope="module")
def cifar_targets():
    path = WEN.parent / "data"
    if not (path / "cifar-10-batches-py").exists():
        pytest.skip("CIFAR-10 not on disk")
    return W.cifar10_train_targets(path)


def test_f4_matches_hand_calculation(cifar_targets):
    part = wen_partition.build(cifar_targets, 4126)
    kl = W.label_kl(cifar_targets, 4126)
    for cid in (1, 16, 30):
        counts = part.counts[cid - 1]
        n = counts.sum()
        hand = sum((c / n) * math.log((c / n) / 0.1) for c in counts if c > 0)
        assert kl[cid] == pytest.approx(hand, rel=1e-12)
        assert len(part.indices[cid]) == n
    vals = sorted(kl.values())
    print(f"\nF4 fleet range (seed 4126): {vals[0]:.4f} – {vals[-1]:.4f}, "
          f"mean {np.mean(vals):.4f}")
    assert all(v >= 0 for v in vals)


def test_kl_edge_cases():
    assert W.kl_to_uniform([1] * 10) == pytest.approx(0.0, abs=1e-15)
    assert W.kl_to_uniform([5, 0, 0, 0, 0, 0, 0, 0, 0, 0]) == pytest.approx(math.log(10))


# --- Test 3: non-dominated set and normalisation ------------------------------------

def test_non_dominated_known_answer():
    pts = {
        1: (1, 9, 5, 5), 2: (2, 8, 5, 5), 3: (3, 7, 5, 5),     # trade-off front
        4: (2, 9, 5, 5),                                        # dominated by 1? no: 1<2
        5: (3, 9, 6, 6),                                        # dominated by 1, 2, 3
        6: (1, 9, 5, 5),                                        # equal to 1: neither dominates
        7: (0, 10, 9, 9),                                       # best F1 only
        8: (4, 4, 4, 4),                                        # front
        9: (4, 4, 4, 5),                                        # dominated by 8
        10: (9, 9, 9, 9),                                       # dominated by 8
    }
    # 4 = (2,9,5,5) is dominated by 1 = (1,9,5,5) and by 2 = (2,8,5,5).
    assert W.non_dominated(pts) == [1, 2, 3, 6, 7, 8]


def test_normalise_edge_cases():
    assert W.normalise({3: (1.0, 2.0, 3.0, 4.0)}) == {3: (0.0, 0.0, 0.0, 0.0)}
    n = W.normalise({1: (1.0, 5.0, 7.0, 0.0), 2: (3.0, 5.0, 8.0, 2.0)})
    assert n == {1: (0.0, 0.0, 0.0, 0.0), 2: (1.0, 0.0, 1.0, 1.0)}


def test_decide_picks_min_mean_and_breaks_ties_by_id():
    obj = {5: (1.0, 2.0, 0.0, 0.0), 2: (2.0, 1.0, 0.0, 0.0)}   # symmetric -> tie
    d = W.decide(obj, t_max=10, e_max=10)
    assert d.pareto == [2, 5] and d.pick == 2 and not d.fallback


def test_decide_feasibility_and_fallback():
    obj = {1: (40.0, 1.0, 0.0, 0.0), 2: (20.0, 50.0, 0.0, 0.0), 3: (30.0, 30.0, 1.0, 1.0)}
    d = W.decide(obj, t_max=34.0, e_max=36.0)
    assert d.feasible == [3] and d.pick == 3 and not d.fallback
    d = W.decide({1: obj[1], 2: obj[2]}, t_max=34.0, e_max=36.0)
    assert d.fallback and d.feasible == [] and d.pick in (1, 2)


# --- Test 5: F3 prior ------------------------------------------------------------------

def _selection(tmp_path, kl=None):
    return W.WenFaithfulSelection(CpuEstimator(K.FLOPS_FWD_PER_IMAGE, BITS), 34.0, 36.0,
                                  kl or {c: 0.5 for c in range(1, 31)}, tmp_path)


def _update(cid, loss):
    return SimpleNamespace(client_id=cid, report=SimpleNamespace(train_loss=loss))


def test_f3_prior_is_the_fleet_mean(tmp_path):
    sel = _selection(tmp_path)
    ctx = SimpleNamespace(state={})
    assert sel.f3(1) == sel.f3(2) == 0.0                     # nobody has reported
    sel.on_reports_received([_update(1, 2.0), _update(2, 1.0)], ctx)
    sel.on_reports_received([_update(1, 0.5)], ctx)
    assert sel.f3(1) == 0.5 and sel.f3(2) == 1.0
    assert sel.f3(3) == pytest.approx((2.0 + 1.0 + 0.5) / 3)  # mean of all reported


def test_f3_equal_before_any_report_normalises_to_zero(tmp_path):
    sel = _selection(tmp_path)
    obj = {c: (25.0 + c, 26.0, sel.f3(c), 0.1 * c) for c in (1, 2, 3)}
    norm = W.normalise(obj)
    assert all(v[2] == 0.0 for v in norm.values())


def test_lost_uploads_do_not_update_f3(tmp_path):
    sel = _selection(tmp_path)
    ctx = SimpleNamespace(state={"sumo_uploaded": {1: False, 2: True}})
    sel.on_reports_received([_update(1, 9.0), _update(2, 1.0)], ctx)
    assert 1 not in sel.last_loss and sel.last_loss[2] == 1.0


# --- Test 7: no estimate path reads ground truth ------------------------------------------

def test_wen_faithful_does_not_import_ground_truth():
    tree = ast.parse((WEN / "wen_faithful.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"wen_truth", "wen_fleet", "wen_trace"}


def test_wen_scores_only_from_reported_records(tmp_path):
    """The selection reads ReportedVehicle via server.wen_report and nothing else."""
    sel = _selection(tmp_path)
    seen = []

    class FakeServer:
        training_clients = {}
        current_round = 1

        def sumo_time(self):
            return 0.0

        def wen_report(self, cid, t):
            seen.append(cid)
            return ReportedVehicle(cid, t, 300.0 + 40 * cid, 25.0, "xavier", 10.0, 2.2, 1667)

    picks = sel.select_clients(list(range(1, 31)), 3, SimpleNamespace(server=FakeServer()))
    assert len(picks) == 3 and len(set(picks)) == 3
    assert set(seen) <= set(range(1, 31))


# --- Tests 4 and 6: on runs ------------------------------------------------------------------

DETERMINISTIC = ("cycle", "round", "client_id", "t_dispatch", "duration",
                 "t_finish_clean", "accepted")


def _rows(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_determinism_two_100_round_runs():
    a, b = RUNS_DIR / "S1_wen_s4126_det1", RUNS_DIR / "S1_wen_s4126_det2"
    if not ((a / "dispatch_log.csv").exists() and (b / "dispatch_log.csv").exists()):
        pytest.skip("determinism runs not present")
    ka = [tuple(r[c] for c in DETERMINISTIC) for r in _rows(a / "dispatch_log.csv")]
    kb = [tuple(r[c] for c in DETERMINISTIC) for r in _rows(b / "dispatch_log.csv")]
    assert len(ka) > 100 and ka == kb
    assert _rows(a / "wen_decisions.csv") == _rows(b / "wen_decisions.csv")


def test_wen_ground_truth_uses_gpu_formulas():
    run = RUNS_DIR / "S1_wen_s4126_smoke"
    if not (run / "dispatch_log.csv").exists():
        pytest.skip("smoke run not present")
    fleet = Fleet.load(FLEET_DIR, TIERS_CSV)
    for r in _rows(run / "dispatch_log.csv"):
        cid = int(r["client_id"])
        t_train = (3 * K.FLOPS_FWD_PER_IMAGE * K.partition_size(cid)
                   / (0.2 * fleet.spec(cid).tflops_fp16 * 1e12)) * fleet.jitter(cid)
        assert float(r["T_train_true"]) == pytest.approx(t_train, rel=1e-9)
        assert float(r["E_train"]) == pytest.approx(K.TIER_POWER_W[r["tier"]] * t_train)
        assert float(r["E_up"]) == pytest.approx(0.2 * float(r["T_up_true"]))
