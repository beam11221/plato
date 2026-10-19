"""Prompt C, Step 2: the epsilon-greedy rule (D1) and the per-family energy estimate (D2)."""

from __future__ import annotations

import csv
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

import wen_constants as K
import wen_estimators
import wen_formulas as F
from conftest import BUNDLE_DIR, FLEET_DIR, RUNS_DIR, TIERS_CSV
from wen_egreedy import WenSelector, egreedy_pick
from wen_fleet import Fleet
from wen_reported import ReportedVehicle
from wen_trace import TraceIndex, load_speeds
from wen_truth import GroundTruth

BITS = 11_173_962 * 32


class FakeScorer:
    """Stands in for the Polaris selector: fixed V_i, no RNG."""

    def __init__(self, scores):
        self.scores = scores
        self.last_tau_hats = {}

    def score_candidates(self, eligible, context):
        return {c: self.scores[c] for c in eligible}, {c: 1.0 for c in eligible}, None, 30

    def setup(self, context):
        pass


def context_at(round_no=1):
    server = SimpleNamespace(training_clients={}, current_round=round_no)
    return SimpleNamespace(server=server, state={})


# --- Test 2: epsilon = 0 always picks rank 1 -------------------------------------

def test_epsilon_zero_always_picks_rank_one():
    rng = np.random.default_rng(7)
    for trial in range(500):
        n = int(rng.integers(2, 28))
        cands = sorted(rng.choice(np.arange(1, 31), n, replace=False).tolist())
        scores = {c: float(rng.random()) for c in cands}
        pick, rec = egreedy_pick(cands, scores, 0.0, np.random.default_rng(trial))
        assert rec.rank_of_pick == 1 and not rec.explored
        assert pick == max(cands, key=lambda c: (scores[c], -c))


def test_ties_go_to_the_lower_client_id():
    pick, rec = egreedy_pick([9, 4, 12], {9: 0.5, 4: 0.5, 12: 0.1}, 0.0,
                             np.random.default_rng(0))
    assert pick == 4 and rec.rank_of_pick == 1


# --- Test 3: epsilon = 0.2 over 10,000 decisions with 27 candidates ---------------

def test_epsilon_0p2_rank_one_share_and_floor():
    cands = list(range(1, 28))
    scores = {c: float(v) for c, v in
              zip(cands, np.random.default_rng(3).permutation(np.linspace(0.1, 0.9, 27)))}
    selector = WenSelector(FakeScorer(scores), "egreedy", 0.2, seed=4126)
    picks, ranks = Counter(), Counter()
    for decision in range(10_000):
        chosen = selector.select_clients(cands, 1, context_at(round_no=decision))
        picks[chosen[0]] += 1
        ranks[selector.last_picks[chosen[0]].rank_of_pick] += 1
    p_rank1 = ranks[1] / 10_000
    assert p_rank1 == pytest.approx(0.8 + 0.2 / 27, abs=0.01)
    floor = 0.2 / 27 - 0.003
    assert all(picks[c] / 10_000 >= floor for c in cands), min(picks.values())


def test_key_is_seed_round_k():
    sel = WenSelector(FakeScorer({c: c / 30 for c in range(1, 31)}), "egreedy", 0.5, 1)
    a = [sel._rng(5).random() for _ in range(3)]
    sel2 = WenSelector(FakeScorer({}), "egreedy", 0.5, 1)
    b = [sel2._rng(5).random() for _ in range(3)]
    assert a == b and len(set(a)) == 3        # k advances within a round


def test_random_needs_epsilon_one():
    with pytest.raises(ValueError):
        WenSelector(None, "egreedy", 0.2, 1)
    WenSelector(None, "egreedy", 1.0, 1)


# --- Test 5: cpu energy estimate (D2), no tier blocked at E_max -------------------

def rv(tier, x):
    return ReportedVehicle(1, 0.0, x, 25.0, tier, K.TIER_TFLOPS_FP16[tier],
                           K.TIER_CPU_GHZ[tier], 1667)


@pytest.mark.parametrize("tier, expected", [("xavier", 24.8), ("orin", 24.8),
                                            ("thor", 34.6)])
def test_cpu_training_energy_matches_d2(tier, expected):
    assert F.cpu_train_energy_joules(1667, K.TIER_CPU_GHZ[tier]) == \
        pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize("arm", ["cpu", "gpu"])
@pytest.mark.parametrize("tier", ["xavier", "orin", "thor"])
def test_no_tier_is_blocked_at_e_max(arm, tier):
    est = wen_estimators.build(arm, K.FLOPS_FWD_PER_IMAGE, BITS, [])
    for x in np.linspace(0, 1999, 200):          # worst case is d = 1,000 m
        assert est.estimate(rv(tier, float(x))).E < K.E_MAX_J


def test_cpu_estimate_is_eq7_plus_upload():
    est = wen_estimators.build("cpu", K.FLOPS_FWD_PER_IMAGE, BITS, [])
    e = est.estimate(rv("thor", 500.0))
    assert e.E == pytest.approx(F.cpu_train_energy_joules(1667, 2.6) + 0.2 * e.T_up)


# --- Test 6: ground-truth energy is the GPU formula for every arm -----------------

def test_ground_truth_energy_is_the_gpu_formula():
    speeds = load_speeds(BUNDLE_DIR)
    truth = GroundTruth(Fleet.load(FLEET_DIR, TIERS_CSV),
                        TraceIndex.from_bundle(BUNDLE_DIR, speeds),
                        K.FLOPS_FWD_PER_IMAGE, BITS, 4126, "S2")
    for cid in (1, 16, 26):
        o = truth.dispatch(cid, 100.0, 1)
        assert o.E_train == pytest.approx(K.TIER_POWER_W[o.tier] * o.T_train_true)
        assert o.E_up == pytest.approx(K.UPLOAD_POWER_W * o.T_up_true)


def _rows(run):
    with (run / "dispatch_log.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_cpu_run_logs_gpu_ground_truth_energy():
    run = RUNS_DIR / "S1_cpu_s4126_smoke2"
    if not (run / "dispatch_log.csv").exists():
        pytest.skip("re-smoke not run")
    for r in _rows(run):
        assert float(r["E_train"]) == pytest.approx(
            K.TIER_POWER_W[r["tier"]] * float(r["T_train_true"]))


# --- Tests 1 and 4: on runs ---------------------------------------------------------

DETERMINISTIC = ("cycle", "round", "client_id", "t_dispatch", "duration",
                 "t_finish_clean", "accepted")


def _key(run):
    return [tuple(r[c] for c in DETERMINISTIC) for r in _rows(run)]


def test_epsilon_one_reproduces_the_random_arm():
    a, b = RUNS_DIR / "S1_random_s4126_eps1chk", RUNS_DIR / "S1_gpu_eps1_s4126_eps1chk"
    if not ((a / "dispatch_log.csv").exists() and (b / "dispatch_log.csv").exists()):
        pytest.skip("epsilon-1 check runs not present")
    assert len(_rows(a)) > 30
    assert _key(a) == _key(b)


def test_proportional_reproduces_the_prompt_b_gpu_log():
    old = RUNS_DIR / "S1_gpu_s4126_smoke"
    new = RUNS_DIR / "S1_gpu_prop_s4126_smoke2"
    if not ((old / "dispatch_log.csv").exists() and (new / "dispatch_log.csv").exists()):
        pytest.skip("runs not present")
    assert _key(old) == _key(new)
