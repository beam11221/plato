"""Step 8 tests 3 and 6, on finished runs (skipped when the runs are absent).

Produce the runs with `./wen_env/run_smoke.sh` from examples/async/fedasync.
"""

from __future__ import annotations

import csv

import pytest

import wen_clock_replay
from conftest import RUNS_DIR

SMOKE = sorted(RUNS_DIR.glob("*_smoke")) if RUNS_DIR.exists() else []
DETERMINISTIC_COLUMNS = (
    "cycle", "round", "client_id", "veh_id", "tier", "t_dispatch", "x_dispatch",
    "V_k", "T_train_true", "x_finish", "d_finish", "chi2", "T_up_true", "duration",
    "t_finish_clean", "accepted", "E_train", "E_up",
)


def _finished(run):
    return (run / "dispatch_log.csv").exists() and (run / "merge_log.csv").exists()


@pytest.mark.parametrize("run", [r for r in SMOKE if _finished(r)] or
                         [pytest.param(None, marks=pytest.mark.skip("no smoke runs"))],
                         ids=lambda r: getattr(r, "name", "none"))
def test_online_clock_equals_reconstruction(run):
    result = wen_clock_replay.replay(run)
    assert result["dispatches"] > 0
    assert result["max_abs_diff_s"] < 1e-6, result


def _dispatches(run):
    with (run / "dispatch_log.csv").open(newline="") as fh:
        return [tuple(r[c] for c in DETERMINISTIC_COLUMNS) for r in csv.DictReader(fh)]


def test_same_seed_gives_an_identical_dispatch_log():
    a, b = RUNS_DIR / "S1_random_s4126_det1", RUNS_DIR / "S1_random_s4126_det2"
    if not (_finished(a) and _finished(b)):
        pytest.skip("determinism repeats not run")
    rows_a, rows_b = _dispatches(a), _dispatches(b)
    assert len(rows_a) > 30
    assert rows_a == rows_b
