"""Prompt M A2: the mechanism log is passive and says what the selectors used.

Unit checks run always; the run checks read the 100-round check runs under
runs/wen_env/<cell>_pm100/ (skipped when absent).
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import RUNS_DIR, WEN
from test_wen_separation import (
    GROUND_TRUTH_FIELDS,
    GROUND_TRUTH_MODULES,
    SELECTOR_SIDE,
    _closure,
)

CHECK_SUFFIX = "_pm100"


# -- passive and separate ------------------------------------------------------

def test_no_selector_imports_the_mechanism_log():
    for name, path in SELECTOR_SIDE.items():
        assert "wen_mechanism_log" not in _closure(path), name


def test_mechanism_log_reads_no_ground_truth():
    path = WEN / "wen_mechanism_log.py"
    assert not _closure(path) & GROUND_TRUTH_MODULES
    import ast
    used = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            used.add(node.value)
    assert not used & GROUND_TRUTH_FIELDS, used & GROUND_TRUTH_FIELDS


def test_passive_g_equals_ours_tracker():
    """PassiveG reproduces FedAsyncPolarisAggregationStrategy's G_i (alpha = 1)."""
    from fedasync_polaris_vi_server import FedAsyncPolarisAggregationStrategy
    from wen_mechanism_log import PassiveG

    gen = torch.Generator().manual_seed(0)
    base = {"conv1.weight": torch.randn(4, 3, 3, 3, generator=gen),
            "fc.weight": torch.randn(2, 4, generator=gen)}
    real = FedAsyncPolarisAggregationStrategy(alpha=1.0)
    ctx = SimpleNamespace(total_clients=30, state={})
    real.total_clients = 30
    real.unexplored_clients = set(range(30))
    ctx.state["polaris"] = {"local_gradient_bounds": np.full(30, 0.5)}
    passive = PassiveG()
    for batch in ([3], [7, 12], [3, 29]):
        ups = [SimpleNamespace(client_id=c, staleness=1) for c in batch]
        ws = [{k: v + 0.01 * c * torch.randn(v.shape, generator=gen)
               for k, v in base.items()} for c in batch]
        real._update_gradient_bounds(ups, base, ws, ctx)
        passive.update(ups, base, ws)
        assert np.array_equal(passive.G, ctx.state["polaris"]["local_gradient_bounds"])


def test_polaris_draw_u_repeats_the_choice():
    """The logged u reproduces Generator.choice's pick (one random() per choice)."""
    for seed in range(50):
        w = np.random.default_rng(seed).random(7)
        w /= w.sum()
        pick = np.random.default_rng([seed, 1, 0, 9]).choice(7, p=w)
        u = np.random.default_rng([seed, 1, 0, 9]).random()
        cdf = np.cumsum(w)
        cdf /= cdf[-1]
        assert pick == int(np.searchsorted(cdf, u, side="right"))


# -- on the 100-round check runs ---------------------------------------------------

def _rows(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _run(name):
    run = RUNS_DIR / f"{name}{CHECK_SUFFIX}"
    if not (run / "dispatch_log.csv").exists():
        pytest.skip(f"{name} not run")
    return run


def test_gpu_candidates_V_rank_and_passive_axes():
    rows = _rows(_run("S1_gpu_s4126") / "gpu_candidates.csv")
    assert rows
    for r in rows:
        G, tau, V = float(r["G"]), float(r["tau_hat"]), float(r["V"])
        assert math.isclose(V, G / math.sqrt(1 + max(tau, 0.0)), rel_tol=1e-12)
        # In the gpu arm the passive axes are the arm's own values.
        assert float(r["tau_hat_ours"]) == tau
        assert float(r["G_ours"]) == G
        assert float(r["T_hat_total_gpu"]) == float(r["T_hat_total"])
        if r["picked"] == "1" and r["explored"] == "0":
            assert r["rank_by_V"] == "1"


def test_gpu_candidates_match_the_dispatch_log():
    run = _run("S1_gpu_s4126")
    picked = [r for r in _rows(run / "gpu_candidates.csv") if r["picked"] == "1"]
    disp = _rows(run / "dispatch_log.csv")
    assert [int(r["client"]) for r in picked] == [int(r["client_id"]) for r in disp]
    for p, d in zip(picked, disp):
        assert float(p["V"]) == float(d["V_pick"])
        assert float(p["tau_hat"]) == float(d["tau_hat"])


def test_polaris_q_sums_to_one_and_picks_match():
    run = _run("S1_polaris_s4126")
    by_dec = defaultdict(list)
    for r in _rows(run / "polaris_candidates.csv"):
        by_dec[r["decision_id"]].append(r)
    for rows in by_dec.values():
        if rows[0]["slot_k"] == "0":
            assert math.isclose(sum(float(r["q_i"]) for r in rows), 1.0, abs_tol=1e-9)
        assert math.isclose(sum(float(r["q_renormalised_at_pick"]) for r in rows), 1.0,
                            abs_tol=1e-9)
        assert sum(r["picked"] == "1" for r in rows) == 1
    picks = [int(r["client"]) for d in sorted(by_dec, key=int) for r in by_dec[d]
             if r["picked"] == "1"]
    assert picks == [int(r["client_id"]) for r in _rows(run / "dispatch_log.csv")]


def test_wen_pick_has_minimum_score_in_pareto():
    run = _run("S1_wen_fresh_s4126")
    by_dec = defaultdict(list)
    for r in _rows(run / "wen_candidates.csv"):
        by_dec[r["decision"]].append(r)
    assert by_dec
    for rows in by_dec.values():
        pareto = [r for r in rows if r["in_pareto"] == "1"]
        pick = [r for r in rows if r["picked"] == "1"]
        assert len(pick) == 1 and pick[0]["in_pareto"] == "1"
        best = min((float(r["score"]), int(r["client_id"])) for r in pareto)
        assert (float(pick[0]["score"]), int(pick[0]["client_id"])) == best
        assert pick[0]["rank_by_score"] == "1"
    passive = _rows(run / "passive_candidates.csv")
    assert {(r["decision_id"], r["client"]) for r in passive} == \
        {(r["decision"], r["client_id"]) for rows in by_dec.values() for r in rows}


def test_snapshot_does_not_redirect_the_scorer_lookup():
    """WenEnvMixin.polaris_selector() finds the scorer by `last_tau_hats`; the
    WenSelector snapshot must not carry that name, or install_wen would wire the
    estimator into the wrong object (seen in the first Prompt M check run)."""
    from wen_egreedy import WenSelector

    sel = WenSelector(None, "egreedy", 1.0, 4126)
    assert not hasattr(sel, "last_tau_hats")
    assert hasattr(sel, "last_call_tau_hats")
