"""Ground truth / reported / estimated separation (audit G item; Prompt I Step 0b.3).

Two static checks over the estimator and selector side, including wen_fresh's
fresh-loss path and Stage 0:

1. Import graph: no estimator / selector module reaches wen_truth, wen_fleet or
   wen_trace, directly or through other repo modules it imports.
2. Field access: none of them reads a ground-truth field (the DispatchOutcome
   fields, the jitter, the server's in-flight outcome tables) by attribute or
   by subscript.

The one intended hand-over -- gpu_oracle's true-training-time callable -- is
built in wen_fedasync.py, outside the checked modules, and is the only use of
`truth.t_train_true` in the run path (checked too).
"""

from __future__ import annotations

import ast
import re

import pytest

from conftest import FEDASYNC, WEN

SELECTOR_SIDE = {
    "wen_estimators": WEN / "wen_estimators.py",
    "wen_reported": WEN / "wen_reported.py",
    "wen_selection": WEN / "wen_selection.py",
    "wen_egreedy": WEN / "wen_egreedy.py",
    "wen_faithful": WEN / "wen_faithful.py",
    "wen_fresh": WEN / "wen_fresh.py",
    "wen_polaris": WEN / "wen_polaris.py",
    "wen_formulas": WEN / "wen_formulas.py",
    "wen_partition": WEN / "wen_partition.py",
    "fedasync_polaris_vi_selection": FEDASYNC / "fedasync_polaris_vi_selection.py",
}
GROUND_TRUTH_MODULES = {"wen_truth", "wen_fleet", "wen_trace"}
# DispatchOutcome fields that are not also reported/estimated quantities, the
# fleet jitter, and the server's tables of outcomes. (`x_finish` is left out:
# the estimators' own Estimate.x_finish is the PREDICTED position, a different
# object with the same field name.)
GROUND_TRUTH_FIELDS = {"T_train_true", "d_finish", "chi2", "T_up_true",
                       "t_finish_clean", "E_train", "E_up", "jitter", "_eps",
                       "t_train_true", "wen_truth", "_in_flight", "_reported",
                       "TraceIndex", "GroundTruth", "DispatchOutcome"}
# Repo modules whose own imports are followed (resolved by file name).
SEARCH_DIRS = [WEN, FEDASYNC, FEDASYNC / "gpu_tiers", FEDASYNC / "gpu_tiers_dataset",
               FEDASYNC / "sumo_close_loop"]


def _imports(path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def _resolve(name):
    for d in SEARCH_DIRS:
        if (d / f"{name}.py").is_file():
            return d / f"{name}.py"
    return None


def _closure(path) -> set[str]:
    seen, todo = set(), [path]
    while todo:
        p = todo.pop()
        for name in _imports(p):
            if name in seen:
                continue
            seen.add(name)
            q = _resolve(name)
            if q is not None:
                todo.append(q)
    return seen


@pytest.mark.parametrize("name", sorted(SELECTOR_SIDE))
def test_selector_side_import_graph_avoids_ground_truth(name):
    reached = _closure(SELECTOR_SIDE[name])
    assert not reached & GROUND_TRUTH_MODULES, (name, reached & GROUND_TRUTH_MODULES)


# wen_formulas is the shared physics both sides evaluate (pure functions; its
# `chi2` is a parameter the estimators pass as 1.0), so only its imports are checked.
FIELD_CHECKED = sorted(set(SELECTOR_SIDE) - {"wen_formulas"})


@pytest.mark.parametrize("name", FIELD_CHECKED)
def test_selector_side_reads_no_ground_truth_field(name):
    tree = ast.parse(SELECTOR_SIDE[name].read_text())
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            used.add(node.value)                        # r["T_train_true"], getattr(x, "...")
    assert not used & GROUND_TRUTH_FIELDS, (name, used & GROUND_TRUTH_FIELDS)


def test_the_only_ground_truth_handover_is_gpu_oracle():
    src = (WEN / "wen_fedasync.py").read_text()
    uses = re.findall(r".*truth\.t_train_true.*", src)
    assert len(uses) == 1 and 'arm == "gpu_oracle"' in uses[0]


def test_stage0_and_fresh_loss_read_reported_records_only():
    """Stage 0 reads ReportedVehicle via server.wen_report; the fresh loss reads
    the global model and the training images -- neither touches outcomes."""
    sel = (WEN / "wen_selection.py").read_text()
    assert "server.wen_report(" in sel
    fresh = ast.parse((WEN / "wen_fresh.py").read_text())
    server_attrs = {n.attr for n in ast.walk(fresh)
                    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                    and n.value.id == "server"}
    assert server_attrs <= {"global_version", "require_trainer", "current_round",
                            "sumo_time"}, server_attrs


def test_polaris_reads_only_observed_quantities():
    """Polaris sees G_i (aggregation state), its own observed staleness, the
    reported partition sizes and the clock -- no estimator, position or tier."""
    src = (WEN / "wen_polaris.py").read_text()
    tree = ast.parse(src)
    server_attrs = {n.attr for n in ast.walk(tree)
                    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                    and n.value.id == "server"}
    assert server_attrs <= {"sumo_time"}, server_attrs
    assert not _imports(WEN / "wen_polaris.py") & {"wen_estimators", "wen_reported",
                                                    "wen_formulas", "wen_selection"}
