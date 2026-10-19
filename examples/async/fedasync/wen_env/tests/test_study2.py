"""Prompt I (Study 2 plug-ins), Steps 0b.2, 0c and 3.

Unit tests: test 2 (Stage 0 in a host rejects exactly what the gpu + Stage 0
arm's Stage 0 rejected, on that run's logged states), test 3 (Stage 0 refuses a
non-eligible vehicle), test 4 (GPU F1 / F2 against a hand calculation).
Run-based tests (skip until the runs exist): test 1, test 5, 0b.2, 0c.
"""

from __future__ import annotations

import csv
import math
import re
from types import SimpleNamespace

import pytest

import wen_constants as K
import wen_fresh
from conftest import BUNDLE_DIR, FLEET_DIR, RUNS_DIR, TIERS_CSV
from wen_estimators import CpuEstimator, GpuEstimator
from wen_fleet import Fleet
from wen_ptable import run_dir
from wen_reported import ReportedVehicle, ReportedView
from wen_selection import WenStage0Gate
from wen_trace import TraceIndex, load_speeds, veh_id

BITS = 357_566_784
DETERMINISTIC = ("cycle", "round", "client_id", "t_dispatch", "duration",
                 "t_finish_clean", "accepted")


def _rows(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _keys(run, max_round):
    return [tuple(r[c] for c in DETERMINISTIC) for r in _rows(run / "dispatch_log.csv")
            if int(r["round"]) <= max_round]


class _Inner:
    def __init__(self):
        self.seen = None

    def select_clients(self, pool, count, context):
        self.seen = list(pool)
        return []


# --- Test 2: Stage 0 inside a host = the gpu + Stage 0 arm's Stage 0 ------------

STAGE0_LINE = re.compile(r"\[WenStage0\] round=(\d+) t=([\d.]+) estimator=gpu pool=(\d+) "
                         r"feasible=(\d+) rejected=\[([\d, ]*)\] fallback=(\d)")


def _in_flight_intervals(run):
    """(client, dispatch cycle, merge cycle): the client is busy at every cycle
    strictly between them (a merge logged with cycle m precedes cycle m's selection)."""
    disp, merges = _rows(run / "dispatch_log.csv"), _rows(run / "merge_log.csv")
    merge_cycles = {}
    for m in merges:
        merge_cycles.setdefault(int(m["client_id"]), []).append(int(m["cycle"]))
    out, k = [], {}
    for d in disp:
        c = int(d["client_id"])
        i = k.get(c, 0)
        k[c] = i + 1
        cm = merge_cycles.get(c, [])
        out.append((c, int(d["cycle"]), cm[i] if i < len(cm) else math.inf))
    return out


def test_stage0_plugin_matches_gpu_stage0_on_logged_states():
    run = run_dir("S2_gpu_stage0", 4126)
    logs = list(run.glob("server_*.log"))
    if not (run / "dispatch_log.csv").exists() or not logs:
        pytest.skip("S2 gpu + Stage 0 seed 4126 run not present")
    speeds = load_speeds(BUNDLE_DIR)
    trace = TraceIndex.from_bundle(BUNDLE_DIR, speeds)
    fleet = Fleet.load(FLEET_DIR, TIERS_CSV)
    view = ReportedView(position=lambda c, t: trace.x_at(veh_id(c), t),
                        speed=lambda c: speeds[veh_id(c)],
                        spec=lambda c: (fleet.spec(c).tier_id, fleet.spec(c).tflops_fp16,
                                        fleet.spec(c).cpu_ghz),
                        n_samples=K.partition_size)
    cycle_rows = _rows(run / "cycle_log.csv")
    cycles = {(int(r["round"]), f"{float(r['t_clean']):.3f}"): int(r["cycle"])
              for r in cycle_rows}
    t_of = {int(r["cycle"]): float(r["t_clean"]) for r in cycle_rows}
    intervals = _in_flight_intervals(run)
    lines = [m for line in logs[0].read_text(errors="replace").splitlines()
             if (m := STAGE0_LINE.search(line))][:300]
    assert len(lines) == 300
    est = GpuEstimator(K.FLOPS_FWD_PER_IMAGE, BITS)
    checked = 0
    for m in lines:
        rnd, t_txt = int(m.group(1)), m.group(2)
        cycle = cycles[(rnd, t_txt)]
        t = t_of[cycle]
        busy = {c for c, start, end in intervals if start < cycle < end}
        eligible = [c for c in range(1, 31) if c not in busy
                    and K.S2_COVERAGE_M[0] <= view.report(c, t).x <= K.S2_COVERAGE_M[1]]
        assert len(eligible) == int(m.group(3)), (cycle, eligible)
        server = SimpleNamespace(sumo_time=lambda t=t: t,
                                 wen_report=lambda c, tt: view.report(c, tt),
                                 wen_is_eligible=lambda c, tt, e=set(eligible): c in e,
                                 current_round=rnd)
        gate = WenStage0Gate(_Inner(), est, scenario="S2")
        gate.select_clients(eligible, 1, SimpleNamespace(server=server))
        rejected = sorted(c for c, ok in gate.last_pass.items() if not ok)
        logged = sorted(int(x) for x in m.group(5).split(",") if x.strip())
        assert rejected == logged, (cycle, rejected, logged)
        assert gate.last_stats["fallback"] == int(m.group(6))
        checked += 1
    assert checked == 300


# --- Test 3: Stage 0 never receives a non-eligible vehicle ------------------------

def test_stage0_refuses_a_non_eligible_vehicle():
    rv = lambda c, t: ReportedVehicle(c, t, 500.0, 25.0, "orin", 78.0, 2.2, 1667)  # noqa: E731
    server = SimpleNamespace(sumo_time=lambda: 10.0, wen_report=rv, current_round=3,
                             wen_is_eligible=lambda c, t: c != 5)
    gate = WenStage0Gate(_Inner(), GpuEstimator(K.FLOPS_FWD_PER_IMAGE, BITS))
    with pytest.raises(RuntimeError, match="non-eligible"):
        gate.select_clients([4, 5, 6], 1, SimpleNamespace(server=server))
    inner = _Inner()
    WenStage0Gate(inner, GpuEstimator(K.FLOPS_FWD_PER_IMAGE, BITS)).select_clients(
        [4, 6], 1, SimpleNamespace(server=server))
    assert inner.seen == [4, 6]


def test_stage0_in_s1_has_no_exit_boundary():
    rv = lambda c, t: ReportedVehicle(c, t, 1990.0, 30.0, "xavier", 10.0, 2.2, 1667)  # noqa: E731
    server = SimpleNamespace(sumo_time=lambda: 0.0, wen_report=rv, current_round=1,
                             wen_is_eligible=lambda c, t: True)
    est = GpuEstimator(K.FLOPS_FWD_PER_IMAGE, BITS)
    s1 = WenStage0Gate(_Inner(), est, scenario="S1")
    s1.select_clients([1, 2], 1, SimpleNamespace(server=server))
    assert all(s1.last_pass.values()) and s1.last_stats["n_rejected"] == 0
    s2 = WenStage0Gate(_Inner(), est, scenario="S2")
    s2.select_clients([1, 2], 1, SimpleNamespace(server=server))
    assert s2.last_stats["n_rej_time"] == 2                 # past the 1,600 m exit


# --- Test 4: GPU F1 / F2 against a hand calculation --------------------------------

def _hand_gpu(tier, n, x, v):
    tflops = {"xavier": 10.0, "orin": 78.0, "thor": 150.0}[tier]
    power = {"xavier": 40.0, "orin": 75.0, "thor": 130.0}[tier]
    t_train = 3 * 556_651_520 * 2 * n * 1 / (0.2 * tflops * 1e12)
    x_hat = (x + v * t_train) % 2000.0
    d = math.sqrt((x_hat - 1000.0) ** 2 + 10.0 ** 2)
    p_w, n_w = 10 ** (23 / 10) / 1000, 10 ** (-114 / 10) / 1000
    rate = 2e6 * math.log2(1 + d ** -3 * 1.0 * p_w / n_w)
    t_up = 357_566_784 / rate
    return t_train + t_up, power * t_train + 0.2 * t_up


@pytest.mark.parametrize("tier,n", [("xavier", 1667), ("orin", 1667), ("thor", 1666)])
@pytest.mark.parametrize("x", [500.0, 1000.0, 1500.0])
def test_gpu_f1_f2_match_hand_calculation(tier, n, x, tmp_path):
    v = 25.0
    spec = {"xavier": (10.0, 2.2), "orin": (78.0, 2.2), "thor": (150.0, 2.6)}[tier]

    class Server:
        training_clients = {}
        current_round = 1
        global_version = 0

        def sumo_time(self):
            return 0.0

        def wen_report(self, cid, t):
            return ReportedVehicle(cid, t, x, v, tier, spec[0], spec[1], n)

    sel = wen_fresh.WenFreshSelection(
        fresh=SimpleNamespace(), stale_f3_test=True,
        estimator=GpuEstimator(K.FLOPS_FWD_PER_IMAGE, BITS), t_max=13.56, e_max=113.5,
        kl={1: 0.5}, out_dir=tmp_path)
    obj, *_ = sel.objectives(Server(), [1], 0.0)
    f1, f2 = _hand_gpu(tier, n, x, v)
    assert obj[1][0] == pytest.approx(f1, rel=1e-9)
    assert obj[1][1] == pytest.approx(f2, rel=1e-9)
    cpu = wen_fresh.WenFreshSelection(
        fresh=SimpleNamespace(), stale_f3_test=True,
        estimator=CpuEstimator(K.FLOPS_FWD_PER_IMAGE, BITS), t_max=34.3, e_max=36.4,
        kl={1: 0.5}, out_dir=tmp_path / "cpu")
    assert cpu.objectives(Server(), [1], 0.0)[0][1][0] != obj[1][0]   # the plug-in swaps F1


# --- Run-based: Step 0b.2, Step 0c, Step 3 tests 1 and 5 ---------------------------

REPLAY_CELLS = ("S1_gpu", "S1_cpu", "S2_gpu_stage0", "S1_cpu_cal", "S1_wen", "S1_wen_fresh")


@pytest.mark.parametrize("cell", REPLAY_CELLS)
def test_replay_200_rounds_matches_existing_run(cell):
    new = RUNS_DIR / f"{cell}_s4126_replay200"
    if not (new / "dispatch_log.csv").exists():
        pytest.skip("replay run not present")
    a, b = _keys(new, 200), _keys(run_dir(cell, 4126), 200)
    assert len(a) >= 200 and a == b


def test_stage0_is_a_noop_in_s1():
    new = RUNS_DIR / "S1_gpu_stage0_s4126_noop200"
    if not (new / "dispatch_log.csv").exists():
        pytest.skip("S1 gpu + Stage 0 run not present")
    st = _rows(new / "stage0_log.csv")
    assert st and all(int(r["n_rejected"]) == 0 and r["fallback"] == "0" for r in st)
    assert _keys(new, 200) == _keys(run_dir("S1_gpu", 4126), 200)


@pytest.mark.parametrize("new,host", [("S2_random_s0_pluginoff_s4126", ("S2_random", 4126)),
                                      ("S1_wen_fresh_gpu_pluginoff_s4126",
                                       ("S1_wen_fresh", 4126))])
def test_plugin_off_reproduces_host(new, host):
    run = RUNS_DIR / new
    if not (run / "dispatch_log.csv").exists():
        pytest.skip("plug-in-off test run not present")
    # Compare every dispatch cycle the 100-round run has. Matching on round <= 100
    # would be wrong in S2: when round 100 is an all-lost cycle, the round number is
    # held and the longer run dispatches again under round 100 after the short run
    # has hit its round cap.
    a = _keys(run, 10**9)
    last_cycle = max(int(k[0]) for k in a)
    b = [k for k in _keys(run_dir(*host), 10**9) if int(k[0]) <= last_cycle]
    assert len(a) >= 100 and a == b


def test_determinism_two_100_round_wen_fresh_gpu_runs():
    a, b = RUNS_DIR / "S1_wen_fresh_gpu_s4126_det1", RUNS_DIR / "S1_wen_fresh_gpu_s4126_det2"
    if not ((a / "dispatch_log.csv").exists() and (b / "dispatch_log.csv").exists()):
        pytest.skip("wen_fresh_gpu determinism runs not present")
    assert _keys(a, 10**9) == _keys(b, 10**9) and len(_keys(a, 10**9)) > 100
    for f in ("wen_decisions.csv", "wen_candidates.csv"):
        assert _rows(a / f) == _rows(b / f)
