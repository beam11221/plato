"""
Smoke tests for T_round computation, conversion utilities, and the
seconds-unit selection path.

Run from the fedasync directory:
    cd plato/examples/async/fedasync
    python test_t_round.py
    # or with pytest (if installed):
    python -m pytest test_t_round.py -v
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import MagicMock, patch

# Ensure local modules are importable when run directly.
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# gpu_tier_sleep.py lives in gpu_tiers/ alongside the CSVs it loads.
sys.path.insert(0, str(_HERE / "gpu_tiers"))

import numpy as np
import gpu_tier_sleep

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TIERS_CSV = _HERE / "gpu_tiers" / "input" / "tiers.csv"
_CLIENTS_CSV = _HERE / "gpu_tiers" / "input" / "clients.csv"

_TOTAL = 100
_PARTITION_SIZE = 500
_LOCAL_EPOCHS = 1

_CLIENT_IDS = list(range(1, _TOTAL + 1))
_PARTITION_SIZES = {cid: _PARTITION_SIZE for cid in _CLIENT_IDS}


def _make_computer(xavier_eta: float = 1.0) -> gpu_tier_sleep.GpuTierSleepComputer:
    """Return a GpuTierSleepComputer optionally with a modified Xavier η."""
    computer = gpu_tier_sleep.GpuTierSleepComputer(_TIERS_CSV, _CLIENTS_CSV)
    if xavier_eta != 1.0:
        orig = computer._tiers["xavier"]
        computer._tiers["xavier"] = gpu_tier_sleep.TierSpec(
            tier_id=orig.tier_id,
            gpu_tflops_fp16=orig.gpu_tflops_fp16,
            eta_nominal=xavier_eta,
            noise_sigma=orig.noise_sigma,
        )
    return computer


# ---------------------------------------------------------------------------
# Test 1: compute_pool_T_round_seconds returns a positive float in sane range
# ---------------------------------------------------------------------------

def test_t_round_positive_and_sane():
    """T_round for 50/35/15 split, |D|=500, E=1 must be in (1e-3, 10) seconds."""
    gpu_tier_sleep._t_round_logged = False

    computer = _make_computer()
    with patch.object(gpu_tier_sleep, "get_computer", return_value=computer):
        t_round = gpu_tier_sleep.compute_pool_T_round_seconds(
            _CLIENT_IDS, _PARTITION_SIZES, _LOCAL_EPOCHS
        )

    logging.info("[test1] T_round = %.9f s", t_round)
    assert isinstance(t_round, float), "T_round must be a float"
    assert t_round > 0, f"T_round must be positive, got {t_round}"
    assert 1e-3 < t_round < 10, (
        f"T_round={t_round:.6f} s is outside the sane range (1e-3, 10). "
        "Check W_MODEL_FLOPS, partition_size, and tier specs."
    )

    # Analytical check:
    # Xavier (50 clients): 1.1e9 * 500 / (10.0 * 1e12 * 1.0) = 0.055 s
    # Orin   (35 clients): 1.1e9 * 500 / (78.0 * 1e12 * 1.0) ≈ 0.007051 s
    # Thor   (15 clients): 1.1e9 * 500 / (150.0 * 1e12 * 1.0) ≈ 0.003667 s
    #
    # This call leaves clients_per_round at its default of 1, and BOTH T_round
    # models reduce to the pool ARITHMETIC MEAN at P=1: "min of 1 draw" is just
    # that draw, and "queue" divides by min(1, 100) = 1.  (This assertion used to
    # expect the pool median, from before the function took P into account at
    # all — it failed for that reason, not because of a defect in the code.)
    t_xavier = gpu_tier_sleep.W_MODEL_FLOPS * 500 / (10.0 * 1e12 * 1.0)
    t_orin = gpu_tier_sleep.W_MODEL_FLOPS * 500 / (78.0 * 1e12 * 1.0)
    t_thor = gpu_tier_sleep.W_MODEL_FLOPS * 500 / (150.0 * 1e12 * 1.0)
    expected = (50 * t_xavier + 35 * t_orin + 15 * t_thor) / 100.0
    assert abs(t_round - expected) < 1e-12, (
        f"T_round={t_round:.9f} s != expected={expected:.9f} s"
    )
    logging.info("[test1] PASS — T_round=%.6f s  expected=%.6f s", t_round, expected)


# ---------------------------------------------------------------------------
# Test 2: η sensitivity — Xavier η=0.5 doubles Xavier predictions
# ---------------------------------------------------------------------------

def test_t_round_eta_sensitivity():
    """When Xavier η halves, Xavier per-client times double; T_round changes."""
    computer_orig = _make_computer(xavier_eta=1.0)
    computer_mod = _make_computer(xavier_eta=0.5)

    gpu_tier_sleep._t_round_logged = False
    with patch.object(gpu_tier_sleep, "get_computer", return_value=computer_orig):
        t_round_orig = gpu_tier_sleep.compute_pool_T_round_seconds(
            _CLIENT_IDS, _PARTITION_SIZES, _LOCAL_EPOCHS
        )

    gpu_tier_sleep._t_round_logged = False
    with patch.object(gpu_tier_sleep, "get_computer", return_value=computer_mod):
        t_round_mod = gpu_tier_sleep.compute_pool_T_round_seconds(
            _CLIENT_IDS, _PARTITION_SIZES, _LOCAL_EPOCHS
        )

    # Xavier per-client time must be exactly 2× in the modified computer.
    with patch.object(gpu_tier_sleep, "get_computer", return_value=computer_orig):
        t1_orig = gpu_tier_sleep.gpu_tops_predict_seconds(1, _PARTITION_SIZE, _LOCAL_EPOCHS)
    with patch.object(gpu_tier_sleep, "get_computer", return_value=computer_mod):
        t1_mod = gpu_tier_sleep.gpu_tops_predict_seconds(1, _PARTITION_SIZE, _LOCAL_EPOCHS)

    ratio = t1_mod / t1_orig
    assert abs(ratio - 2.0) < 1e-9, (
        f"Xavier prediction ratio = {ratio:.6f}, expected 2.0  "
        f"(orig={t1_orig:.9f} s, mod={t1_mod:.9f} s)"
    )

    assert t_round_mod != t_round_orig, (
        "T_round did not change when Xavier η was halved "
        f"(both = {t_round_orig:.9f} s)"
    )
    logging.info(
        "[test2] PASS — Xavier ratio=%.6f  T_round orig=%.6f s  mod=%.6f s",
        ratio, t_round_orig, t_round_mod,
    )


# ---------------------------------------------------------------------------
# Test 3: seconds_to_rounds correctness
# ---------------------------------------------------------------------------

def test_seconds_to_rounds():
    """Verify the three contract cases and the ValueError guard."""
    T = 0.031

    r1 = gpu_tier_sleep.seconds_to_rounds(T, T)
    assert abs(r1 - 1.0) < 1e-12, f"seconds_to_rounds(T,T) = {r1}, expected 1.0"

    r2 = gpu_tier_sleep.seconds_to_rounds(0.0, T)
    assert r2 == 0.0, f"seconds_to_rounds(0, T) = {r2}, expected 0.0"

    r3 = gpu_tier_sleep.seconds_to_rounds(2 * T, T)
    assert abs(r3 - 2.0) < 1e-12, f"seconds_to_rounds(2T, T) = {r3}, expected 2.0"

    # Negative seconds → clipped to 0.0 (non-negative contract)
    r_neg = gpu_tier_sleep.seconds_to_rounds(-1.0, T)
    assert r_neg == 0.0, f"seconds_to_rounds(-1, T) = {r_neg}, expected 0.0"

    raised_zero = False
    try:
        gpu_tier_sleep.seconds_to_rounds(1.0, 0.0)
    except ValueError:
        raised_zero = True
    assert raised_zero, "Expected ValueError for T_round_seconds=0"

    raised_neg = False
    try:
        gpu_tier_sleep.seconds_to_rounds(1.0, -0.5)
    except ValueError:
        raised_neg = True
    assert raised_neg, "Expected ValueError for T_round_seconds<0"

    logging.info("[test3] PASS — seconds_to_rounds contracts verified")


# ---------------------------------------------------------------------------
# Test 4: gpu_tops_predict_seconds is deterministic (no noise applied)
# ---------------------------------------------------------------------------

def test_gpu_tops_predict_deterministic():
    """Even with noise_sigma > 0 in the tier spec, the function never applies noise."""
    computer = _make_computer()
    orig = computer._tiers["xavier"]
    computer._tiers["xavier"] = gpu_tier_sleep.TierSpec(
        tier_id=orig.tier_id,
        gpu_tflops_fp16=orig.gpu_tflops_fp16,
        eta_nominal=orig.eta_nominal,
        noise_sigma=0.5,   # large noise — must be ignored by gpu_tops_predict_seconds
    )

    with patch.object(gpu_tier_sleep, "get_computer", return_value=computer):
        r1 = gpu_tier_sleep.gpu_tops_predict_seconds(1, _PARTITION_SIZE, _LOCAL_EPOCHS)
        r2 = gpu_tier_sleep.gpu_tops_predict_seconds(1, _PARTITION_SIZE, _LOCAL_EPOCHS)

    assert r1 == r2, (
        f"gpu_tops_predict_seconds returned different values on two calls: "
        f"{r1} vs {r2} — noise is being applied (it must not be)"
    )

    expected = gpu_tier_sleep.W_MODEL_FLOPS * _PARTITION_SIZE / (10.0 * 1e12 * 1.0)
    assert abs(r1 - expected) < 1e-15, (
        f"gpu_tops_predict_seconds={r1:.9f} != noise-free expected={expected:.9f}"
    )
    logging.info("[test4] PASS — gpu_tops_predict_seconds is deterministic, r=%.9f s", r1)


# ---------------------------------------------------------------------------
# Test 5: End-to-end — selector with GPU-TOPS predictor and seconds unit
# ---------------------------------------------------------------------------

def _inject_plato_stubs() -> None:
    """Inject minimal plato stubs into sys.modules so selection imports succeed.

    Only needed when running this test file in isolation (without plato installed
    in the environment).  Each stub has only the interface actually exercised by
    fedasync_polaris_vi_selection and fedasync_selection.
    """
    if "plato" in sys.modules:
        return  # real plato is available — nothing to do

    class _StubRandomSelectionStrategy:
        """Stand-in for plato's RandomSelectionStrategy."""
        def setup(self, context):
            pass

        def select_clients(self, clients_pool, clients_count, context):
            return list(clients_pool)[:clients_count]

    # Build a hierarchy of stub modules that satisfy the import chain:
    #   plato
    #   plato.servers
    #   plato.servers.strategies
    #   plato.servers.strategies.base            → ServerContext (any type alias)
    #   plato.servers.strategies.client_selection
    #   plato.servers.strategies.client_selection.random_selection  → RandomSelectionStrategy
    def _mod(name: str) -> ModuleType:
        m = ModuleType(name)
        sys.modules[name] = m
        return m

    plato = _mod("plato")
    servers = _mod("plato.servers")
    strategies = _mod("plato.servers.strategies")
    base = _mod("plato.servers.strategies.base")
    base.ServerContext = type("ServerContext", (), {})  # type alias placeholder
    cs = _mod("plato.servers.strategies.client_selection")
    rand = _mod("plato.servers.strategies.client_selection.random_selection")
    rand.RandomSelectionStrategy = _StubRandomSelectionStrategy
    fedavg_srv = _mod("plato.servers.fedavg")

    plato.servers = servers
    servers.strategies = strategies
    strategies.base = base
    strategies.client_selection = cs
    cs.random_selection = rand


def test_selector_with_gpu_tops_predictor():
    """Instantiate selector with gpu_tops_predict_seconds and run select_clients."""
    _inject_plato_stubs()

    # Import after stubs are in place (may be a no-op if already imported).
    import fedasync_polaris_vi_selection  # noqa: PLC0415

    LOCAL_EPOCHS = 1
    N_CLIENTS = 10
    partition_sizes = {cid: _PARTITION_SIZE for cid in range(1, N_CLIENTS + 1)}
    t_round = 0.031

    computer = _make_computer()
    with patch.object(gpu_tier_sleep, "get_computer", return_value=computer):
        selector = fedasync_polaris_vi_selection.PolarisViClientSelectionStrategy(
            predict_staleness=lambda cid, ctx: gpu_tier_sleep.gpu_tops_predict_seconds(
                cid, partition_sizes[cid], LOCAL_EPOCHS
            ),
            predict_staleness_unit="seconds",
        )

    mock_server = SimpleNamespace(
        training_clients={},
        clients_per_round=N_CLIENTS,
        current_round=1,
    )
    G = np.full(N_CLIENTS, 0.5, dtype=float)
    mock_context = SimpleNamespace(
        server=mock_server,
        current_round=1,
        total_clients=N_CLIENTS,
        state={
            "T_round_seconds": t_round,
            "polaris": {
                "local_gradient_bounds": G,
                "local_stalenesses": np.full(N_CLIENTS, 0.01, dtype=float),
            },
        },
    )

    selector.setup(mock_context)

    clients_pool = list(range(1, N_CLIENTS + 1))
    clients_count = 5

    with patch.object(gpu_tier_sleep, "get_computer", return_value=computer):
        selected = selector.select_clients(clients_pool, clients_count, mock_context)

    assert isinstance(selected, list), f"Expected list, got {type(selected)}"
    assert len(selected) == clients_count, (
        f"Expected {clients_count} selected clients, got {len(selected)}: {selected}"
    )
    assert all(1 <= c <= N_CLIENTS for c in selected), (
        f"Selected clients out of range: {selected}"
    )
    logging.info("[test5] PASS — selected=%s (count=%d)", selected, len(selected))


# ---------------------------------------------------------------------------
# Test 6: CPU tiers — lower heterogeneity than GPU tiers
#
# tiers_cpu.csv encodes cpu_cores × cpu_freq_ghz / 1000 in gpu_tflops_fp16:
#   xavier → 0.016  orin → 0.024  thor → 0.0312
# Heterogeneity ratio (max/min predicted time) must be smaller than GPU's.
# ---------------------------------------------------------------------------

_TIERS_CPU_CSV = _HERE / "gpu_tiers" / "input" / "tiers_cpu.csv"


def test_cpu_tiers_lower_heterogeneity():
    """CPU tiers load correctly and have lower heterogeneity than GPU tiers."""
    gpu_computer = gpu_tier_sleep.GpuTierSleepComputer(_TIERS_CSV, _CLIENTS_CSV)
    cpu_computer = gpu_tier_sleep.GpuTierSleepComputer(_TIERS_CPU_CSV, _CLIENTS_CSV)

    # Noise-free training time for representative clients (500 samples, 1 epoch).
    def _predict(computer, cid):
        tier = computer.tier_for(cid)
        work = gpu_tier_sleep.W_MODEL_FLOPS * _PARTITION_SIZE * _LOCAL_EPOCHS
        return work / (tier.gpu_tflops_fp16 * 1e12 * tier.eta_nominal)

    gpu_times = {
        "xavier": _predict(gpu_computer, 1),   # cid 1  → xavier
        "orin":   _predict(gpu_computer, 51),  # cid 51 → orin
        "thor":   _predict(gpu_computer, 86),  # cid 86 → thor
    }
    cpu_times = {
        "xavier": _predict(cpu_computer, 1),
        "orin":   _predict(cpu_computer, 51),
        "thor":   _predict(cpu_computer, 86),
    }

    gpu_ratio = max(gpu_times.values()) / min(gpu_times.values())
    cpu_ratio = max(cpu_times.values()) / min(cpu_times.values())

    logging.info(
        "[test6] GPU tier times: xavier=%.6f orin=%.6f thor=%.6f  ratio=%.2f",
        gpu_times["xavier"], gpu_times["orin"], gpu_times["thor"], gpu_ratio,
    )
    logging.info(
        "[test6] CPU tier times: xavier=%.6f orin=%.6f thor=%.6f  ratio=%.2f",
        cpu_times["xavier"], cpu_times["orin"], cpu_times["thor"], cpu_ratio,
    )

    assert cpu_ratio < gpu_ratio, (
        f"CPU heterogeneity ratio {cpu_ratio:.2f} must be lower than GPU's {gpu_ratio:.2f}"
    )

    # Sanity-check expected values: make_cpu_tiers.py --metric freq writes
    # cpu_freq_ghz/1000, so Xavier and Orin (both 2.00 GHz) are IDENTICAL and
    # only Thor (2.60 GHz) is faster.
    work = gpu_tier_sleep.W_MODEL_FLOPS * _PARTITION_SIZE * _LOCAL_EPOCHS
    assert abs(cpu_times["xavier"] - work / (0.002 * 1e12)) < 1e-10
    assert abs(cpu_times["orin"]   - work / (0.002 * 1e12)) < 1e-10
    assert abs(cpu_times["thor"]   - work / (0.0026 * 1e12)) < 1e-10

    # CPU ratio ~1.30×; GPU ratio ~15× — assert CPU < 5× for a generous bound.
    assert cpu_ratio < 5.0, f"CPU ratio {cpu_ratio:.2f} unexpectedly large"

    logging.info("[test6] PASS — CPU ratio=%.2f < GPU ratio=%.2f", cpu_ratio, gpu_ratio)


def test_cpu_tiers_T_round_positive():
    """T_round computed from CPU tiers is positive and slower than GPU T_round."""
    gpu_computer = gpu_tier_sleep.GpuTierSleepComputer(_TIERS_CSV, _CLIENTS_CSV)
    cpu_computer = gpu_tier_sleep.GpuTierSleepComputer(_TIERS_CPU_CSV, _CLIENTS_CSV)

    gpu_tier_sleep._t_round_logged = False
    with patch.object(gpu_tier_sleep, "get_computer", return_value=gpu_computer):
        t_round_gpu = gpu_tier_sleep.compute_pool_T_round_seconds(
            _CLIENT_IDS, _PARTITION_SIZES, _LOCAL_EPOCHS
        )

    gpu_tier_sleep._t_round_logged = False
    with patch.object(gpu_tier_sleep, "get_computer", return_value=cpu_computer):
        t_round_cpu = gpu_tier_sleep.compute_pool_T_round_seconds(
            _CLIENT_IDS, _PARTITION_SIZES, _LOCAL_EPOCHS
        )

    assert t_round_cpu > 0, f"CPU T_round must be positive, got {t_round_cpu}"
    assert t_round_cpu > t_round_gpu, (
        f"CPU T_round ({t_round_cpu:.6f} s) should be slower than GPU T_round ({t_round_gpu:.6f} s)"
    )
    logging.info(
        "[test7] PASS — T_round_cpu=%.6f s > T_round_gpu=%.6f s",
        t_round_cpu, t_round_gpu,
    )


# ---------------------------------------------------------------------------
# Test 8: t_round_model defaults to "min" and rejects typos
# ---------------------------------------------------------------------------

def test_t_round_model_default_and_validation():
    """Absent config -> "min" (historical).  A typo raises instead of falling back."""
    gpu_tier_sleep._t_round_model = None
    assert gpu_tier_sleep.t_round_model() == "min", (
        "t_round_model() must default to 'min' so configs written before the key "
        "existed keep producing byte-identical runs."
    )

    def _resolve(raw):
        gpu_tier_sleep._t_round_model = None
        cfg = SimpleNamespace(config=SimpleNamespace(
            gpu_tier=SimpleNamespace(t_round_model=raw)))
        with patch.dict(sys.modules, {"plato.config": SimpleNamespace(
                Config=lambda: cfg)}):
            return gpu_tier_sleep.t_round_model()

    # Case and surrounding whitespace are normalised on purpose — a config is
    # hand-written and "Queue" is the same intent as "queue".
    for ok in ("queue", "QUEUE ", " Queue"):
        assert _resolve(ok) == "queue", f"t_round_model() rejected {ok!r}"
    assert _resolve("min") == "min"

    # Anything that is not one of the two models must raise.  A silent fallback
    # would put two arms of a sweep on different staleness scales with nothing in
    # the log to say so — the `staleness_predictor` typo trap (HANDOFF § 4.10).
    for bad in ("queue_", "steady-state", "e_min", ""):
        raised = False
        try:
            _resolve(bad)
        except ValueError:
            raised = True
        assert raised, f"t_round_model() silently accepted {bad!r}"

    gpu_tier_sleep._t_round_model = None
    logging.info("[test8] PASS — default 'min', case-insensitive, typos raise")


# ---------------------------------------------------------------------------
# Test 9: the two models, against the seconds/round the finished runs measured
# ---------------------------------------------------------------------------

def test_t_round_queue_matches_observed_rate():
    """"queue" reproduces the observed merge rate; "min" is wrong, gap-dependent.

    Ground truth is d(elapsed_time)/d(round) from the four finished gpu runs of
    the post-bugfix sweep, with the dispatch-weighted mean sleep of each run and
    the in-flight concurrency check_run.py measured (9 for all four).

    The point of the test is not just that "queue" is closer.  It is that the
    error of "min" is a monotone function of the gap — the sweep's own
    independent variable — so it cannot be treated as a constant that cancels.
    """
    # gap -> (per-tier sleep seconds, dispatch-weighted mean, measured s/round)
    observed = {
        "01": ((20.0, 20.0, 20.0), 20.000, 2.183),
        "10": ((20.0, 6.325, 2.000), 11.625, 1.336),
        "20": ((20.0, 4.472, 1.000), 10.302, 1.205),
        "30": ((20.0, 3.651, 0.667), 9.564, 1.147),
    }
    concurrency = 9  # measured, not configured: one of the 10 slots is mid-refill

    min_errors = []
    for gap, ((t_x, t_o, t_t), sel_mean, measured) in observed.items():
        pool = [t_x] * 50 + [t_o] * 35 + [t_t] * 15

        gpu_tier_sleep._t_round_model = "min"
        t_min = gpu_tier_sleep._t_round_kernel(pool, 10)

        # "queue" is evaluated on the live busy set, which is what the selector
        # actually passes it — the dispatch-weighted mean over the clients in
        # flight, divided by how many are in flight.
        gpu_tier_sleep._t_round_model = "queue"
        t_queue = gpu_tier_sleep._t_round_kernel([sel_mean] * concurrency, 10)

        err_min = abs(t_min / measured - 1.0)
        err_queue = abs(t_queue / measured - 1.0)
        min_errors.append((gap, err_min))

        assert err_queue < 0.10, (
            f"gap {gap}: 'queue' predicts {t_queue:.3f} s/round vs {measured:.3f} "
            f"measured ({err_queue:+.1%}) — expected within 10%."
        )
        assert err_queue < err_min, (
            f"gap {gap}: 'queue' ({err_queue:+.1%}) is not closer to the measured "
            f"{measured:.3f} s/round than 'min' ({err_min:+.1%})."
        )
        logging.info(
            "[test9] gap %s: measured=%.3f  min=%.3f (%+.1f%%)  queue=%.3f (%+.1f%%)",
            gap, measured, t_min, err_min * 100, t_queue, err_queue * 100,
        )

    # The error of "min" must shrink monotonically as the gap widens — that is
    # the confound.  At G=1 every client takes 20 s, so "min of 10" IS 20 s.
    errs = [e for _, e in min_errors]
    assert errs == sorted(errs, reverse=True), (
        f"'min' error is expected to decay monotonically with the gap, got "
        f"{[(g, f'{e:+.1%}') for g, e in min_errors]}"
    )
    assert errs[0] > 5.0, (
        f"'min' should be off by >500% at gap 01 (all clients identical), "
        f"got {errs[0]:+.1%}"
    )

    gpu_tier_sleep._t_round_model = None
    logging.info("[test9] PASS — queue within 10%% at every gap; min decays %s",
                 " -> ".join(f"{e:+.0%}" for e in errs))


# ---------------------------------------------------------------------------
# Runner (also works as a plain script without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_t_round_positive_and_sane,
        test_t_round_eta_sensitivity,
        test_seconds_to_rounds,
        test_gpu_tops_predict_deterministic,
        test_selector_with_gpu_tops_predictor,
        test_cpu_tiers_lower_heterogeneity,
        test_cpu_tiers_T_round_positive,
        test_t_round_model_default_and_validation,
        test_t_round_queue_matches_observed_rate,
    ]
    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:  # noqa: BLE001
            import traceback
            logging.error("FAIL %s: %s\n%s", fn.__name__, exc, traceback.format_exc())
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)
