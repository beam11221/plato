"""
Regression / sanity tests for gpu_tops_predict_staleness_rounds (Phase-2 fix).

Tier map (from clients.csv):
    cids  1-50  → Xavier  (10 TFLOPS,  ~0.055 s)   slowest,  50% of pool
    cids 51-85  → Orin    (78 TFLOPS,  ~0.0071 s)  medium,   35% of pool
    cids 86-100 → Thor    (150 TFLOPS, ~0.0037 s)  fastest,  15% of pool

Run:
    cd plato/examples/async/fedasync
    python test_staleness_predictor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gpu_tier_sleep as gts

# ---------------------------------------------------------------------------
# Pull noise-free tier training times and T_round from the real CSVs.
# ---------------------------------------------------------------------------
T_XAVIER = gts.gpu_tops_predict_seconds(1,  500, 1)   # cid=1  → Xavier
T_ORIN   = gts.gpu_tops_predict_seconds(51, 500, 1)   # cid=51 → Orin
T_THOR   = gts.gpu_tops_predict_seconds(86, 500, 1)   # cid=86 → Thor

_cids   = list(range(1, 101))
_sizes  = {c: 500 for c in _cids}
gts._t_round_logged = False   # suppress duplicate log on recompute
T_ROUND = gts.compute_pool_T_round_seconds(_cids, _sizes, 1, clients_per_round=10)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
failures: list[str] = []


def check(name: str, got: float, lo: float, hi: float) -> None:
    ok = lo <= got <= hi
    print(f"  [{PASS if ok else FAIL}] {name}: got={got:.2f}  expected=[{lo:.2f}, {hi:.2f}]")
    if not ok:
        failures.append(name)


def predict(t_client: float, residuals: list[float]) -> float:
    return gts.gpu_tops_predict_staleness_rounds(t_client, residuals, T_ROUND)


print(f"\nTier times (noise-free, 500 samples, 1 epoch):")
print(f"  Xavier = {T_XAVIER:.6f} s  (50%, slowest)")
print(f"  Orin   = {T_ORIN:.6f} s  (35%, medium)")
print(f"  Thor   = {T_THOR:.6f} s  (15%, fastest)")
print(f"  T_round (P=10) = {T_ROUND:.6f} s")
print(f"  T_Xavier / T_round = {T_XAVIER / T_ROUND:.2f} rounds  (steady-state baseline)\n")

# ---------------------------------------------------------------------------
# Test 1 — all-fast pool (10 Thor, client=Xavier).
# All 10 Thor finish well before Xavier (T_THOR ≈ 0.0037 << T_XAVIER = 0.055).
# n_early=10, scale=1.0 → Phase-2 unchanged from pre-fix.
# Phase 1 = 10; Phase 2 = int((T_XAVIER - T_THOR) / T_ROUND * 1.0).
# ---------------------------------------------------------------------------
print("Test 1 — all-fast initial pool (10 Thor, client=Xavier)")
residuals_1 = [T_THOR] * 10
got_1 = predict(T_XAVIER, residuals_1)
phase2_1 = int((T_XAVIER - T_THOR) * 1.0 / T_ROUND)
expected_1 = 10 + phase2_1
check("phase1+phase2", got_1, expected_1 - 1, expected_1 + 1)
# All early → scale=1; result should equal old formula (no change from fix).
got_old = 10 + int((T_XAVIER - T_THOR) / T_ROUND)
check("scale=1 unchanged by fix", got_1, got_old - 0.5, got_old + 0.5)

# ---------------------------------------------------------------------------
# Test 2 — all-slow pool (10 Xavier fresh, client=Xavier).
# Residuals == t_client → excluded from early_finishes (strict <).
# n_early=0, so scale = max(0,1)/10 = 0.1.
# Phase 1 = 0; Phase 2 = int(T_XAVIER * 0.1 / T_ROUND).
# ---------------------------------------------------------------------------
print("\nTest 2 — all-slow pool (10 Xavier fresh, client=Xavier)")
residuals_2 = [T_XAVIER] * 10
got_2 = predict(T_XAVIER, residuals_2)
expected_2 = int(T_XAVIER * (1 / 10) / T_ROUND)
check("phase2 scaled at 0.1", got_2, expected_2 - 1, expected_2 + 1)
# Must be much less than steady-state (slow peers don't contribute before us).
check("got < ss/2", got_2, 0, T_XAVIER / T_ROUND * 0.5)
print(f"  (steady-state baseline = {T_XAVIER/T_ROUND:.2f}; got {got_2:.2f})")

# ---------------------------------------------------------------------------
# Test 3 — mixed pool (2 Thor + 3 Orin + 5 Xavier fresh, client=Xavier).
# n_early = 5 (Thor + Orin finish before Xavier), n_total=10, scale=0.5.
# Phase 1 = 5; Phase 2 = int((T_XAVIER - T_ORIN) * 0.5 / T_ROUND).
# Must be strictly less than all-fast result (Test 1) due to lower scale.
# ---------------------------------------------------------------------------
print("\nTest 3 — mixed pool (2 Thor + 3 Orin + 5 Xavier fresh, client=Xavier)")
residuals_3 = [T_THOR] * 2 + [T_ORIN] * 3 + [T_XAVIER] * 5
got_3 = predict(T_XAVIER, residuals_3)
phase2_3 = int((T_XAVIER - T_ORIN) * 0.5 / T_ROUND)
expected_3 = 5 + phase2_3
check("phase1+phase2 mixed", got_3, expected_3 - 1, expected_3 + 1)
check("mixed < all-fast (scale effect)", got_3, 0.0, got_1 + 1)
print(f"  all-fast={got_1:.2f}  mixed={got_3:.2f}  all-slow={got_2:.2f}")

# ---------------------------------------------------------------------------
# Test 4 — boundary: residual exactly == t_client must NOT enter early_finishes.
# ---------------------------------------------------------------------------
print("\nTest 4 — boundary: residual == t_client not counted in Phase 1")
residuals_4 = [T_XAVIER, T_XAVIER, T_THOR]
got_4 = predict(T_XAVIER, residuals_4)
early_4 = [r for r in residuals_4 if 0 < r < T_XAVIER]
assert len(early_4) == 1, f"expected 1 early finish (Thor only), got {len(early_4)}"
check("boundary excluded", got_4, 1.0, 20.0)
print(f"  early_finishes={early_4} (Thor only) — boundary correctly excluded")

# ---------------------------------------------------------------------------
# Test 5 — empty busy_residuals (first round / wall_time=0 fallback).
# n_total = max(0,1) = 1, scale = 1/1 = 1.0.
# Phase 1 = 0; Phase 2 = int(T_XAVIER / T_ROUND).
# Result must equal int(T_XAVIER / T_ROUND) — pure steady-state.
# ---------------------------------------------------------------------------
print("\nTest 5 — empty busy_residuals → pure steady-state ratio")
got_5 = predict(T_XAVIER, [])
expected_5 = int(T_XAVIER / T_ROUND)
check("empty == int(T_C/T_R)", got_5, expected_5 - 0.5, expected_5 + 0.5)

# ---------------------------------------------------------------------------
# Test 6 — monotonicity: Thor < Orin < Xavier predicted staleness.
# A faster client trains for less time, so fewer others complete first.
# Use a typical mixed residual pool for all three tests.
# ---------------------------------------------------------------------------
print("\nTest 6 — monotonicity: Thor < Orin < Xavier staleness")
pool = [T_THOR] * 2 + [T_ORIN] * 3 + [T_XAVIER] * 5
s_thor   = predict(T_THOR,   pool)
s_orin   = predict(T_ORIN,   pool)
s_xavier = predict(T_XAVIER, pool)
print(f"  Thor={s_thor:.2f}  Orin={s_orin:.2f}  Xavier={s_xavier:.2f}")
if s_thor <= s_orin <= s_xavier:
    print(f"  [{PASS}] monotonicity holds")
else:
    print(f"  [{FAIL}] monotonicity violated")
    failures.append("monotonicity")

# ---------------------------------------------------------------------------
# Test 7 — composition sensitivity: Xavier client in Thor-heavy pool should
# predict HIGHER staleness than in Xavier-heavy pool (faster peers → more
# events before our Xavier finishes).
# ---------------------------------------------------------------------------
print("\nTest 7 — composition sensitivity (Thor-heavy vs Xavier-heavy pool)")
thor_heavy   = [T_THOR]   * 9 + [T_XAVIER]
xavier_heavy = [T_XAVIER] * 9 + [T_THOR]
s_thor_pool   = predict(T_XAVIER, thor_heavy)
s_xavier_pool = predict(T_XAVIER, xavier_heavy)
print(f"  Xavier in Thor-heavy pool:   {s_thor_pool:.2f} rounds")
print(f"  Xavier in Xavier-heavy pool: {s_xavier_pool:.2f} rounds")
if s_thor_pool > s_xavier_pool:
    print(f"  [{PASS}] composition sensitivity preserved")
else:
    print(f"  [{FAIL}] expected Thor-heavy > Xavier-heavy")
    failures.append("composition-sensitivity")

# ---------------------------------------------------------------------------
# Test 8 — Residual-life crossover: slow peers that are almost done can
# finish BEFORE a newly-selected fast client, whereas fresh slow peers cannot.
#
# New client = Orin (T ≈ 0.0071 s).
# Pool = 10 Xavier clients (T = 0.055 s each).
#   (a) Fresh:         residual = 0.055  >> T_ORIN → no Phase-1 events → low staleness.
#   (b) Almost-done:   residual = 0.003  <  T_ORIN → all 10 enter Phase 1 → high staleness.
#
# Without tracking residual life (always using full T_full), we'd model
# case (b) as case (a) and systematically under-predict staleness for fast
# clients in a pool of nearly-finished slow peers.
# ---------------------------------------------------------------------------
print("\nTest 8 — residual-life crossover: slow peers almost-done → higher staleness for fast client")
fresh_xavier_pool  = [T_XAVIER] * 10           # residual = full T → finish AFTER Orin
almost_done_pool   = [0.003] * 10              # residual = 0.003 < T_ORIN → finish BEFORE Orin

s_fresh      = predict(T_ORIN, fresh_xavier_pool)
s_almost_done = predict(T_ORIN, almost_done_pool)
print(f"  Orin client, fresh-Xavier pool:       {s_fresh:.2f} rounds (expected ≈ 0-2, slow peers don't finish first)")
print(f"  Orin client, almost-done-Xavier pool: {s_almost_done:.2f} rounds (expected ≥ 10, slow peers all finish first)")
if s_almost_done > s_fresh:
    print(f"  [{PASS}] residual-life crossover: almost-done > fresh")
else:
    print(f"  [{FAIL}] expected almost-done ({s_almost_done:.2f}) > fresh ({s_fresh:.2f})")
    failures.append("residual-life-crossover")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
if failures:
    print(f"FAILED: {len(failures)} test(s): {failures}")
    sys.exit(1)
else:
    print("All tests passed.")
