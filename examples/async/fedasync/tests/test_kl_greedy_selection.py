"""
Unit tests for KLGreedyClientSelectionStrategy.

Tests verify the KL(P || Q̃_i) math in isolation — no PLATO server,
no CSV, no network.  All tests call _kl_divergence() as a static method
so they exercise the exact computation path used at runtime.

Validation requirements (from task spec)
-----------------------------------------
1. Client with uniform label distribution across all K classes → KL < 0.01.
2. Client with all samples in one class → KL is large but FINITE, not NaN.
3. Two clients, one balanced and one skewed → KL(balanced) < KL(skewed).
4. KL is non-negative for all valid inputs (tested over many random draws).

Run
---
    cd plato/examples/async/fedasync
    python -m pytest tests/test_kl_greedy_selection.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the parent directory importable without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from fedasync_kl_greedy_selection import KLGreedyClientSelectionStrategy

# Convenience alias
_kl = KLGreedyClientSelectionStrategy._kl_divergence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_p(k: int) -> np.ndarray:
    """Return a uniform probability vector of length k."""
    return np.ones(k, dtype=np.float64) / k


def _balanced_counts(k: int, n: int) -> np.ndarray:
    """Return counts with n // k samples per class (as balanced as possible)."""
    counts = np.full(k, n // k, dtype=np.int64)
    counts[: n % k] += 1  # distribute remainder
    return counts


# ---------------------------------------------------------------------------
# Test 1 — uniform client → KL ≈ 0 (spec: < 0.01)
# ---------------------------------------------------------------------------

class TestUniformClientSmallKL:
    """Client whose local distribution matches the global uniform P."""

    @pytest.mark.parametrize("k,n", [(10, 500), (10, 100), (10, 1000), (5, 200)])
    def test_kl_below_threshold(self, k: int, n: int):
        p = _uniform_p(k)
        counts = _balanced_counts(k, n)
        kl = _kl(p, counts)
        assert kl >= 0.0, f"KL must be non-negative, got {kl}"
        assert kl < 0.01, (
            f"Uniform client (k={k}, n={n}) should have KL < 0.01, got {kl:.6f}. "
            "Check that smoothing is not over-regularising."
        )

    def test_perfectly_balanced_10class(self):
        """50 samples per class, 10 classes — closest to uniform."""
        p = _uniform_p(10)
        counts = np.array([50] * 10, dtype=np.int64)
        kl = _kl(p, counts)
        assert 0.0 <= kl < 0.01, f"KL={kl:.6f}"

    def test_near_uniform_large_n(self):
        """Large n means smoothing effect is tiny → KL even closer to 0."""
        p = _uniform_p(10)
        counts = np.array([1000] * 10, dtype=np.int64)
        kl = _kl(p, counts)
        assert 0.0 <= kl < 0.001, f"KL={kl:.8f}"


# ---------------------------------------------------------------------------
# Test 2 — single-class client → finite, non-NaN KL
# ---------------------------------------------------------------------------

class TestSingleClassClientFiniteKL:
    """Client with all samples concentrated in one class."""

    @pytest.mark.parametrize("dominant_class", range(10))
    def test_finite_not_nan(self, dominant_class: int):
        p = _uniform_p(10)
        counts = np.zeros(10, dtype=np.int64)
        counts[dominant_class] = 500
        kl = _kl(p, counts)
        assert not math.isnan(kl), (
            f"KL must not be NaN for single-class client (class={dominant_class})"
        )
        assert math.isfinite(kl), (
            f"KL must be finite for single-class client (class={dominant_class}), got {kl}"
        )

    def test_single_class_positive(self):
        """A non-uniform client must have strictly positive KL."""
        p = _uniform_p(10)
        counts = np.zeros(10, dtype=np.int64)
        counts[0] = 500
        kl = _kl(p, counts)
        assert kl > 0.0, f"Single-class client KL must be > 0, got {kl}"

    def test_value_range_single_class(self):
        """For uniform P, KL of a single-class client has a known upper bound.

        With K=10, α=0.5, n=500:
            q̃_dom ≈ 500.5 / (500 + 5) = ~0.991
            q̃_other ≈ 0.5 / 505 ≈ 0.001
            KL ≈ 0.1 * log(0.1/0.991) * 1  +  9 * 0.1 * log(0.1/0.001)
               ≈ 0.1 * (-2.29) + 0.9 * 4.6 ≈ 3.9

        The exact value depends on α; just ensure it's in a sane range.
        """
        p = _uniform_p(10)
        counts = np.zeros(10, dtype=np.int64)
        counts[0] = 500
        kl = _kl(p, counts)
        assert 1.0 < kl < 10.0, f"Single-class KL unexpectedly outside [1, 10]: {kl}"


# ---------------------------------------------------------------------------
# Test 3 — balanced < skewed
# ---------------------------------------------------------------------------

class TestBalancedVsSkewed:
    """The client closer to uniform should always have a lower KL score."""

    def test_basic_ordering(self):
        p = _uniform_p(10)
        balanced_counts = np.array([50] * 10, dtype=np.int64)
        skewed_counts = np.zeros(10, dtype=np.int64)
        skewed_counts[0] = 450
        skewed_counts[1] = 50
        kl_bal = _kl(p, balanced_counts)
        kl_skw = _kl(p, skewed_counts)
        assert kl_bal < kl_skw, (
            f"KL(balanced)={kl_bal:.6f} should be < KL(skewed)={kl_skw:.6f}"
        )

    @pytest.mark.parametrize("seed", [0, 7, 42, 99, 137])
    def test_ordering_over_random_pairs(self, seed: int):
        """Randomly generate pairs; the more uniform one must have lower KL."""
        rng = np.random.default_rng(seed)
        k = 10
        p = _uniform_p(k)

        # Client A: near-uniform
        alpha_a = 5.0
        props_a = rng.dirichlet(np.full(k, alpha_a))
        counts_a = (props_a * 500).astype(np.int64)

        # Client B: very skewed (α=0.1 → most mass on one class)
        alpha_b = 0.1
        props_b = rng.dirichlet(np.full(k, alpha_b))
        counts_b = (props_b * 500).astype(np.int64)

        kl_a = _kl(p, counts_a)
        kl_b = _kl(p, counts_b)
        assert kl_a < kl_b, (
            f"seed={seed}: near-uniform KL={kl_a:.6f} should be < "
            f"skewed KL={kl_b:.6f}"
        )

    def test_single_vs_five_class(self):
        """5-class spread is more similar to uniform than 1-class."""
        p = _uniform_p(10)
        five_class = np.array([100, 100, 100, 100, 100, 0, 0, 0, 0, 0], dtype=np.int64)
        one_class = np.zeros(10, dtype=np.int64)
        one_class[0] = 500
        assert _kl(p, five_class) < _kl(p, one_class)


# ---------------------------------------------------------------------------
# Test 4 — non-negativity across random inputs
# ---------------------------------------------------------------------------

class TestNonNegativity:
    """KL(P || Q̃_i) must be ≥ 0 for all valid inputs."""

    @pytest.mark.parametrize("seed", range(50))
    def test_non_negative_random_dirichlet(self, seed: int):
        rng = np.random.default_rng(seed)
        k = 10
        p = _uniform_p(k)
        # Vary both concentration and dataset size
        concentration = float(rng.uniform(0.05, 5.0))
        n = int(rng.integers(10, 1000))
        props = rng.dirichlet(np.full(k, concentration))
        counts = (props * n).astype(np.int64)
        kl = _kl(p, counts)
        assert kl >= 0.0, (
            f"KL must be non-negative (seed={seed}, conc={concentration:.3f}, "
            f"n={n}, counts={counts.tolist()}), got {kl}"
        )

    def test_non_negative_all_zero_except_one(self):
        """Edge case: only 1 sample total in one class."""
        p = _uniform_p(10)
        counts = np.zeros(10, dtype=np.int64)
        counts[3] = 1
        assert _kl(p, counts) >= 0.0

    def test_non_negative_non_uniform_p(self):
        """KL should also be non-negative for non-uniform server distributions."""
        rng = np.random.default_rng(0)
        k = 10
        raw = rng.dirichlet(np.ones(k) * 2)
        p = raw / raw.sum()
        counts = np.array([50] * k, dtype=np.int64)
        assert _kl(p, counts) >= 0.0


# ---------------------------------------------------------------------------
# Edge cases identified by math review
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Verify the two edge cases flagged by the Opus math review."""

    def test_zero_total_samples_returns_inf(self):
        """Client with NO data at all must return +inf, not ~0.

        Without this guard, Laplace smoothing collapses an empty counts vector
        to the uniform distribution, giving KL≈0 and incorrectly ranking the
        client as most globally representative.
        """
        p = _uniform_p(10)
        counts = np.zeros(10, dtype=np.int64)  # all-zero: no data at all
        kl = _kl(p, counts)
        assert math.isinf(kl) and kl > 0, (
            f"All-zero counts must return +inf, got {kl}"
        )

    def test_zero_in_some_classes_not_inf(self):
        """Zero in individual classes (normal Dirichlet α=0.5 behaviour) must be finite.

        Smoothing gives q̃_{i,k} = α/(n_i + α·K) > 0 for zero-count classes.
        Only ALL-zero clients should return inf.
        """
        p = _uniform_p(10)
        counts = np.zeros(10, dtype=np.int64)
        counts[0] = 500  # data in one class, zero in all others
        kl = _kl(p, counts)
        assert math.isfinite(kl), (
            "Client with data in one class and zeros elsewhere must return finite KL."
        )
        assert kl > 0, "Single-class client must have positive KL."

    def test_float_clamp_non_negative(self):
        """Perfectly balanced client may have tiny rounding error; result must be ≥ 0.

        Mathematically KL=0 for a uniform client against uniform P.
        Float64 accumulation of log terms can theoretically produce a value
        slightly below zero; the clamp must catch this.
        """
        p = _uniform_p(10)
        # 50 samples/class → q̃_k = 50.5/505 = 0.1 exactly → KL=0 analytically
        counts = np.array([50] * 10, dtype=np.int64)
        kl = _kl(p, counts)
        assert kl >= 0.0, f"Clamped result must be ≥ 0, got {kl}"
        assert kl < 1e-12, f"Balanced client KL must be ≈ 0, got {kl}"

    def test_zero_in_classes_ordered_above_data_clients(self):
        """A client with no data (inf) must never be preferred over one with data."""
        p = _uniform_p(10)
        empty = np.zeros(10, dtype=np.int64)
        with_data = np.zeros(10, dtype=np.int64)
        with_data[0] = 500
        kl_empty = _kl(p, empty)
        kl_skewed = _kl(p, with_data)
        assert kl_empty > kl_skewed, (
            "Empty client (inf) must have higher KL than any client with data."
        )


# ---------------------------------------------------------------------------
# Additional: smoothing parameter sensitivity
# ---------------------------------------------------------------------------

class TestSmoothing:
    """Verify that different α values produce finite, stable results."""

    @pytest.mark.parametrize("alpha", [0.01, 0.5, 1.0, 2.0])
    def test_various_alpha_values_finite(self, alpha: float):
        p = _uniform_p(10)
        counts = np.zeros(10, dtype=np.int64)
        counts[0] = 500
        kl = KLGreedyClientSelectionStrategy._kl_divergence(p, counts, alpha=alpha)
        assert math.isfinite(kl), f"KL not finite with alpha={alpha}: {kl}"
        assert kl >= 0.0, f"KL negative with alpha={alpha}: {kl}"

    def test_more_smoothing_reduces_kl_for_extreme_client(self):
        """Higher α pulls Q̃ toward uniform, reducing KL for extreme clients."""
        p = _uniform_p(10)
        counts = np.zeros(10, dtype=np.int64)
        counts[0] = 500
        kl_low_alpha = KLGreedyClientSelectionStrategy._kl_divergence(p, counts, alpha=0.01)
        kl_high_alpha = KLGreedyClientSelectionStrategy._kl_divergence(p, counts, alpha=10.0)
        assert kl_low_alpha > kl_high_alpha, (
            f"More smoothing should reduce KL for extreme client: "
            f"α=0.01 → {kl_low_alpha:.4f}, α=10.0 → {kl_high_alpha:.4f}"
        )
