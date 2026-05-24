"""
KL-greedy client selection strategy for async federated learning.

Selects the top-K available clients with the LOWEST KL divergence between
their local label distribution and the server's global test distribution.
Lower KL = more similar to global = preferred.

This is a diversity-only oracle selector: it reads ground-truth label
distributions from ClientLabelProfile (pre-generated CSV), not from client
reports.  No estimation, no noise — exact ground truth, same discipline as
SpeedGreedy reading true GPU TFLOPS.

MATH
----
Server global distribution P is computed once from the test dataset at first
select_clients() call (testset is not yet available at setup() time):

    p_k = count_k_in_testset / total_testset_samples

Client i's smoothed local distribution Q̃_i (Laplace / Jeffreys prior, α=0.5):

    q̃_{i,k} = (n_{i,k} + α) / (|D_i| + α · K)

KL divergence (P first, local second):

    KL(P || Q̃_i) = Σ_k  p_k · log( p_k / (q̃_{i,k} + ε) )

where ε = 1e-300 is a defensive guard (smoothing already prevents zeros).

CONFIGURATION (TOML)
--------------------
[server.client_selection]
strategy = "kl_greedy"

[label_distribution]
csv_path = "/path/to/client_label_distributions.csv"

PER-ROUND LOGGING
-----------------
Each round emits:
  • client_id → KL score for every selected client
  • mean / median / p95 KL of the selected pool
  • mean / median / p95 KL of the full available pool
  • Round 1 only: top-3 lowest and top-3 highest KL with label histograms
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

import client_label_distribution
from fedasync_selection import FedAsyncClientSelectionStrategy
from plato.servers.strategies.base import ServerContext

# Jeffreys prior for Laplace smoothing on Q_i
_ALPHA: float = 0.5

# Defensive numerical guard inside log (smoothing already prevents zeros)
_EPSILON: float = 1e-300


class KLGreedyClientSelectionStrategy(FedAsyncClientSelectionStrategy):
    """
    Deterministic lowest-KL-divergence client selection.

    Inherits the busy-client filter and open-slot cap from
    FedAsyncClientSelectionStrategy, then sorts survivors by
    KL(P || Q̃_i) ascending and returns the top-K.
    """

    def __init__(self) -> None:
        super().__init__()
        self._global_p: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Strategy lifecycle
    # ------------------------------------------------------------------

    def setup(self, context: ServerContext) -> None:
        super().setup(context)
        # _global_p is computed lazily on first select_clients() call because
        # context.server.testset is not populated until after configure()
        # returns from fedavg.Server, which happens after setup() is called.

    # ------------------------------------------------------------------
    # Global distribution (lazy, cached)
    # ------------------------------------------------------------------

    def _compute_global_p(self, context: ServerContext) -> np.ndarray:
        """Compute and cache the server's global test class distribution P."""
        server = context.server
        testset = getattr(server, "testset", None)
        if testset is None:
            raise RuntimeError(
                "[KLGreedy] server.testset is None — cannot compute global P. "
                "Ensure the server has loaded its test dataset before selecting."
            )
        targets = getattr(testset, "targets", None)
        if targets is None:
            raise RuntimeError(
                "[KLGreedy] server.testset has no 'targets' attribute. "
                "Make sure the datasource exposes .targets on the test set."
            )
        targets_arr = np.array(targets, dtype=np.int64)
        counts = np.bincount(targets_arr)
        p = counts.astype(np.float64) / counts.sum()
        logging.info(
            "[KLGreedy] Global test distribution P computed (%d classes): %s",
            len(p),
            np.round(p, 4).tolist(),
        )
        return p

    # ------------------------------------------------------------------
    # Core KL computation
    # ------------------------------------------------------------------

    @staticmethod
    def _kl_divergence(
        p: np.ndarray,
        counts_i: np.ndarray,
        alpha: float = _ALPHA,
    ) -> float:
        """
        KL(P || Q̃_i) with Laplace smoothing on Q_i only.

        Parameters
        ----------
        p        : global test distribution, shape (K,), sums to 1, no zeros
        counts_i : raw class counts for client i, shape (K,), dtype int-like
        alpha    : Laplace smoothing parameter (Jeffreys prior = 0.5)

        Returns
        -------
        Non-negative float.  Finite even when individual class counts are
        zero: smoothing gives q̃_{i,k} = α/(n_i + α·K) > 0, so the log
        is always defined.  This is the normal Dirichlet α=0.5 case where
        many classes have no samples — smoothing handles it correctly.

        Returns +inf when counts_i.sum() == 0 (client has NO data at all).
        Smoothing alone cannot distinguish "no data" from "uniform data":
        with n_i=0, q̃_{i,k} = α/(α·K) = 1/K for every k, making KL≈0
        and incorrectly ranking the empty client as maximally representative.
        Returning inf excludes such clients from ever being selected.
        """
        K = len(p)
        n_i = float(counts_i.sum())

        # Only fires when ALL class counts are zero — not when individual
        # classes are zero (which smoothing handles correctly above).
        if n_i == 0:
            return math.inf

        denom = n_i + alpha * K
        kl = 0.0
        for k in range(K):
            if p[k] <= 0.0:
                continue
            q_tilde_k = (float(counts_i[k]) + alpha) / denom
            kl += p[k] * math.log(p[k] / (q_tilde_k + _EPSILON))

        # Clamp: individual log terms can be negative when q̃_k > p_k,
        # so the float64 sum can underflow to a tiny negative value at the
        # uniform-client boundary. Gibbs' inequality guarantees KL ≥ 0
        # analytically; clamp here to honour that invariant in code.
        return max(kl, 0.0)

    # ------------------------------------------------------------------
    # Main selection
    # ------------------------------------------------------------------

    def select_clients(
        self,
        clients_pool: list[int],
        clients_count: int,
        context: ServerContext,
    ) -> list[int]:
        # Lazily compute P from the test dataset on first call.
        if self._global_p is None:
            self._global_p = self._compute_global_p(context)

        server = context.server
        busy = set(getattr(server, "training_clients", {}) or {})
        clients_per_round = getattr(server, "clients_per_round", clients_count)
        open_slots = max(0, clients_per_round - len(busy))

        eligible = [cid for cid in clients_pool if cid not in busy]
        effective_count = min(clients_count, open_slots, len(eligible))

        if effective_count == 0:
            logging.info(
                "[KLGreedy] No eligible clients (busy=%d, open_slots=%d, pool=%d).",
                len(busy), open_slots, len(eligible),
            )
            return []

        if effective_count < clients_count:
            logging.info(
                "[KLGreedy] %d busy, %d open slot(s), %d eligible; "
                "selecting %d of %d requested.",
                len(busy), open_slots, len(eligible),
                effective_count, clients_count,
            )

        oracle = client_label_distribution.get_oracle()

        # Score every eligible client
        kl_scores: dict[int, float] = {}
        for cid in eligible:
            try:
                counts = oracle.label_counts_for(cid)
                kl_scores[cid] = self._kl_divergence(self._global_p, counts)
            except KeyError:
                logging.warning(
                    "[KLGreedy] No label profile for client %d — excluding from selection.",
                    cid,
                )

        if not kl_scores:
            logging.warning(
                "[KLGreedy] No clients with label profiles in eligible pool — "
                "returning empty selection."
            )
            return []

        # Sort ascending by KL (lowest = most similar to global)
        sorted_eligible = sorted(kl_scores, key=kl_scores.__getitem__)

        current_round = getattr(server, "current_round", context.current_round)
        self._log_round_stats(
            current_round, kl_scores, sorted_eligible, effective_count
        )

        selected = sorted_eligible[:effective_count]
        logging.info("[KLGreedy] Round %d — selected clients: %s", current_round, selected)
        return selected

    # ------------------------------------------------------------------
    # Per-round logging helpers
    # ------------------------------------------------------------------

    def _log_round_stats(
        self,
        current_round: int,
        kl_scores: dict[int, float],
        sorted_eligible: list[int],
        effective_count: int,
    ) -> None:
        selected_ids = sorted_eligible[:effective_count]
        selected_kl = np.array([kl_scores[cid] for cid in selected_ids])
        all_kl = np.array(list(kl_scores.values()))

        logging.info(
            "[KLGreedy] Round %d — KL scores of selected clients: %s",
            current_round,
            {cid: round(kl_scores[cid], 6) for cid in selected_ids},
        )
        logging.info(
            "[KLGreedy] Round %d — Selected pool KL:   "
            "mean=%.6f  median=%.6f  p95=%.6f",
            current_round,
            float(np.mean(selected_kl)),
            float(np.median(selected_kl)),
            float(np.percentile(selected_kl, 95)),
        )
        logging.info(
            "[KLGreedy] Round %d — Available pool KL:  "
            "mean=%.6f  median=%.6f  p95=%.6f",
            current_round,
            float(np.mean(all_kl)),
            float(np.median(all_kl)),
            float(np.percentile(all_kl, 95)),
        )

        if current_round == 1:
            self._log_round1_sanity(kl_scores, sorted_eligible)

    def _log_round1_sanity(
        self,
        kl_scores: dict[int, float],
        sorted_eligible: list[int],
    ) -> None:
        """Round-1 sanity: log top-3 lowest and top-3 highest KL with histograms.

        Top-3 (lowest KL) should look balanced; bottom-3 (highest KL) skewed.
        If that is not the case, the CSV oracle or the KL direction may be wrong.
        """
        oracle = client_label_distribution.get_oracle()
        top3_low = sorted_eligible[:3]
        top3_high = sorted_eligible[-3:][::-1]

        logging.info("[KLGreedy] ═══ Round 1 Sanity Check ═══")
        logging.info(
            "[KLGreedy] Top-3 LOWEST KL (most similar to global — should look balanced):"
        )
        for cid in top3_low:
            try:
                counts = oracle.label_counts_for(cid)
                logging.info(
                    "[KLGreedy]   client=%d  KL=%.6f  histogram=%s",
                    cid, kl_scores[cid], counts.tolist(),
                )
            except KeyError:
                logging.info("[KLGreedy]   client=%d  KL=%.6f  histogram=<missing>", cid, kl_scores[cid])

        logging.info(
            "[KLGreedy] Top-3 HIGHEST KL (most skewed — should look concentrated):"
        )
        for cid in top3_high:
            try:
                counts = oracle.label_counts_for(cid)
                logging.info(
                    "[KLGreedy]   client=%d  KL=%.6f  histogram=%s",
                    cid, kl_scores[cid], counts.tolist(),
                )
            except KeyError:
                logging.info("[KLGreedy]   client=%d  KL=%.6f  histogram=<missing>", cid, kl_scores[cid])

        logging.info("[KLGreedy] ═══════════════════════════════")
