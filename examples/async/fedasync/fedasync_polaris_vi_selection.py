"""
Polaris V_i client selection strategy for async federated learning.

Scores each available client as

    V_i = G_i / sqrt(1 + τ̂_i)

where
    G_i   — gradient-bound estimate read from context.state["polaris"]["local_gradient_bounds"],
             populated each round by FedAsyncPolarisAggregationStrategy.
    τ̂_i   — predicted staleness, supplied via a pluggable callable
             (default: observed τ_i from context.state["polaris"]["local_stalenesses"]).
    s(τ)  — staleness discount 1/sqrt(1+τ) (monotone-decreasing, smooth at τ=0).

The top-K clients by V_i are selected (deterministic), after applying the
FedAsync busy-client filter and open-slot cap inherited from
FedAsyncClientSelectionStrategy.

Pluggable staleness prediction
------------------------------
Pass ``predict_staleness`` to the constructor to override the default observed-
staleness stub with a model-based predictor (e.g., GPU-TOPS formula).  The
callable must have the signature::

    predict_staleness(client_id: int, context: ServerContext) -> float

Return value should be ≥ 0.  A value of 0 means "expected immediately"; high
values penalise slow clients.

Integration
-----------
This strategy reads G_i but does NOT write it.  Pair it with
``FedAsyncPolarisAggregationStrategy`` (fedasync_polaris_vi_server.py), which
computes G_i from conv-layer weight deltas during each aggregation event and
writes the result to the shared context.state["polaris"] dict.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Literal, Optional

import numpy as np

from fedasync_selection import FedAsyncClientSelectionStrategy
from gpu_tier_sleep import (
    compute_t_round_from_times,
    gpu_tops_predict_seconds,
    gpu_tops_predict_staleness_rounds,
    seconds_to_rounds,
)
from plato.servers.strategies.base import ServerContext


# ---------------------------------------------------------------------------
# Default staleness predictor — returns observed staleness from polaris state
# ---------------------------------------------------------------------------

def _observed_staleness(client_id: int, context: ServerContext) -> float:
    """Return the most recently observed staleness for client_id.

    Falls back to 0.01 for clients that have never reported.
    """
    polaris_state = context.state.get("polaris", {})
    stalenesses = polaris_state.get("local_stalenesses")
    if stalenesses is None or client_id - 1 >= len(stalenesses):
        return 0.01
    return max(float(stalenesses[client_id - 1]), 0.0)


# ---------------------------------------------------------------------------
# Main strategy
# ---------------------------------------------------------------------------

class PolarisViClientSelectionStrategy(FedAsyncClientSelectionStrategy):
    """Deterministic top-K selection by V_i = G_i / sqrt(1 + τ̂_i).

    Inherits the busy-client filter and open-slot cap from
    FedAsyncClientSelectionStrategy, then replaces random sampling with a
    deterministic sort on V_i scores.
    """

    def __init__(
        self,
        predict_staleness: Optional[Callable[[int, ServerContext], float]] = None,
        predict_staleness_unit: Literal["seconds", "rounds"] = "rounds",
    ) -> None:
        super().__init__()
        self._predict_staleness: Callable[[int, ServerContext], float] = (
            predict_staleness or _observed_staleness
        )
        self._predict_staleness_unit: Literal["seconds", "rounds"] = predict_staleness_unit
        self._total_clients: int = 0
        # Predicted full training time (s) of a BUSY client, for the live
        # T_round and the residuals.  None -> gpu_tops_predict_seconds(cid, ...),
        # the historical behaviour.  Stage 7b sets it so the busy terms read the
        # identity frozen at dispatch rather than the slot's fleet-file row.
        self.busy_predict_seconds: Optional[Callable[[int, ServerContext], float]] = None
        # tau_hat (rounds) each candidate was scored with in the last call — the
        # value V_i actually used, exposed so it can be logged per dispatch.
        self.last_tau_hats: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, context: ServerContext) -> None:
        super().setup(context)
        self._total_clients = context.total_clients
        polaris_state = context.state.setdefault("polaris", {})
        polaris_state.setdefault(
            "local_gradient_bounds",
            np.full(self._total_clients, 0.5, dtype=float),
        )
        polaris_state.setdefault(
            "local_stalenesses",
            np.full(self._total_clients, 0.01, dtype=float),
        )

    # ------------------------------------------------------------------
    # Report hook — update observed stalenesses from client reports
    # ------------------------------------------------------------------

    def on_reports_received(
        self, updates: list, context: ServerContext
    ) -> None:
        if not updates:
            return
        polaris_state = context.state.setdefault("polaris", {})
        stalenesses = polaris_state.setdefault(
            "local_stalenesses",
            np.full(self._total_clients, 0.01, dtype=float),
        )
        predictions: dict[int, float] = polaris_state.setdefault("staleness_predictions", {})

        for update in updates:
            cid = update.client_id
            idx = cid - 1
            # Store the raw PLATO staleness (current_round − round_when_selected).
            # Do NOT add an offset here — _observed_staleness reads this value
            # directly and any artificial inflation would make all clients look
            # equally stale, destroying the predictor's ability to differentiate.
            tau_actual = float(getattr(update, "staleness", 0.0))
            stalenesses[idx] = tau_actual

            # Validate prediction accuracy: compare τ̂_i (predicted at selection
            # time) against the actual τ_i that arrived with this report.
            tau_hat = predictions.pop(cid, None)
            if tau_hat is not None:
                error = tau_actual - tau_hat
                denom = max(tau_actual, 0.01)
                rel_error = 100.0 * error / denom
                logging.info(
                    "[PolarisVi] Staleness prediction accuracy:"
                    " cid=%d  predicted=%.4f rounds  actual=%.4f rounds"
                    "  error=%+.4f  rel_error=%+.1f%%",
                    cid, tau_hat, tau_actual, error, rel_error,
                )
            else:
                logging.debug(
                    "[PolarisVi] Staleness accuracy: cid=%d  actual=%.4f rounds"
                    " (no stored prediction — client was not selected via this strategy)",
                    cid, tau_actual,
                )

    # ------------------------------------------------------------------
    # Core selection
    # ------------------------------------------------------------------

    def select_clients(
        self,
        clients_pool: list[int],
        clients_count: int,
        context: ServerContext,
    ) -> list[int]:
        server = context.server
        busy = set(getattr(server, "training_clients", {}) or {})
        clients_per_round = getattr(server, "clients_per_round", clients_count)
        open_slots = max(0, clients_per_round - len(busy))

        eligible = [cid for cid in clients_pool if cid not in busy]
        effective_count = min(clients_count, open_slots, len(eligible))

        if effective_count == 0:
            logging.info(
                "[PolarisVi] No eligible clients (busy=%d, open_slots=%d, pool=%d).",
                len(busy), open_slots, len(clients_pool),
            )
            return []

        if effective_count < clients_count:
            logging.info(
                "[PolarisVi] %d busy, %d open slot(s), %d eligible; "
                "selecting %d of %d requested.",
                len(busy), open_slots, len(eligible),
                effective_count, clients_count,
            )

        scores, tau_hats, G, total = self.score_candidates(eligible, context)
        polaris_state = context.state.setdefault("polaris", {})

        # Stochastic V_i-weighted sampling (without replacement).
        # Sampling proportional to V_i avoids the tier-clustering that arises
        # when all G_i are equal and deterministic top-K always fills every slot
        # with the fastest tier (Thor).  High-scoring clients are still selected
        # with higher probability, preserving the Polaris intent.
        scores_arr = np.array(
            [max(scores[cid], 0.0) for cid in eligible], dtype=float
        )
        total_weight = float(scores_arr.sum())
        if total_weight > 0 and effective_count < len(eligible):
            probs = scores_arr / total_weight
            chosen = np.random.choice(
                len(eligible), size=effective_count, replace=False, p=probs
            )
            selected = [eligible[int(i)] for i in chosen]
        else:
            # All scores zero, or selecting the entire pool: fall back to top-K.
            sorted_eligible = sorted(eligible, key=lambda c: scores[c], reverse=True)
            selected = sorted_eligible[:effective_count]

        # Persist predicted staleness for each selected client so that
        # on_reports_received() can compare it against actual staleness.
        # Only selected clients are stored — they immediately enter the busy
        # set and cannot be re-scored until they report, so there is no risk
        # of overwriting an unconsumed prediction.
        predictions: dict[int, float] = polaris_state.setdefault("staleness_predictions", {})
        for cid in selected:
            predictions[cid] = tau_hats[cid]

        current_round = getattr(server, "current_round", context.current_round)
        self._log_round(current_round, scores, selected, G, total)

        return selected

    # ------------------------------------------------------------------
    # Scoring (shared by select_clients and the Wen e-greedy rule)
    # ------------------------------------------------------------------

    def score_candidates(self, eligible: list[int], context: ServerContext):
        """V_i and tau_hat for every eligible client; no sampling.

        Returns (scores, tau_hats, G, total). Extracted unchanged from
        select_clients so another selection rule (wen_env's e-greedy) can rank
        by the same V_i without consuming the sampling RNG.
        """
        server = context.server
        busy = set(getattr(server, "training_clients", {}) or {})
        polaris_state = context.state.setdefault("polaris", {})
        total = self._total_clients
        G = polaris_state.get(
            "local_gradient_bounds",
            np.full(total, 0.5, dtype=float),
        )

        # Pre-compute busy-client data once — shared by all candidate scores.
        # Only used for the "seconds" (gpu_tops) path; the "rounds" path is unchanged.
        busy_residuals: list[float] = []
        # Start from the static startup estimate; overwritten below when the busy
        # set is non-empty so the denominator reflects the live pool composition.
        t_round: float = context.state.get("T_round_seconds", 1.0)
        if self._predict_staleness_unit == "seconds":
            wall_now = float(getattr(server, "wall_time", 0.0))
            training_clients = getattr(server, "training_clients", {}) or {}
            sizes = context.state.get("partition_sizes", {})
            local_epochs = int(context.state.get("local_epochs", 1))
            clients_per_round = int(context.state.get("clients_per_round", max(1, len(busy) or 1)))
            default_size = next(iter(sizes.values()), 500) if sizes else 500

            def _busy_seconds(busy_cid: int) -> float:
                if self.busy_predict_seconds is not None:
                    return float(self.busy_predict_seconds(busy_cid, context))
                return gpu_tops_predict_seconds(
                    busy_cid, sizes.get(busy_cid, default_size), local_epochs
                )

            if busy and sizes:
                # Recompute T_round from the current busy pool's GPU-TOPS speeds.
                # This is the proactive signal: the denominator is derived purely
                # from hardware specs of who is actually training now, not from
                # the static startup average over all 100 clients.  As faster or
                # slower clients cycle in and out of the busy set (IoV dynamic
                # environment), T_round shifts accordingly — with no reliance on
                # historical observations.
                busy_full_times = [_busy_seconds(cid) for cid in busy]
                t_round_live = compute_t_round_from_times(busy_full_times, clients_per_round)
                if t_round_live > 0.0:
                    t_round = t_round_live

            if wall_now > 0.0 and training_clients and sizes:
                for busy_cid in busy:
                    t_full = _busy_seconds(busy_cid)
                    start_t = float(
                        (training_clients.get(busy_cid) or {}).get("start_time", wall_now)
                    )
                    residual = max(0.0, t_full - (wall_now - start_t))
                    busy_residuals.append(residual)

            logging.debug(
                "[PolarisVi] busy_residuals (n=%d): %s  t_round=%.6f s",
                len(busy_residuals),
                [round(r, 5) for r in sorted(busy_residuals)],
                t_round,
            )

        # Score every eligible client
        scores: dict[int, float] = {}
        tau_hats: dict[int, float] = {}  # tau_hat_rounds per client, for prediction logging
        for cid in eligible:
            g_i = float(G[cid - 1])
            tau_predicted = self._predict_staleness(cid, context)
            if self._predict_staleness_unit == "seconds":
                if busy_residuals:
                    tau_hat_rounds = gpu_tops_predict_staleness_rounds(
                        tau_predicted, busy_residuals, t_round
                    )
                else:
                    # Fallback: no busy clients yet (first round) or wall_time unavailable.
                    tau_hat_rounds = seconds_to_rounds(tau_predicted, t_round)
            else:
                # unit == "rounds": observed-staleness path — unchanged.
                tau_hat_rounds = tau_predicted
            v_i = g_i / math.sqrt(1.0 + max(tau_hat_rounds, 0.0))
            scores[cid] = v_i
            tau_hats[cid] = tau_hat_rounds
            logging.debug(
                "[PolarisVi] cid=%d  tau_predicted=%.6f (%s)  "
                "tau_hat_rounds=%.6f  V_i=%.6f",
                cid, tau_predicted, self._predict_staleness_unit,
                tau_hat_rounds, v_i,
            )

        self.last_tau_hats = dict(tau_hats)
        # The T_round this call converted seconds to rounds with (Prompt M
        # mechanism logging); read by the passive logger only.
        self.last_t_round = t_round
        return scores, tau_hats, G, total

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_round(
        self,
        current_round: int,
        scores: dict[int, float],
        selected: list[int],
        G: np.ndarray,
        total: int,
    ) -> None:
        logging.info(
            "[PolarisVi] Round %d — selected clients %s with V_i scores %s",
            current_round,
            selected,
            {cid: round(scores[cid], 6) for cid in selected},
        )
        g_vals = G[:total]
        logging.info(
            "[PolarisVi] Round %d — G_i stats: min=%.6f  max=%.6f  mean=%.6f",
            current_round,
            float(np.min(g_vals)),
            float(np.max(g_vals)),
            float(np.mean(g_vals)),
        )
