"""
Stage 4 Task 2 instrumentation — log every scored candidate per dispatch, not
only the winner.

This is a NEW file. It does not change any selector's decisions: it wraps an
already-fully-constructed selection strategy (the same pattern
`sumo_close_loop/sumo_gates.py`'s `RsuSelectionGate` uses), lets the wrapped
strategy make the real call first, and only AFTERWARDS recomputes the same
per-candidate quantities purely for logging. The recomputation reads the same
`context.state` the real call just read (nothing mutates it in between) and,
for the seconds-predictor arms, calls the exact same `_predict_staleness`
closure object the real strategy holds — not a re-implementation of it — so
the logged numbers cannot drift from what selection actually used to decide.

Column semantics (see prompts/STAGE4_verify_scoring.md Task 2):
    tau_hat_raw   — tau_hat in rounds, exactly as fed into s(.) at
                    fedasync_polaris_vi_selection.py:266 (for baseline: blank,
                    it never computes one).
    s_tau_hat     — 1 / sqrt(1 + tau_hat_raw).
    G_i           — context.state["polaris"]["local_gradient_bounds"][cid-1]
                    at the moment of scoring (blank for baseline).
    V_i           — G_i * s_tau_hat (blank for baseline).
    g_is_prior    — True if this candidate is still in the aggregation
                    strategy's `unexplored_clients` set, i.e. G_i is the
                    alpha * mean(observed) proxy rather than a measured norm
                    (fedasync_polaris_vi_server.py:279-293). Always blank for
                    baseline, which has no G_i tracking at all.
    g_prior_value — G_i's current value when g_is_prior is true (the proxy IS
                    G_i in that row); blank otherwise.
    selected      — 1 if this candidate was in the real, unmodified selection
                    result this dispatch; 0 otherwise.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Optional

from gpu_tier_sleep import (
    compute_t_round_from_times,
    gpu_tops_predict_staleness_rounds,
    seconds_to_rounds,
    tier_for,
)
from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext


class ViTermLogger(ClientSelectionStrategy):
    """Wraps a client-selection strategy; logs every eligible candidate.

    Pure passthrough for every lifecycle hook except `select_clients`, where
    it calls `self.inner.select_clients(...)` FIRST (the real, unmodified
    decision), then recomputes per-candidate diagnostics for logging only.
    """

    FIELDS = [
        "round", "client_id", "tier", "selected",
        "s_tau_hat", "tau_hat_raw", "G_i", "V_i",
        "rounds_since_last_selected", "times_selected_so_far",
        "g_is_prior", "g_prior_value",
    ]

    def __init__(self, inner: ClientSelectionStrategy, log_path: str | Path) -> None:
        self.inner = inner
        self.log_path = Path(log_path)
        self._last_selected_round: dict[int, int] = {}
        self._times_selected: dict[int, int] = {}
        self._file = None
        self._writer = None

    # -- lifecycle: passthrough, plus opening/closing the CSV --------------

    def setup(self, context: ServerContext) -> None:
        self.inner.setup(context)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.FIELDS)

    def teardown(self, context: ServerContext) -> None:
        self.inner.teardown(context)
        if self._file is not None:
            self._file.close()

    def on_clients_selected(self, selected_clients, context: ServerContext) -> None:
        self.inner.on_clients_selected(selected_clients, context)

    def on_reports_received(self, updates, context: ServerContext) -> None:
        self.inner.on_reports_received(updates, context)

    # -- the logged hook -----------------------------------------------------

    def select_clients(
        self,
        clients_pool: list[int],
        clients_count: int,
        context: ServerContext,
    ) -> list[int]:
        # The real decision, completely unmodified.
        selected = self.inner.select_clients(clients_pool, clients_count, context)

        try:
            self._log_candidates(clients_pool, selected, context)
        except Exception:  # noqa: BLE001 — logging must never break a run
            logging.exception("[ViTermLogger] failed to log this dispatch")

        current_round = getattr(context.server, "current_round", context.current_round)
        for cid in selected:
            self._last_selected_round[cid] = current_round
            self._times_selected[cid] = self._times_selected.get(cid, 0) + 1

        return selected

    # -- read-only recomputation, for logging only ---------------------------

    def _log_candidates(
        self,
        clients_pool: list[int],
        selected: list[int],
        context: ServerContext,
    ) -> None:
        server = context.server
        current_round = getattr(server, "current_round", context.current_round)
        selected_set = set(selected)

        busy = set(getattr(server, "training_clients", {}) or {})
        eligible = [cid for cid in clients_pool if cid not in busy]

        predict_staleness = getattr(self.inner, "_predict_staleness", None)
        predict_unit = getattr(self.inner, "_predict_staleness_unit", None)
        is_scored = predict_staleness is not None  # False for baseline (plain random)

        polaris_state = context.state.get("polaris", {})
        G = polaris_state.get("local_gradient_bounds")

        aggregation_strategy = getattr(server, "aggregation_strategy", None)
        unexplored = getattr(aggregation_strategy, "unexplored_clients", None) or set()

        # Mirrors fedasync_polaris_vi_selection.py:199-248 exactly, read-only,
        # for the "seconds" predictor family (gpu/cpu/gpu_oracle). Not
        # exercised by Task 2's two runs (baseline has no predictor; gpu's
        # `predict_staleness_unit` IS "seconds"), but kept general so this file
        # also works unmodified if a future task points it at cpu/observed.
        busy_residuals: list[float] = []
        t_round: float = context.state.get("T_round_seconds", 1.0)
        if is_scored and predict_unit == "seconds":
            wall_now = float(getattr(server, "wall_time", 0.0))
            training_clients = getattr(server, "training_clients", {}) or {}
            sizes = context.state.get("partition_sizes", {})
            local_epochs = int(context.state.get("local_epochs", 1))
            clients_per_round = int(
                context.state.get("clients_per_round", max(1, len(busy) or 1))
            )
            default_size = next(iter(sizes.values()), 500) if sizes else 500

            if busy and sizes:
                # predict_staleness(cid, ctx) returns seconds for THIS client's
                # own full training time; that is also exactly what a busy
                # client's full time is, so reuse it instead of importing
                # gpu_tops_predict_seconds separately.
                busy_full_times = [predict_staleness(cid, context) for cid in busy]
                t_round_live = compute_t_round_from_times(busy_full_times, clients_per_round)
                if t_round_live > 0.0:
                    t_round = t_round_live

            if wall_now > 0.0 and training_clients and sizes:
                for busy_cid in busy:
                    t_full = predict_staleness(busy_cid, context)
                    start_t = float(
                        (training_clients.get(busy_cid) or {}).get("start_time", wall_now)
                    )
                    residual = max(0.0, t_full - (wall_now - start_t))
                    busy_residuals.append(residual)

        for cid in eligible:
            tau_hat_raw: Optional[float] = None
            s_tau_hat: Optional[float] = None
            g_i: Optional[float] = None
            v_i: Optional[float] = None
            g_is_prior: Optional[bool] = None
            g_prior_value: Optional[float] = None

            if is_scored:
                tau_predicted = predict_staleness(cid, context)
                if predict_unit == "seconds":
                    if busy_residuals:
                        tau_hat_raw = gpu_tops_predict_staleness_rounds(
                            tau_predicted, busy_residuals, t_round
                        )
                    else:
                        tau_hat_raw = seconds_to_rounds(tau_predicted, t_round)
                else:
                    tau_hat_raw = tau_predicted
                s_tau_hat = 1.0 / math.sqrt(1.0 + max(tau_hat_raw, 0.0))
                if G is not None:
                    g_i = float(G[cid - 1])
                    v_i = g_i * s_tau_hat
                g_is_prior = (cid - 1) in unexplored
                if g_is_prior:
                    g_prior_value = g_i

            try:
                tier = tier_for(cid).tier_id
            except KeyError:
                tier = "unknown"

            self._writer.writerow([
                current_round,
                cid,
                tier,
                int(cid in selected_set),
                _fmt(s_tau_hat),
                _fmt(tau_hat_raw),
                _fmt(g_i),
                _fmt(v_i),
                current_round - self._last_selected_round.get(cid, 0),
                self._times_selected.get(cid, 0),
                "" if g_is_prior is None else int(g_is_prior),
                _fmt(g_prior_value),
            ])
        self._file.flush()


def _fmt(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.8f}"
