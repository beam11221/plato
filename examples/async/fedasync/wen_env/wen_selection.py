"""
Selection wrappers for the Wen environment: the coverage gate and Stage 0.

Order, outermost first (the same layering as sumo_close_loop):

    WenCoverageGate  ->  WenStage0Gate (S2, optional)  ->  the arm's selector

WenCoverageGate
    S1: every idle vehicle is eligible. S2: only vehicles with
    x_dispatch in [400, 1600]. It asks for the pipeline DEFICIT
    (C - training - awaiting merge) rather than PLATO's requested count, so a
    slot that coverage left empty is refilled at the next event
    (sumo_close_loop/README.md, "The pipeline must be refilled to its full
    depth").

WenStage0Gate (Prompt B, Step 6) -- pluggable around ANY selector:
    feasible iff R_i = (1600 - x_i) / V_k  >=  T_hat_train + T_hat_up
             and E_hat_i <= E_max
    with the arm's own estimator and the vehicle's REPORTED V_k (not v_max).
    If nothing is feasible the unfiltered pool is used and the event is logged,
    the existing Stage 0 rule ("never idle a slot it could have filled").
    Prompt I: in S1 coverage is the whole loop, so there is no exit boundary
    and R_i is infinite (the time test cannot fail). Study 2 arms run Stage 0
    with the gpu estimator whatever their host (wen_server.install_wen). The
    gate refuses any vehicle that is not eligible (idle, and in S2 inside
    coverage): eligibility is applied before Stage 0, never by it.

Both read only `ReportedVehicle` records (via `report(cid, t)`) and estimates.
"""

from __future__ import annotations

import logging
import math

import wen_constants as K
from wen_estimators import Estimator

from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext


class _Wrapper(ClientSelectionStrategy):
    def __init__(self, inner: ClientSelectionStrategy) -> None:
        self.inner = inner

    def setup(self, context: ServerContext) -> None:
        self.inner.setup(context)

    def teardown(self, context: ServerContext) -> None:
        self.inner.teardown(context)

    def on_clients_selected(self, selected_clients, context: ServerContext) -> None:
        self.inner.on_clients_selected(selected_clients, context)

    def on_reports_received(self, updates, context: ServerContext) -> None:
        self.inner.on_reports_received(updates, context)


def stage0_checks(rv, estimate, e_max: float = K.E_MAX_J,
                  exit_x: float | None = K.S2_COVERAGE_M[1]) -> tuple[bool, bool, float]:
    """(time ok, energy ok, R_i). Exit boundary 1600 m (one-way, eastbound);
    exit_x None = no boundary (S1), R_i = inf."""
    remaining = math.inf if exit_x is None else (exit_x - rv.x) / rv.speed
    return remaining >= estimate.total, estimate.E <= e_max, remaining


def stage0_feasible(rv, estimate, e_max: float = K.E_MAX_J,
                    exit_x: float | None = K.S2_COVERAGE_M[1]) -> tuple[bool, float]:
    """(feasible, R_i)."""
    time_ok, energy_ok, remaining = stage0_checks(rv, estimate, e_max, exit_x)
    return time_ok and energy_ok, remaining


TIERS = ("xavier", "orin", "thor")


class WenStage0Gate(_Wrapper):
    def __init__(self, inner, estimator: Estimator, e_max: float = K.E_MAX_J,
                 scenario: str = "S2") -> None:
        super().__init__(inner)
        self.estimator = estimator
        self.e_max = e_max
        self.exit_x = K.S2_COVERAGE_M[1] if scenario == "S2" else None
        self.last_pass: dict[int, bool] = {}
        self.last_stats: dict = {}
        self.cycles = self.candidates = self.rejected = self.fallbacks = 0

    def select_clients(self, clients_pool, clients_count, context):
        server = context.server
        t = server.sumo_time()
        not_eligible = [c for c in clients_pool if not server.wen_is_eligible(c, t)]
        if not_eligible:
            raise RuntimeError(f"[WenStage0] received non-eligible vehicles {not_eligible} "
                               f"at t={t!r}; eligibility must be applied before Stage 0.")
        passed, failed = [], []
        self.last_pass = {}
        stats = {"n_eligible": len(clients_pool), "n_rej_time": 0, "n_rej_energy": 0,
                 "n_rej_both": 0, **{f"n_rej_{tr}": 0 for tr in TIERS}}
        for cid in clients_pool:
            rv = server.wen_report(cid, t)
            time_ok, energy_ok, _ = stage0_checks(rv, self.estimator.estimate(rv),
                                                  self.e_max, self.exit_x)
            ok = time_ok and energy_ok
            self.last_pass[cid] = ok
            (passed if ok else failed).append(cid)
            if not ok:
                key = ("n_rej_both" if not (time_ok or energy_ok) else
                       "n_rej_time" if not time_ok else "n_rej_energy")
                stats[key] += 1
                stats[f"n_rej_{rv.tier}"] += 1

        self.cycles += 1
        self.candidates += len(clients_pool)
        self.rejected += len(failed)
        fallback = bool(clients_pool) and not passed
        self.fallbacks += int(fallback)
        stats.update(n_passed=len(passed), n_rejected=len(failed), fallback=int(fallback))
        self.last_stats = stats
        if hasattr(server, "record_stage0"):
            server.record_stage0(t, stats)
        logging.info(
            "[WenStage0] round=%d t=%.3f estimator=%s pool=%d feasible=%d "
            "rejected=%s fallback=%d",
            getattr(server, "current_round", -1), t, self.estimator.name,
            len(clients_pool), len(passed), sorted(failed), int(fallback),
        )
        return self.inner.select_clients(clients_pool if fallback else passed,
                                         clients_count, context)

    def log_summary(self) -> None:
        logging.info(
            "[WenStage0] summary estimator=%s cycles=%d candidates=%d rejected=%d "
            "(%.1f%%) fallback_cycles=%d",
            self.estimator.name, self.cycles, self.candidates, self.rejected,
            100 * self.rejected / max(self.candidates, 1), self.fallbacks,
        )


class WenCoverageGate(_Wrapper):
    def __init__(self, inner, scenario: str, diagnostic: Estimator) -> None:
        super().__init__(inner)
        self.scenario = scenario
        # The pool-size diagnostic ("feasible-and-idle", Step 9) always uses
        # the gpu estimator, in every arm, so the number is comparable across
        # cells; it never affects selection.
        self.diagnostic = diagnostic

    @staticmethod
    def open_slots(server) -> int:
        training = len(getattr(server, "training_clients", {}) or {})
        awaiting = len(getattr(server, "reported_clients", []) or [])
        return max(0, server.clients_per_round - training - awaiting)

    def eligible(self, server, clients_pool, t) -> list[int]:
        if self.scenario == "S1":
            return list(clients_pool)
        lo, hi = K.S2_COVERAGE_M
        return [c for c in clients_pool if lo <= server.wen_report(c, t).x <= hi]

    def select_clients(self, clients_pool, clients_count, context):
        server = context.server
        t = server.sumo_time()
        eligible = self.eligible(server, clients_pool, t)
        feasible = sum(
            1 for c in eligible
            if stage0_feasible(server.wen_report(c, t),
                               self.diagnostic.estimate(server.wen_report(c, t)))[0]
        )
        slots = self.open_slots(server)
        server.record_pool(t, len(clients_pool), len(eligible), feasible, slots)
        selected = self.inner.select_clients(eligible, slots, context) if slots else []
        server.record_dispatches(selected, context)
        return selected
