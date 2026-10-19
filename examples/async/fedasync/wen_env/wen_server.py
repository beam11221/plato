"""
FedAsync servers for the Wen-aligned environment (Prompt B, Steps 3, 4, 5, 7).

`WenEnvMixin` sits in front of a FedAsync server and owns the environment:

Clock (Step 3)
    PLATO's simulated clock is started at 0.0. Each dispatch's whole outcome
    (true training time, finish position, fading, upload time, verdict) is
    computed by `wen_truth.GroundTruth` AT DISPATCH, from the clean time
    t_dispatch, and its duration T_train_true + T_up_true is written into the
    report in `process_client_info` (the per-dispatch override; PLATO itself
    only knows a per-client table). The host's `processing_time` is zeroed in
    the same place, so nothing measured on this machine enters the clock.
    The mixin keeps its own online clock (`_online_now`) and checks PLATO
    against it at every dispatch and every merge; a disagreement raises.

Coverage and waste (Step 4)
    S1 accepts every upload. S2 dispatches only inside [400, 1600] and accepts
    an upload only if the vehicle stays inside for the whole upload interval.
    A rejected upload frees its slot at t_finish_clean and is logged as waste
    (GPU-seconds, upload-seconds, energy). A cycle whose only update was lost
    does not count as a round (same rule as sumo_close_loop).

Estimators (Step 5)
    The Polaris selector's `predict_staleness` returns T_hat_total for a
    candidate now; `busy_predict_seconds` returns the T_hat_total frozen when
    that busy client was dispatched. T_round and tau-hat keep their existing
    definitions (queue kernel, two-phase residual count) applied to T_hat_total.
    The startup T_round is mean(T_hat_total over the fleet at t = 0) / C.

    `observed` keeps Polaris's estimator: tau-hat = the client's last delivered
    PLATO staleness, in rounds (`context.state["polaris"]["local_stalenesses"]`).
    Because each report's duration is now T_train + T_up, that staleness is the
    number of merges that happened during the vehicle's whole task, training
    AND upload; nothing else is stored. Lost (undelivered) reports never update
    it.

Logs (Step 7), all in the run's base_path:
    dispatch_log.csv   one row per dispatch, the Step 7 columns plus cycle and k
    merge_log.csv      one row per update PLATO popped, in pop order
    cycle_log.csv      one row per dispatch cycle: clean time, idle advance
    pool_log.csv       one row per selection: pool sizes (S2 feasibility diagnostic)
"""

from __future__ import annotations

import csv
import logging
import math
from collections import defaultdict
from pathlib import Path

import fedasync_polaris_vi_selection
import fedasync_server
import gpu_tier_sleep
import wen_constants as K
from fedasync_polaris_vi_server import FedAsyncPolarisAggregationStrategy
from sumo_gates import UPLOADED_KEY, UploadGate, delivered
from wen_egreedy import WenSelector
from wen_estimators import Estimator, GpuEstimator
from wen_reported import ReportedView
from wen_selection import WenCoverageGate, WenStage0Gate
from wen_truth import DispatchOutcome, GroundTruth

from plato.config import Config
from plato.servers import fedavg

CLOCK_TOL_S = 1e-6
IDLE_STEP_S = 0.1
SUMMARY_EVERY = 200

DISPATCH_FIELDS = [
    "run_id", "scenario", "arm", "seed", "cycle", "round", "client_id", "veh_id",
    "tier", "t_dispatch", "x_dispatch", "V_k", "T_hat_train", "T_hat_up",
    "T_hat_total", "tau_hat", "s_tau_hat", "G_i", "g_is_prior", "score",
    "stage0_pass", "T_train_true", "x_finish", "d_finish", "chi2", "T_up_true",
    "duration", "t_finish_clean", "accepted", "E_train", "E_up",
    "selection_rule", "epsilon", "coin_u", "explored", "n_candidates",
    "rank_of_pick", "V_max", "V_min", "V_pick", "E_hat",
]
MERGE_FIELDS = ["cycle", "round", "client_id", "t_finish_clean", "t_plato",
                "accepted", "order_violation"]
CYCLE_FIELDS = ["cycle", "round", "t_clean", "idle_advance", "n_dispatched"]
POOL_FIELDS = ["cycle", "round", "t", "selectable", "eligible",
               "feasible_gpu", "open_slots"]
STAGE0_FIELDS = ["cycle", "round", "t", "n_eligible", "n_passed", "n_rejected",
                 "n_rej_time", "n_rej_energy", "n_rej_both",
                 "n_rej_xavier", "n_rej_orin", "n_rej_thor", "fallback"]


class ClockMismatchError(RuntimeError):
    """PLATO's clock and the online clean clock disagree. Stop condition."""


class _CsvLog:
    def __init__(self, path: Path, fields: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=fields)
        self._writer.writeheader()

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()


def _pick_fields(record) -> dict:
    """The D1 columns of one dispatch; blank if the pick was not recorded."""
    keys = ("selection_rule", "epsilon", "coin_u", "explored", "n_candidates",
            "rank_of_pick", "V_max", "V_min", "V_pick")
    if record is None:
        return {k: "" for k in keys}
    return {k: ("" if getattr(record, k) is None else getattr(record, k)) for k in keys}


def _fmt(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return repr(value)          # full precision, for the clock replay
    return value


class WenEnvMixin:
    """Environment model for a FedAsync server. Mix in before the server."""

    def install_wen(self, truth: GroundTruth, view: ReportedView,
                    estimator: Estimator | None, arm: str, scenario: str,
                    stage0: bool, run_id: str, seed: int, out_dir: Path,
                    selection: str = "egreedy", epsilon: float | None = 0.2,
                    selector=None, stage0_estimator: Estimator | None = None) -> None:
        self.wen_truth = truth
        self.wen_view = view
        self.wen_estimator = estimator
        self.wen_arm = arm
        self.wen_scenario = scenario
        self.wen_run_id = run_id
        self.wen_seed = seed

        # Clean clock from 0.0 (see the module docstring).
        self.initial_wall_time = 0.0
        self.wall_time = 0.0
        self._online_now = 0.0
        # Number of aggregations that changed the global model (Prompt H: the
        # wen_fresh F3 cache key). Version 0 is the initial model.
        self.global_version = 0

        self._dispatch_cycles = 0
        self._round_was_empty = False
        self._in_flight: dict[int, DispatchOutcome] = {}
        self._reported: dict[int, DispatchOutcome] = {}
        self._frozen_hat: dict[int, float] = {}
        self._report_cache: dict = {}
        self._estimate_cache: dict = {}
        self._pending_idle = 0.0
        self.order_violations = 0
        self.processing_time_removed = 0.0
        self.tally = defaultdict(float)

        out_dir = Path(out_dir)
        self._dispatch_log = _CsvLog(out_dir / "dispatch_log.csv", DISPATCH_FIELDS)
        self._merge_log = _CsvLog(out_dir / "merge_log.csv", MERGE_FIELDS)
        self._cycle_log = _CsvLog(out_dir / "cycle_log.csv", CYCLE_FIELDS)
        self._pool_log = _CsvLog(out_dir / "pool_log.csv", POOL_FIELDS)

        # D1: every arm selects through WenSelector. The Polaris selector, when
        # the arm has one, is only the scorer (and the proportional ablation).
        current = self.client_selection_strategy
        scorer = current if hasattr(current, "score_candidates") else None
        # The wen arm (Prompt E) brings its own selection rule instead of D1.
        self.wen_selector = selector or WenSelector(scorer, selection, epsilon, seed)
        inner = self.wen_selector
        self.wen_stage0 = None
        if stage0:
            # Stage 0's estimator: the arm's own (existing arms), or one handed in
            # (Study 2 plug-in arms always use the gpu estimator, Prompt I S2-4).
            stage0_estimator = stage0_estimator or estimator
            if stage0_estimator is None:
                raise ValueError("[WenEnv] Stage 0 needs an estimator.")
            self.wen_stage0 = WenStage0Gate(inner, stage0_estimator, scenario=scenario)
            self._stage0_log = _CsvLog(out_dir / "stage0_log.csv", STAGE0_FIELDS)
            inner = self.wen_stage0
        diagnostic = GpuEstimator(truth.flops_fwd, truth.model_bits)
        self.client_selection_strategy = WenCoverageGate(inner, scenario, diagnostic)
        self.aggregation_strategy = UploadGate(self.aggregation_strategy)

        selector = self.polaris_selector()
        if estimator is not None:
            if selector is None:
                raise ValueError(f"[WenEnv] arm {arm!r} needs the Polaris selector.")
            selector._predict_staleness = self._predict_total
            selector._predict_staleness_unit = "seconds"
            selector.busy_predict_seconds = self._busy_total
        logging.info("[WenEnv] installed: scenario=%s arm=%s stage0=%s estimator=%s "
                     "selection=%s epsilon=%s", scenario, arm, stage0,
                     getattr(estimator, "name", None), selection,
                     getattr(self.wen_selector, "epsilon", None))

    # -- helpers ---------------------------------------------------------

    def sumo_time(self) -> float:
        return self.wall_time - self.initial_wall_time

    def polaris_selector(self):
        candidate = self.client_selection_strategy
        for _ in range(10):
            if candidate is None:
                return None
            if hasattr(candidate, "last_tau_hats"):
                return candidate
            candidate = getattr(candidate, "inner", None)
        return None

    def wen_report(self, client_id: int, t: float):
        key = (client_id, t)
        if key not in self._report_cache:
            if len(self._report_cache) > 10_000:
                self._report_cache.clear()
            self._report_cache[key] = self.wen_view.report(client_id, t)
        return self._report_cache[key]

    def wen_estimate(self, client_id: int, t: float):
        key = (client_id, t)
        if key not in self._estimate_cache:
            if len(self._estimate_cache) > 10_000:
                self._estimate_cache.clear()
            self._estimate_cache[key] = self.wen_estimator.estimate(
                self.wen_report(client_id, t))
        return self._estimate_cache[key]

    def _predict_total(self, client_id: int, _context) -> float:
        return self.wen_estimate(client_id, self.sumo_time()).total

    def _busy_total(self, client_id: int, _context) -> float:
        return self._frozen_hat[client_id]

    def _set_startup_t_round(self) -> None:
        if self.wen_estimator is None:
            return
        totals = [self.wen_estimate(c, 0.0).total
                  for c in range(1, self.total_clients + 1)]
        t_round = gpu_tier_sleep.compute_t_round_from_times(totals, self.clients_per_round)
        self.context.state["T_round_seconds"] = t_round
        logging.info("[WenEnv] startup T_round = %.6f s (mean T_hat_total %.6f s / C=%d, "
                     "model=%s)", t_round, sum(totals) / len(totals),
                     self.clients_per_round, gpu_tier_sleep.t_round_model())

    # -- lifecycle ---------------------------------------------------------

    def configure(self) -> None:
        super().configure()
        clients = Config().clients
        if not getattr(clients, "comm_simulation", True) or \
                getattr(clients, "compute_comm_time", False):
            raise ValueError("[WenEnv] needs comm_simulation = true and "
                             "compute_comm_time off, so comm_time is exactly 0.")
        if gpu_tier_sleep.t_round_model() != "queue":
            raise ValueError("[WenEnv] [gpu_tier] t_round_model must be 'queue'.")

    def _selectable(self) -> set[int]:
        pending = {info[2]["client_id"] for info in self.reported_clients}
        return set(range(1, self.total_clients + 1)) - set(self.training_clients) - pending

    def _advance_to_coverage(self) -> float:
        """S2: with nothing in flight and nobody dispatchable in coverage, step
        the clock to the first moment a selectable vehicle is inside."""
        if self.wen_scenario != "S2" or self.training_clients:
            return 0.0
        lo, hi = K.S2_COVERAGE_M
        selectable = self._selectable()
        t = self.sumo_time()
        if any(lo <= self.wen_report(c, t).x <= hi for c in selectable):
            return 0.0
        waits = []
        for c in selectable:
            rv = self.wen_report(c, t)
            gap = lo - rv.x if rv.x < lo else K.ROAD_LENGTH_M - rv.x + lo
            waits.append(gap / rv.speed)
        delta = min(waits)
        while not any(lo <= self.wen_report(c, t + delta).x <= hi for c in selectable):
            delta += IDLE_STEP_S
        self.wall_time += delta
        self._online_now += delta
        logging.info("[WenEnv] nobody dispatchable in coverage at t=%.3f s; idled "
                     "%.3f s", t, delta)
        return delta

    async def _select_clients(self, for_next_batch: bool = False) -> None:
        if not for_next_batch:
            idle = self._advance_to_coverage()
            if self._dispatch_cycles == 0:
                self._set_startup_t_round()
            self._dispatch_cycles += 1
            if self._round_was_empty:
                self.current_round -= 1
                self._round_was_empty = False
            if abs(self.sumo_time() - self._online_now) > CLOCK_TOL_S:
                raise ClockMismatchError(
                    f"dispatch cycle {self._dispatch_cycles}: PLATO t = "
                    f"{self.sumo_time()!r}, online t = {self._online_now!r}")
            self._pending_idle = idle
            before = len(self._in_flight)
            await super()._select_clients(for_next_batch=for_next_batch)
            self._cycle_log.write({
                "cycle": self._dispatch_cycles, "round": self.current_round,
                "t_clean": repr(self._online_now), "idle_advance": repr(idle),
                "n_dispatched": len(self._in_flight) - before,
            })
            return
        await super()._select_clients(for_next_batch=for_next_batch)

    def record_pool(self, t, selectable, eligible, feasible, slots) -> None:
        self._pool_log.write({
            "cycle": self._dispatch_cycles, "round": self.current_round, "t": repr(t),
            "selectable": selectable, "eligible": eligible, "feasible_gpu": feasible,
            "open_slots": slots,
        })

    def wen_is_eligible(self, client_id: int, t: float) -> bool:
        """The environment's eligibility rule: idle (not training, no report
        awaiting merge), and in S2 inside coverage at time t."""
        if client_id not in self._selectable():
            return False
        if self.wen_scenario == "S1":
            return True
        lo, hi = K.S2_COVERAGE_M
        return lo <= self.wen_report(client_id, t).x <= hi

    def record_stage0(self, t, stats: dict) -> None:
        self._stage0_log.write({"cycle": self._dispatch_cycles, "round": self.current_round,
                                "t": repr(t), **stats})

    def record_dispatches(self, selected: list[int], context) -> None:
        t = self._online_now
        selector = self.polaris_selector()
        tau_hats = selector.last_tau_hats if selector is not None else {}
        polaris = context.state.get("polaris", {})
        bounds = polaris.get("local_gradient_bounds")
        agg = getattr(self.aggregation_strategy, "inner", self.aggregation_strategy)
        unexplored = getattr(agg, "unexplored_clients", None) or set()
        stage0_pass = self.wen_stage0.last_pass if self.wen_stage0 else {}
        picks = self.wen_selector.last_picks

        for cid in selected:
            if cid in self._in_flight:
                raise RuntimeError(f"[WenEnv] client {cid} dispatched while in flight")
            rv = self.wen_report(cid, t)
            est = self.wen_estimate(cid, t) if self.wen_estimator else None
            outcome = self.wen_truth.dispatch(cid, t, self.current_round)
            if self.wen_scenario == "S2" and not (
                    K.S2_COVERAGE_M[0] <= outcome.x_dispatch <= K.S2_COVERAGE_M[1]):
                raise RuntimeError(f"[WenEnv] S2 dispatched {cid} outside coverage")
            self._in_flight[cid] = outcome
            if est is not None:
                self._frozen_hat[cid] = est.total

            tau = tau_hats.get(cid)
            s_tau = 1.0 / math.sqrt(1.0 + max(tau, 0.0)) if tau is not None else None
            g_i = float(bounds[cid - 1]) if bounds is not None else None
            self._dispatch_log.write({k: _fmt(v) for k, v in {
                "run_id": self.wen_run_id, "scenario": self.wen_scenario,
                "arm": self.wen_arm, "seed": self.wen_seed,
                "cycle": self._dispatch_cycles, "round": self.current_round,
                "client_id": cid, "veh_id": outcome.veh_id, "tier": outcome.tier,
                "t_dispatch": t, "x_dispatch": outcome.x_dispatch, "V_k": rv.speed,
                "T_hat_train": est.T_train if est else "",
                "T_hat_up": est.T_up if est else "",
                "T_hat_total": est.total if est else "",
                "tau_hat": tau if tau is not None else "",
                "s_tau_hat": s_tau if s_tau is not None else "",
                "G_i": g_i if g_i is not None else "",
                "g_is_prior": (cid - 1) in unexplored if bounds is not None else "",
                "score": g_i * s_tau if (g_i is not None and s_tau is not None) else "",
                "stage0_pass": stage0_pass.get(cid, "") if self.wen_stage0 else "",
                "T_train_true": outcome.T_train_true, "x_finish": outcome.x_finish,
                "d_finish": outcome.d_finish, "chi2": outcome.chi2,
                "T_up_true": outcome.T_up_true, "duration": outcome.duration,
                "t_finish_clean": outcome.t_finish_clean,
                "accepted": outcome.accepted,
                "E_train": outcome.E_train, "E_up": outcome.E_up,
                **_pick_fields(picks.get(cid)),
                "E_hat": est.E if est else "",
            }.items()})

    # -- report path -----------------------------------------------------------

    async def process_client_info(self, client_id, sid) -> None:
        report = self.reports[sid]
        cid = report.client_id
        outcome = self._in_flight.pop(cid)
        if abs(self.training_clients[cid]["start_time"] - outcome.t_dispatch) > CLOCK_TOL_S:
            raise ClockMismatchError(
                f"client {cid}: PLATO start {self.training_clients[cid]['start_time']!r}"
                f" != online dispatch {outcome.t_dispatch!r}")
        self.processing_time_removed += float(getattr(report, "processing_time", 0.0))
        report.processing_time = 0.0
        report.training_time = outcome.duration
        self._reported[cid] = outcome
        self.context.state.setdefault(UPLOADED_KEY, {})[cid] = outcome.accepted

        tier = outcome.tier
        self.tally[f"dispatched_{tier}"] += 1
        if not outcome.accepted:
            self.tally[f"wasted_{tier}"] += 1
            self.tally["waste_gpu_s"] += outcome.T_train_true
            self.tally["waste_up_s"] += outcome.T_up_true
            self.tally["waste_energy_j"] += outcome.E_train + outcome.E_up
        self.tally["energy_j"] += outcome.E_train + outcome.E_up
        await super().process_client_info(client_id, sid)

    async def _process_reports(self) -> None:
        upcoming_cycle = self._dispatch_cycles + 1
        for update in self.updates:
            outcome = self._reported.pop(update.client_id)
            earlier = [o for o in list(self._in_flight.values()) + list(self._reported.values())
                       if o.t_finish_clean < outcome.t_finish_clean]
            if earlier:
                self.order_violations += 1
                logging.warning("[WenEnv] merge-order residual: client %d (t=%.6f) "
                                "merged before %s", update.client_id,
                                outcome.t_finish_clean,
                                [(o.client_id, round(o.t_finish_clean, 6)) for o in earlier])
            # PLATO sets wall_time to each popped update's finish time
            # (servers/base.py:1159), even when a merge-order residual makes
            # that earlier than the previous merge; mirror it exactly.
            self._online_now = outcome.t_finish_clean
            self._merge_log.write({
                "cycle": upcoming_cycle, "round": self.current_round,
                "client_id": update.client_id,
                "t_finish_clean": repr(outcome.t_finish_clean),
                "t_plato": repr(self.sumo_time()), "accepted": int(outcome.accepted),
                "order_violation": int(bool(earlier)),
            })
        if self.updates and abs(self.sumo_time() - self._online_now) > CLOCK_TOL_S:
            raise ClockMismatchError(
                f"merge before cycle {upcoming_cycle}: PLATO t = {self.sumo_time()!r},"
                f" online t = {self._online_now!r}")

        if self._reports_seen() % SUMMARY_EVERY == 0:
            self._log_summary()

        if not delivered(self.updates, self.context):
            self._round_was_empty = True
            logging.info("[WenEnv] round %d produced no aggregation: all %d update(s) "
                         "were rejected by coverage.", self.current_round,
                         len(self.updates))
            return
        await super()._process_reports()
        self.global_version += 1

    # -- summaries ---------------------------------------------------------

    def _reports_seen(self) -> int:
        return int(sum(v for k, v in self.tally.items() if k.startswith("dispatched_")))

    def _log_summary(self) -> None:
        parts = []
        for tier in ("xavier", "orin", "thor"):
            d = self.tally.get(f"dispatched_{tier}", 0)
            w = self.tally.get(f"wasted_{tier}", 0)
            parts.append(f"{tier}={int(w)}/{int(d)}")
        logging.info(
            "[WenEnv] summary round=%d cycles=%d t=%.3f waste %s gpu_s=%.3f up_s=%.3f "
            "E=%.1f J (total E %.1f J) order_residuals=%d processing_time_removed=%.3f s",
            self.current_round, self._dispatch_cycles, self._online_now,
            " ".join(parts), self.tally["waste_gpu_s"], self.tally["waste_up_s"],
            self.tally["waste_energy_j"], self.tally["energy_j"],
            self.order_violations, self.processing_time_removed,
        )

    def server_will_close(self) -> None:
        self._log_summary()
        if self.wen_stage0 is not None:
            self.wen_stage0.log_summary()
        super().server_will_close()


class WenBaselineServer(WenEnvMixin, fedasync_server.Server):
    """The `random` arm: uniform choice among eligible idle vehicles."""

    def __init__(self, wen: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.install_wen(**wen)


class WenPolarisServer(WenEnvMixin, fedasync_server.Server):
    """V_i = G_i / sqrt(1 + tau_hat) with tau_hat from T_hat_total (or observed).

    Builds its strategies directly, like fedasync_polaris_vi_server.Server does,
    but WITHOUT that class's startup call into gpu_tier_sleep's tier predictor:
    the Wen path computes nothing from a tier file through gpu_tier_sleep, so
    no anchor key can be read on it (tests/test_wen_no_anchor.py).
    """

    def __init__(self, wen: dict, model=None, datasource=None, algorithm=None,
                 trainer=None, callbacks=None) -> None:
        server_cfg = Config().server
        sizes = {c: K.partition_size(c) for c in range(1, K.N_CLIENTS + 1)}
        aggregation = FedAsyncPolarisAggregationStrategy(
            alpha=float(server_cfg.polaris_alpha),
            initial_gradient_bound=float(getattr(server_cfg,
                                                 "polaris_initial_gradient_bound", 0.5)),
            initial_staleness=float(getattr(server_cfg, "polaris_initial_staleness", 0.01)),
            t_round_seconds=0.0,           # set at the first dispatch cycle
            partition_sizes=sizes,
            local_epochs=K.LOCAL_EPOCHS,
            pool_mean_T_seconds=0.0,
            clients_per_round=int(Config().clients.per_round),
        )
        if wen["estimator"] is not None:
            selection = fedasync_polaris_vi_selection.PolarisViClientSelectionStrategy(
                predict_staleness=lambda cid, ctx: 0.0,    # replaced in install_wen
                predict_staleness_unit="seconds",
            )
        else:
            selection = fedasync_polaris_vi_selection.PolarisViClientSelectionStrategy()
        fedavg.Server.__init__(
            self, model=model, datasource=datasource, algorithm=algorithm,
            trainer=trainer, callbacks=callbacks,
            aggregation_strategy=aggregation, client_selection_strategy=selection,
        )
        self.install_wen(**wen)


__all__ = ["WenBaselineServer", "WenPolarisServer", "ClockMismatchError"]
