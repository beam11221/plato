"""
The open-population FedAsync server: slots, recycles, and per-occupant physics.

`sumo_close_loop/sumo_server.py` can assume one client id == one vehicle for
the whole run.  Here 10,523 vehicles pass through 20 slots, so nothing that
describes a vehicle may be read off the client id.  Stage 7b routes every such
read through ONE lookup, `occupant_identity.IdentityResolver`:

  tier, predictor input -> the occupant identity at scoring time; the identity
                           frozen at dispatch for busy-pool terms
  sleep time            -> the dispatched identity's, rewritten into the report
                           before PLATO turns it into a finish time
  data shard, seed      -> sent to the client in the server response
  observation count     -> keyed on the identity, delivered reports only

and the identity is the trace vehicle at p_recur = 0, the slot's fixed fleet
vehicle at p_recur = 1.  Only the coverage verdict stays with the TRACE
vehicle, because that is mobility, not identity.

None of this touches anything under `plato/`.  The interception points are
hooks PLATO already exposes: `process_client_info` and
`customize_server_response` (`plato/servers/base.py:1493`).

What is deliberately NOT here
-----------------------------
The Stage 0 feasibility filter.  This cell tests the scoring claim, and the
filter has already been shown to dominate the selector when it is on, so it
stays off (the stage prompt's Step 3).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import fedasync_polaris_vi_server
import fedasync_server
import gpu_tier_sleep
from occupant_identity import IdentityResolver, OccupantIdentity
from open_stage0 import FLEET_MEAN_T_HAT_S, OpenStage0Config, OpenStage0Gate
from stage7_population import apply_recycle, unexplored_owner

# Stage 9: the reactive arm's tau for a never-seen vehicle.
UNSEEN_TAU_DEFAULT = "const_0.01"
UNSEEN_TAU_RULES = frozenset({UNSEEN_TAU_DEFAULT, "fleet_running_mean"})
from sumo_gates import UPLOADED_KEY, UploadGate, delivered

from plato.config import Config
from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext

SUMMARY_EVERY = 500


class OpenSlotSelectionGate(ClientSelectionStrategy):
    """Restrict selection to slots that currently hold an in-coverage vehicle.

    The open-population counterpart of `sumo_gates.RsuSelectionGate`.  A slot
    is occupied exactly while its vehicle is inside coverage, so "in range now"
    and "has a slot" are the same predicate here and there is only one gate.
    """

    def __init__(self, inner: ClientSelectionStrategy, server) -> None:
        self.inner = inner
        self.server = server

    def setup(self, context: ServerContext) -> None:
        self.inner.setup(context)

    def teardown(self, context: ServerContext) -> None:
        self.inner.teardown(context)

    def on_clients_selected(self, selected_clients, context: ServerContext) -> None:
        self.inner.on_clients_selected(selected_clients, context)

    def on_reports_received(self, updates, context: ServerContext) -> None:
        # A report that never reached the RSU teaches the server nothing —
        # the same reasoning as the closed-loop gate, and it matters more here
        # because the `observed` arm is the one under test.
        arrived = delivered(updates, context)
        self.server.record_delivered_staleness(arrived)
        self.inner.on_reports_received(arrived, context)

    @staticmethod
    def _open_slots(server) -> int:
        training = len(getattr(server, "training_clients", {}) or {})
        awaiting_merge = len(getattr(server, "reported_clients", []) or [])
        return max(0, server.clients_per_round - training - awaiting_merge)

    def select_clients(self, clients_pool, clients_count, context) -> list[int]:
        server = context.server
        occupied = set(server.timeline.occupied_slots())
        eligible = [cid for cid in clients_pool if cid in occupied]
        open_slots = self._open_slots(server)

        logging.info(
            "[Stage7] round=%d t=%.1f s: %d of %d slot(s) occupied and idle; "
            "%d dispatch slot(s) open.",
            getattr(server, "current_round", -1), server.sumo_time(),
            len(eligible), len(clients_pool), open_slots,
        )
        selected = self.inner.select_clients(eligible, open_slots, context)
        server.record_dispatches(selected, context)
        return selected


class OpenPopulationMixin:
    """Drives the slot allocator from the simulated clock and keeps the
    vehicle's physics attached to whichever slot it is holding."""

    def install_open_population(
        self, timeline, base_seed: int, p_recur: float, resolver: IdentityResolver,
        stage0: OpenStage0Config | None = None, positions=None,
    ) -> None:
        self.timeline = timeline
        self.base_seed = base_seed
        self.p_recur = p_recur
        # Stage 7b: the ONE answer to "who is in slot k" -- see occupant_identity.
        self.resolver = resolver
        self._startup_t_round_done = False
        # Stage 9: running mean of every DELIVERED staleness observation, over
        # all vehicles.  Read only under observed_unseen_tau = fleet_running_mean.
        self._tau_sum = 0.0
        self._tau_count = 0
        self.unseen_tau_rule = UNSEEN_TAU_DEFAULT

        self.dispatched_per_tier: dict[str, int] = defaultdict(int)
        self.wasted_per_tier: dict[str, int] = defaultdict(int)
        self._reports_seen = 0
        self._dispatch_cycles = 0
        self._round_was_empty = False
        self.state_resets = 0

        # The vehicle that held each slot at DISPATCH time.  A slot can be
        # recycled to somebody else before the report comes back, so the
        # coverage verdict and the sleep rewrite must both be evaluated
        # against the vehicle that was actually dispatched, not the current
        # occupant.  Getting this wrong would silently score one vehicle's
        # work against another's mobility.
        self._dispatch_vehicle: dict[int, str] = {}
        # The occupant IDENTITY frozen at dispatch: tier, sleep, partition and
        # seed of the job in flight all come from here, never from whoever holds
        # the slot when the report comes back.
        self._dispatch_identity: dict[int, OccupantIdentity] = {}
        # Tier the proactive predictor used for each slot's latest score.
        self._predictor_tier: dict[int, str] = {}
        # DELIVERED reports per identity -- what the server has actually learnt
        # from it.  Keyed on identity, so a new vehicle starts at 0 by
        # construction at p_recur = 0 and a returning slot identity keeps its
        # count at p_recur = 1.  (Stage 7 counted undelivered reports too, and
        # keyed on the slot.)
        self._observations: dict[str, int] = defaultdict(int)

        # Stage 10: realised training durations (report arrival - dispatch, on
        # the server clock), DELIVERED reports only -- a wasted dispatch never
        # tells the server how long it took.  Per identity, and fleet-wide.
        self._dur_sum: dict[str, float] = defaultdict(float)
        self._dur_n: dict[str, int] = defaultdict(int)
        self._fleet_dur_sum = 0.0
        self._fleet_dur_n = 0

        # Gate order (Stage 5, unchanged): slot gate (in-range-now, pool already
        # excludes busy) -> Stage 0 -> the arm's own strategy.  stage0 None or
        # disabled -> no gate: every Stage 7-9 config is unchanged.
        strategy = self.client_selection_strategy
        self.stage0_gate = None
        if stage0 is not None and stage0.enabled:
            self.stage0_gate = OpenStage0Gate(strategy, self, stage0, positions)
            strategy = self.stage0_gate
            logging.info("[Stage0] open-population gate installed: t_hat=%s r_i=vmax_distance "
                         "rsu_x=%.1f radius=%.1f v_max=%.6f",
                         stage0.t_hat_source, stage0.rsu_x_m, stage0.radius_m, stage0.v_max_mps)
        self.client_selection_strategy = OpenSlotSelectionGate(strategy, self)
        self.install_returning_population()
        self.aggregation_strategy = UploadGate(self.aggregation_strategy)

    # -- clock ----------------------------------------------------------

    def sumo_time(self) -> float:
        return self.wall_time - self.initial_wall_time

    def _advance_population(self) -> None:
        """Apply coverage arrivals/departures up to now, then the p_recur policy."""
        for event in self.timeline.advance_to(self.sumo_time()):
            apply_recycle(self.context, event, self)
            if event.state_reset:
                self.state_resets += 1
                # The observation count needs no wipe: it is keyed on the
                # identity, and at p_recur = 0 every recycle brings a new one.

    def _selectable_slots(self) -> set[int]:
        """Occupied slots PLATO would actually offer the selector.

        PLATO withholds two groups (servers/base.py:567-585): clients still
        training, and clients whose report has arrived but has not been merged
        yet.  An occupied slot in either group is NOT dispatchable, so asking
        "is any slot occupied?" is the wrong question — see
        `_advance_clock_to_occupancy`.
        """
        pending_merge = {info[2]["client_id"] for info in self.reported_clients}
        return (
            set(self.timeline.occupied_slots())
            - set(self.training_clients)
            - pending_merge
        )

    def _advance_clock_to_occupancy(self) -> None:
        """Idle forward until a *dispatchable* slot exists.

        The question has to be asked about the SELECTABLE set, not about
        occupancy.  This is the same bug `sumo_server._advance_clock_to_
        coverage` documents in the closed-loop cell -- the cpu arm hung at
        round 719 on 2026-09-06 because the guard checked all 20 vehicles,
        three were in range but sitting in the reported-not-yet-merged queue,
        and it concluded no idling was needed while the selector was handed an
        empty pool.  Stage 7 reproduced it exactly: `cpu_open` froze at round
        2,766, t = 5,239.5 s, logging "0 of 17 slot(s) occupied and idle" with
        nothing training, and sat there for 14 hours.

        Looping rather than a single jump: the next coverage event may be a
        departure, or an arrival that lands on a slot which is still withheld,
        so one step forward does not necessarily produce a dispatchable slot.
        """
        if self.training_clients:
            return

        idled = 0.0
        steps = 0
        while not self._selectable_slots():
            next_event = self.timeline.peek_next_time()
            if next_event is None:
                return
            delta = next_event - self.sumo_time()
            if delta > 0:
                self.wall_time += delta
                idled += delta
            self._advance_population()
            steps += 1
            if steps > 100_000:
                logging.warning(
                    "[Stage7] gave up idling after %d coverage events at "
                    "t=%.1f s; no dispatchable slot appeared.",
                    steps, self.sumo_time(),
                )
                return

        if idled > 0:
            logging.info(
                "[Stage7] No dispatchable slot; idled %.1f s over %d event(s) "
                "to t=%.1f s.", idled, steps, self.sumo_time(),
            )

    # -- lifecycle hooks -------------------------------------------------

    def configure(self) -> None:
        super().configure()
        clients = Config().clients
        if not getattr(clients, "comm_simulation", True):
            raise ValueError("[Stage7] requires [clients] comm_simulation = true.")
        if getattr(clients, "compute_comm_time", False):
            raise ValueError("[Stage7] requires compute_comm_time off.")

    async def _select_clients(self, for_next_batch: bool = False) -> None:
        if not for_next_batch:
            self._advance_population()
            self._advance_clock_to_occupancy()
            self._advance_population()
            if not self._startup_t_round_done:
                self._set_startup_t_round()
                self._startup_t_round_done = True
            self._dispatch_cycles += 1
            if self._round_was_empty:
                self.current_round -= 1
                self._round_was_empty = False
        await super()._select_clients(for_next_batch=for_next_batch)

    async def _process_reports(self) -> None:
        if not delivered(self.updates, self.context):
            self._round_was_empty = True
            logging.info("[Stage7] Round %d produced no aggregation: all %d "
                         "update(s) were lost to mobility.",
                         self.current_round, len(self.updates))
            return
        await super()._process_reports()

    # -- the occupant identity on the server side ---------------------------

    def _polaris_selector(self):
        """The Polaris V_i strategy under the gates, or None (baseline).

        Walks the whole `.inner` chain: with Stage 0 installed the selector is
        two levels down, and a one-level lookup would silently skip the
        predictor install, the unseen-tau rule and the tau_hat logging.
        """
        candidate = getattr(self.client_selection_strategy, "inner", None)
        for _ in range(10):
            if candidate is None:
                return None
            if hasattr(candidate, "last_tau_hats"):
                return candidate
            candidate = getattr(candidate, "inner", None)
        return None

    # -- Stage 11: partial recurrence -- state follows the identity ------------

    def install_returning_population(self) -> None:
        """Hook the p_return identity policy into the allocator (Stage 11).

        The identity is decided at the moment of each allocation; everything
        the server holds per SLOT is snapshotted when a commuter leaves and
        restored into whatever slot it next occupies.  Per-IDENTITY state
        (observation counts, training-time history) needs nothing: it is keyed
        on the identity already.  A transient starts from the neutral prior,
        exactly as at p_recur = 0.  A commuter's first appearance also starts
        from the prior (there is nothing to carry yet).
        """
        self.returning = getattr(self.resolver, "returning", None)
        if self.returning is None:
            return
        self._commuter_state: dict[str, dict] = {}
        self.commuter_reentries = self.commuter_first_entries = self.transient_allocations = 0
        allocator = self.timeline.allocator
        allocator.on_assign = self._on_assign
        allocator.on_release = self._on_release
        logging.info("[Stage11] partial recurrence: p_return=%.2f commuter pool K=%d "
                     "(identity RNG from the physics seed)",
                     self.returning.p_return, len(self.returning.commuters))

    def _slot_state(self, idx: int):
        """The slot-indexed Polaris arrays, or None where absent or too short
        (baseline has none; the same bounds guard as reset_slot_state)."""
        polaris = self.context.state.get("polaris", {}) if getattr(self, "context", None) else {}
        bounds, stal = polaris.get("local_gradient_bounds"), polaris.get("local_stalenesses")
        bounds = bounds if bounds is not None and idx < len(bounds) else None
        stal = stal if stal is not None and idx < len(stal) else None
        return bounds, stal, polaris.get("staleness_predictions"), unexplored_owner(self)

    def _on_release(self, slot: int, veh_id: str) -> None:
        identity, kind = self.returning.release(slot)
        if kind != "commuter":
            return
        idx = slot - 1
        bounds, stal, _pred, owner = self._slot_state(idx)
        self._commuter_state[identity.identity_id] = {
            "G": None if bounds is None else float(bounds[idx]),
            "tau": None if stal is None else float(stal[idx]),
            "unexplored": owner is not None and idx in owner.unexplored_clients,
        }

    def _on_assign(self, slot: int, veh_id: str, tier: str) -> bool:
        identity, kind, exhausted = self.returning.assign(
            slot, veh_id, tier, self.resolver.base_seed, self.resolver.slots)
        carried = kind == "commuter" and identity.identity_id in self._commuter_state
        if carried:
            snap = self._commuter_state[identity.identity_id]
            idx = slot - 1
            bounds, stal, pred, owner = self._slot_state(idx)
            if stal is not None and snap["tau"] is not None:
                stal[idx] = snap["tau"]
            if isinstance(pred, dict):
                pred.pop(slot, None)  # a prediction in flight belongs to the previous occupant
            if owner is not None:
                if snap["unexplored"]:
                    owner.unexplored_clients.add(idx)
                    if bounds is not None:
                        proxy = getattr(owner, "last_proxy", None)
                        bounds[idx] = snap["G"] if proxy is None else proxy
                else:
                    owner.unexplored_clients.discard(idx)
                    if bounds is not None and snap["G"] is not None:
                        bounds[idx] = snap["G"]
            self.commuter_reentries += 1
        elif kind == "commuter":
            self.commuter_first_entries += 1
        else:
            self.transient_allocations += 1
        logging.info(
            "[Stage11] alloc t=%.3f slot=%d trace_veh_id=%s identity_id=%s kind=%s tier=%s "
            "state_carried=%d exhausted=%d",
            self.timeline.event_t, slot, veh_id, identity.identity_id, kind, identity.tier,
            int(carried), int(exhausted))
        # False -> apply_recycle resets the slot to the neutral prior (transient,
        # or a commuter's first appearance); True -> the restored state stays.
        return carried

    # -- Stage 10: observed training time, for the reactive filter -------------

    def observed_duration_mean(self, identity_id: str) -> float | None:
        n = self._dur_n.get(identity_id, 0)
        return self._dur_sum[identity_id] / n if n else None

    def fleet_duration_mean(self) -> float:
        if self._fleet_dur_n == 0:
            return FLEET_MEAN_T_HAT_S
        return self._fleet_dur_sum / self._fleet_dur_n

    def _proactive(self) -> bool:
        selector = self._polaris_selector()
        return selector is not None and \
            getattr(selector, "_predict_staleness_unit", "rounds") == "seconds"

    @staticmethod
    def predict_seconds_for_tier(tier: str) -> float:
        """The arm's datasheet prediction for one tier -- the GPU (or CPU)
        equation, unchanged; only its input now comes from the identity."""
        if gpu_tier_sleep.oracle_sleeps() is not None:
            raise ValueError("[Stage7b] gpu_oracle is keyed on client_id; not supported.")
        computer = gpu_tier_sleep.get_computer()
        partition_size = int(getattr(Config().data, "partition_size", 500))
        epochs = int(getattr(Config().trainer, "epochs", 1))
        return gpu_tier_sleep._predicted_seconds(
            computer.tier_by_id(tier), partition_size, epochs
        )

    def _occupant_predict_seconds(self, slot: int, _context) -> float:
        """Candidate scoring: the TOPS the occupant's beacon reply would carry."""
        identity = self.resolver.of_slot(slot)
        if identity is None:
            raise RuntimeError(f"[Stage7b] scoring empty slot {slot}.")
        self._predictor_tier[slot] = identity.tier
        return self.predict_seconds_for_tier(identity.tier)

    def _dispatched_predict_seconds(self, slot: int, _context) -> float:
        """Busy-pool terms: the identity frozen at dispatch, because the slot
        may have been recycled while that job trains."""
        return self.predict_seconds_for_tier(self._dispatch_identity[slot].tier)

    # -- Stage 9: what `observed` assumes about a never-seen vehicle ---------

    def record_delivered_staleness(self, updates) -> None:
        for update in updates:
            self._tau_sum += float(getattr(update, "staleness", 0.0))
            self._tau_count += 1

    def tau_running_mean(self) -> float:
        """Mean delivered staleness so far; C - 1 before the first report."""
        if self._tau_count == 0:
            return float(self.clients_per_round - 1)
        return self._tau_sum / self._tau_count

    def _identity_unseen(self, slot: int) -> bool:
        identity = self.resolver.of_slot(slot)
        return identity is not None and self._observations.get(identity.identity_id, 0) == 0

    def _observed_or_running_mean(self, slot: int, context) -> float:
        if self._identity_unseen(slot):
            return self.tau_running_mean()
        return self._observed_staleness_default(slot, context)

    def install_unseen_tau_rule(self) -> None:
        """[server] observed_unseen_tau: const_0.01 (default, Stages 3-8) or
        fleet_running_mean (Stage 9).  Reactive (`observed`) arm only: the
        proactive arms compute tau from spec for every vehicle, and baseline
        has no tau at all."""
        rule = str(getattr(Config().server, "observed_unseen_tau", UNSEEN_TAU_DEFAULT))
        if rule not in UNSEEN_TAU_RULES:
            raise ValueError(f"[Stage9] observed_unseen_tau={rule!r}; expected one of "
                             f"{sorted(UNSEEN_TAU_RULES)}")
        self.unseen_tau_rule = rule
        selector = self._polaris_selector()
        if rule == UNSEEN_TAU_DEFAULT or selector is None or self._proactive():
            return
        self._observed_staleness_default = selector._predict_staleness
        selector._predict_staleness = self._observed_or_running_mean
        logging.info("[Stage9] observed arm: never-seen vehicles get the fleet running "
                     "mean of delivered staleness (initial C-1 = %d).",
                     self.clients_per_round - 1)

    def install_occupant_predictor(self) -> None:
        """Point the proactive arms' predictor at the occupant identity."""
        if not self._proactive():
            return
        selector = self._polaris_selector()
        selector._predict_staleness = self._occupant_predict_seconds
        selector.busy_predict_seconds = self._dispatched_predict_seconds
        logging.info("[Stage7b] proactive predictor keyed on the occupant identity "
                     "(candidates) and the dispatched identity (busy pool).")

    def _set_startup_t_round(self) -> None:
        """Log the startup T_round the proactive arms will use, and why.

        Stage 7b decision (user, 2026-09-23; STAGE7B_report.md section 4): KEEP the
        fleet-file value the Polaris server computed at init (gpu 2.0808 s, cpu
        3.6202 s) in BOTH regimes -- identical to Stages 3-6.  The audit's
        "initial occupants" rule was implemented, measured and dropped: it gave
        1.74 s at p_recur = 0 and 2.66 s at p_recur = 1, the latter an artefact
        of FIFO slot order (slots 1-15 fill first; 1-10 are xavier), so the tau
        scale would have differed between regimes for reasons unrelated to
        identity churn.  The long-run pool is within 2.7% of the fleet file in
        both regimes.  The initial-occupant value is still computed and logged
        for the record, but it is NOT written to context.state.
        """
        if not self._proactive():
            return
        used = self.context.state.get("T_round_seconds")
        occupants = [self.resolver.of_slot(k) for k in self.timeline.occupied_slots()]
        occupants = [o for o in occupants if o is not None]
        mix = defaultdict(int)
        for o in occupants:
            mix[o.tier] += 1
        initial = gpu_tier_sleep.compute_t_round_from_times(
            [self.predict_seconds_for_tier(o.tier) for o in occupants],
            self.clients_per_round,
        ) if occupants else float("nan")
        logging.info(
            "[Stage7b] startup T_round=%.6f s (fleet file, kept; not overridden). "
            "For the record, %d initial occupant(s) %s would give %.6f s.",
            used, len(occupants), dict(sorted(mix.items())), initial,
        )

    # -- dispatch bookkeeping --------------------------------------------

    def record_dispatches(self, selected: list[int], context) -> None:
        """Freeze the occupant identity of each selected slot, and log it.

        Every field an audit needs is on this one line: which identity was
        dispatched, which tier the predictor scored it on, and the tau_hat V_i
        ACTUALLY used (Stage 7 logged the observed-tau array here even for the
        proactive arms).  `tau_obs` is the observed-tau array, kept separately.
        """
        polaris = context.state.get("polaris", {})
        bounds = polaris.get("local_gradient_bounds")
        stalenesses = polaris.get("local_stalenesses")
        owner = unexplored_owner(self)
        unexplored = owner.unexplored_clients if owner is not None else set()
        proxy = getattr(owner, "last_proxy", None) if owner is not None else None
        basis = getattr(owner, "last_proxy_basis", None) if owner is not None else None
        unseen_rule = getattr(self, "unseen_tau_rule", UNSEEN_TAU_DEFAULT)
        selector = self._polaris_selector()
        used = selector.last_tau_hats if selector is not None else {}

        for slot in selected:
            veh_id = self.timeline.vehicle_in_slot(slot)
            identity = self.resolver.of_slot(slot)
            if veh_id is None or identity is None:
                continue
            self._dispatch_vehicle[slot] = veh_id
            self._dispatch_identity[slot] = identity

            idx = slot - 1
            g_i = float(bounds[idx]) if bounds is not None else float("nan")
            tau_obs = float(stalenesses[idx]) if stalenesses is not None else float("nan")
            tau_hat = float(used[slot]) if slot in used else float("nan")
            at_prior = idx in unexplored
            logging.info(
                "[Stage7] dispatch round=%d slot=%d veh_id=%s tier=%s "
                "observations=%d G_i=%.6f G_i_at_prior=%d tau_hat=%.6f "
                "identity_id=%s pred_tier=%s tau_obs=%.6f partition=%d "
                "sleep_s=%.6f G_proxy=%s G_proxy_basis=%s tau_unseen=%s",
                getattr(self, "current_round", -1), slot, veh_id, identity.tier,
                self._observations.get(identity.identity_id, 0), g_i,
                int(at_prior), tau_hat,
                identity.identity_id, self._predictor_tier.get(slot, "-")
                if self._proactive() else "-",
                tau_obs, identity.partition, identity.sleep_s,
                "-" if proxy is None else f"{proxy:.6f}",
                "-" if basis is None else f"{basis:.6f}",
                f"{self.tau_running_mean():.6f}"
                if unseen_rule == "fleet_running_mean" and not self._proactive() else "-",
            )

    def observations_for_slot(self, slot: int) -> int:
        """Delivered reports from the identity occupying `slot` now."""
        identity = self.resolver.of_slot(slot)
        return 0 if identity is None else self._observations.get(identity.identity_id, 0)

    # -- the report path --------------------------------------------------

    async def process_client_info(self, client_id, sid) -> None:
        report = self.reports[sid]
        reporting_id = report.client_id

        veh_id = self._dispatch_vehicle.get(reporting_id)
        identity = self._dispatch_identity.get(reporting_id)
        tier = identity.tier if identity is not None else None

        if identity is not None:
            # The dispatched IDENTITY's sleep replaces whatever the client
            # reported.  Must happen BEFORE super(), which is where base.py
            # turns training_time into the finish time that drives the whole
            # simulated timeline (plato/servers/base.py:1026).
            report.training_time = identity.sleep_s

        start_time = self.training_clients[reporting_id]["start_time"]
        finish_time = start_time + report.training_time + report.processing_time
        finish_t = finish_time - self.initial_wall_time

        # Ground truth: was the vehicle THAT WAS DISPATCHED still in coverage
        # when it finished?  Evaluated against its own window, because its slot
        # may already belong to somebody else by now.
        raw_in_range = self.timeline.vehicle_in_coverage_at(veh_id, finish_t) \
            if veh_id is not None else False
        self.context.state.setdefault(UPLOADED_KEY, {})[reporting_id] = raw_in_range

        duration = finish_time - start_time
        if raw_in_range and identity is not None:
            self._observations[identity.identity_id] += 1
            self._dur_sum[identity.identity_id] += duration
            self._dur_n[identity.identity_id] += 1
            self._fleet_dur_sum += duration
            self._fleet_dur_n += 1

        self.dispatched_per_tier[tier or "unknown"] += 1
        if not raw_in_range:
            self.wasted_per_tier[tier or "unknown"] += 1

        logging.info(
            "[Stage7] report round=%d slot=%d veh_id=%s tier=%s dispatch_t=%.2f "
            "finish_t=%.2f uploaded=%d raw_in_range=%d sleep=%.6f identity_id=%s "
            "duration=%.6f",
            self.current_round, reporting_id, veh_id, tier,
            start_time - self.initial_wall_time, finish_t,
            int(raw_in_range), int(raw_in_range), report.training_time,
            identity.identity_id if identity is not None else "-", duration,
        )

        self._reports_seen += 1
        if self._reports_seen % SUMMARY_EVERY == 0:
            self._log_summary()

        await super().process_client_info(client_id, sid)

    # -- what the client is told ------------------------------------------

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        """Tell the client which identity it is standing in for this round.

        The client builds its sampler from `stage7_partition` and seeds its
        training run from `stage7_identity` (sumo_open_fedasync.py).  Both come
        from the identity frozen at dispatch.
        """
        server_response = super().customize_server_response(server_response, client_id)
        identity = self._dispatch_identity.get(client_id) or self.resolver.of_slot(client_id)
        if identity is not None:
            server_response["stage7_veh_id"] = self._dispatch_vehicle.get(client_id) or \
                self.timeline.vehicle_in_slot(client_id)
            server_response["stage7_identity"] = identity.identity_id
            server_response["stage7_tier"] = identity.tier
            server_response["stage7_partition"] = identity.partition
        return server_response

    # -- summaries ---------------------------------------------------------

    def _log_summary(self) -> None:
        total_d = sum(self.dispatched_per_tier.values())
        total_w = sum(self.wasted_per_tier.values())
        parts = [
            f"{t}={self.wasted_per_tier[t]}/{self.dispatched_per_tier[t]}"
            for t in sorted(self.dispatched_per_tier)
        ]
        logging.info(
            "[Stage7] waste round=%d cycles=%d total=%d/%d (%.1f%%) %s",
            self.current_round, self._dispatch_cycles, total_w, total_d,
            100 * total_w / max(total_d, 1), "  ".join(parts),
        )

    def server_will_close(self) -> None:
        self._log_summary()
        if getattr(self, "stage0_gate", None) is not None:
            self.stage0_gate.log_summary()
        if getattr(self, "returning", None) is not None:
            r = self.returning
            logging.info(
                "[Stage11] summary p_return=%.2f allocations=%d commuter_allocations=%d "
                "realised_share=%.4f exhaustions=%d commuter_reentries=%d commuter_first=%d "
                "transients=%d identity_digest=%s kind_tiers=%s",
                r.p_return, r.allocations, r.commuter_allocations,
                r.commuter_allocations / max(1, r.allocations), r.exhaustions,
                self.commuter_reentries, self.commuter_first_entries,
                self.transient_allocations, r.digest(), r.kind_tiers)
        logging.info(
            "[Stage7] summary p_recur=%g recycles=%d state_resets=%d "
            "slot_exhausted=%d occupancy_digest=%s",
            self.p_recur, len(self.timeline.allocator.recycles), self.state_resets,
            self.timeline.turned_away, self.timeline.occupancy_digest(),
        )
        super().server_will_close()


class OpenBaselineServer(OpenPopulationMixin, fedasync_server.Server):
    """The `baseline` arm: random selection among occupied slots."""

    def __init__(self, timeline, base_seed: int, p_recur: float, resolver,
                 stage0=None, positions=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.install_open_population(timeline, base_seed, p_recur, resolver, stage0, positions)


class OpenPolarisServer(OpenPopulationMixin, fedasync_polaris_vi_server.Server):
    """The `gpu`, `cpu` and `observed` arms: V_i scoring among occupied slots."""

    def __init__(self, timeline, base_seed: int, p_recur: float, resolver,
                 stage0=None, positions=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.install_open_population(timeline, base_seed, p_recur, resolver, stage0, positions)
        self.install_occupant_predictor()
        self.install_unseen_tau_rule()
