"""
FedAsync servers that know where the vehicles are.

`SumoRsuMixin` adds four things to any FedAsync server in this repo, and
nothing else:

  1. **A clock that can wait.** With `simulate_wall_time = true` the server is
     driven purely by arriving client reports (servers/base.py:1066). If no
     vehicle is in RSU range and none is training, no report will ever arrive
     and the run hangs. The mixin advances the simulated clock to the next
     moment a vehicle enters range instead. Measured on the n20 trace this is
     needed for 3 of 36 150 seconds, but a hang costs a whole run.

  2. **An upload decision.** When a report arrives, the server recomputes the
     finish time exactly as base.py does and asks the trace whether the vehicle
     was still in range at that instant. The verdict goes into
     `context.state["sumo_uploaded"]`, where the gates in sumo_gates.py read it.

  3. **A round that only counts when an aggregation happens.** The federated
     layer and the client layer are kept separate: a client trains and reports
     whether or not the RSU can hear it, but a ROUND exists only when something
     was actually aggregated. A dispatch cycle whose update was lost to mobility
     leaves the round counter where it was -- no weight update, no evaluation,
     no results row -- while the clock still advances and the slot is refilled.
     `_dispatch_cycles` and `current_round` therefore diverge by exactly the
     number of fully-wasted cycles, and both are logged.

  4. **A waste counter.** Every discarded update is compute the fleet spent for
     nothing. This is the headline number the next step (location prediction)
     is meant to reduce, so it is logged per report and summarised per tier.

Stage 5 (prompts/STAGE5_stage0_filter.md) adds an optional fifth thing, a
feasibility PREDICATE rather than a verdict: `install_sumo(..., stage0=...)`
wraps the arm's selection strategy in `stage0_filter.Stage0FeasibilityGate`
before RsuSelectionGate, so a candidate whose predicted remaining dwell time
can't cover its predicted training time is filtered out before scoring ever
sees it. `stage0=None` (every config written before this existed) leaves the
pipeline exactly as it always was.

Two concrete servers are exported, one per arm family:

    SumoBaselineServer   random selection            -> the `baseline` arm
    SumoPolarisServer    V_i selection + G_i tracking -> `gpu`, `cpu`, `observed`

Both are the stock FedAsync servers with the mixin in front, so every arm keeps
its own selection and aggregation logic unchanged.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import fedasync_polaris_vi_server
import fedasync_server
import gpu_tier_sleep
from stage0_filter import Stage0Config, Stage0FeasibilityGate
from sumo_gates import UPLOADED_KEY, RsuSelectionGate, UploadGate, delivered

from plato.config import Config

# How often to print the cumulative waste table, in reports.
SUMMARY_EVERY = 200

# Stage 4 Task 4 (prompts/STAGE4_verify_scoring.md): the two upload policies
# the coverage-gating cell compares. "coverage_gated" is this file's original,
# only behaviour -- an upload from outside RSU range is discarded. Absent the
# key, that is still exactly what happens, so every config written before this
# existed produces a byte-identical run. "always_accept" is the new
# counterfactual: every upload succeeds regardless of range, which isolates
# whether wasted work (as opposed to just a shrunken candidate pool) is what
# breaks the closed-pool conservation law -- the question Task 4 exists to
# answer. An unrecognised value raises, following resolve_staleness_predictor's
# precedent in fedasync_polaris_vi_server.py: a silent fallback here would
# make an "always_accept" typo an undetectable duplicate of "coverage_gated".
UPLOAD_POLICIES = frozenset({"coverage_gated", "always_accept"})
DEFAULT_UPLOAD_POLICY = "coverage_gated"


def resolve_upload_policy(raw) -> str:
    if raw is None:
        return DEFAULT_UPLOAD_POLICY
    name = str(raw).strip().lower()
    if name not in UPLOAD_POLICIES:
        raise ValueError(
            f"[SumoRsu] [sumo] upload_policy = {raw!r} is not recognised; "
            f"expected one of {sorted(UPLOAD_POLICIES)}. Omit the key entirely "
            f"to get {DEFAULT_UPLOAD_POLICY!r}."
        )
    return name


class SumoRsuMixin:
    """RSU coverage gating for a FedAsync server. Mix in before the server."""

    def install_sumo(
        self,
        oracle,
        upload_policy: str = DEFAULT_UPLOAD_POLICY,
        stage0: Stage0Config | None = None,
        position_oracle=None,
    ) -> None:
        """Wrap the already-built strategies with the coverage gates.

        Called at the end of __init__, once the concrete server has installed
        whichever selection and aggregation strategies its arm uses.
        """
        self.oracle = oracle
        self.upload_policy = resolve_upload_policy(upload_policy)
        self.dispatched_per_tier: dict[str, int] = defaultdict(int)
        self.wasted_per_tier: dict[str, int] = defaultdict(int)
        self._reports_seen = 0

        # A dispatch cycle is one pass of select -> train -> report. It becomes a
        # ROUND only if something reached the RSU, so the two counters diverge by
        # exactly the number of fully-wasted cycles. Tracking both makes that
        # visible instead of leaving `round` looking mysteriously slow.
        self._dispatch_cycles = 0
        self._round_was_empty = False

        # Stage 5 (prompts/STAGE5_stage0_filter.md): the feasibility predicate
        # sits BETWEEN the in-range-now gate and the arm's own scoring, so it
        # is built bottom-up -- the arm's original strategy first, then Stage
        # 0 around it, then RsuSelectionGate outermost. stage0=None (every
        # config written before this stage existed) keeps the pipeline exactly
        # what it always was: RsuSelectionGate directly around the arm.
        selection_strategy = self.client_selection_strategy
        self.stage0_gate: Stage0FeasibilityGate | None = None
        if stage0 is not None and stage0.enabled:
            self.stage0_gate = Stage0FeasibilityGate(
                selection_strategy, oracle, stage0, position_oracle
            )
            selection_strategy = self.stage0_gate

        self.client_selection_strategy = RsuSelectionGate(selection_strategy, oracle)
        self.aggregation_strategy = UploadGate(self.aggregation_strategy)

    # ------------------------------------------------------------------
    # Clock
    # ------------------------------------------------------------------

    def sumo_time(self) -> float:
        """Simulated seconds since training began = a timestamp in the trace.

        PLATO seeds `wall_time` with `time.time()`, so elapsed simulated time is
        the difference against `initial_wall_time`. The trace's t = 0 is the end
        of its warm-up, which is where we start.
        """
        return self.wall_time - self.initial_wall_time

    def _selectable_clients(self) -> set[int]:
        """The clients PLATO will actually offer the selector this round.

        It withholds two groups (servers/base.py:567-585): those still training,
        and those whose report has arrived but has not been merged yet. The
        second group is easy to forget -- under `simulate_wall_time` a report
        arrives long before its simulated finish time, so a vehicle can be
        physically silent, in RSU range, and still unselectable.
        """
        pending_merge = {info[2]["client_id"] for info in self.reported_clients}
        return set(self.oracle.client_ids) - set(self.training_clients) - pending_merge

    def _advance_clock_to_coverage(self) -> None:
        """Idle the RSU forward until a *dispatchable* vehicle is in range.

        Only safe -- and only necessary -- when nothing is training: a client in
        flight will report and move the clock on its own.

        The question has to be asked about the selectable set, not the whole
        fleet. On 2026-09-06 the cpu arm hung at round 719 because this checked
        all 20 vehicles: three were in range but sitting in the reported-not-yet-
        merged queue, so the guard concluded no idling was needed while the
        selector was handed an empty pool. Nothing was dispatched, so no report
        could arrive, so nothing would ever wake the server again.
        """
        if self.training_clients:
            return

        selectable = self._selectable_clients()
        t = self.sumo_time()
        if any(self.oracle.in_coverage(cid, t) for cid in selectable):
            return

        t_normalized = self.oracle.normalize(t)
        t_next = self.oracle.next_entry_time(t_normalized, selectable)
        if t_next is None:
            # Past the last entry the trace records. Step to the end of the
            # trace so the next call wraps into a fresh loop.
            delta = self.oracle.t_end - t_normalized + 1.0
        else:
            delta = t_next - t_normalized

        if delta <= 0:
            return

        # Advance by a delta rather than assigning an absolute time, so the
        # clock stays monotonic even when the trace wraps.
        self.wall_time += delta
        logging.info(
            "[SumoRsu] No vehicle in range at t=%.1f s; idling %.1f s to t=%.1f s.",
            t,
            delta,
            self.sumo_time(),
        )

    # ------------------------------------------------------------------
    # Hooks into the server lifecycle
    # ------------------------------------------------------------------

    def configure(self) -> None:
        super().configure()

        # The upload decision needs the finish time, and we recompute it from
        # the report the same way base.py does -- except for comm_time, which
        # base.py sets only after we have run. Under `comm_simulation = true`
        # with no `compute_comm_time` it is exactly zero (servers/base.py:1007),
        # which is what every config in this experiment uses. Fail loudly rather
        # than silently mis-time uploads if that ever changes.
        clients = Config().clients
        if not getattr(clients, "comm_simulation", True):
            raise ValueError(
                "[SumoRsu] requires [clients] comm_simulation = true, so that "
                "comm_time is zero and the upload time can be derived from the "
                "report alone."
            )
        if getattr(clients, "compute_comm_time", False):
            raise ValueError(
                "[SumoRsu] requires [clients] compute_comm_time to be off, so "
                "that comm_time is zero."
            )

    async def _process_reports(self) -> None:
        """Skip the whole round when nothing reached the RSU.

        A round is a federated-layer event: it exists only when an aggregation
        actually happens. If every candidate update was lost to mobility there
        is nothing to aggregate, so there is no round -- no weight update, no
        evaluation, and no row in the results CSV. The dispatch cycle still
        happened, the clock still moved and the slot is still freed; those are
        client-layer facts and they are recorded as waste.
        """
        if not delivered(self.updates, self.context):
            self._round_was_empty = True
            logging.info(
                "[SumoRsu] Round %d produced no aggregation: all %d candidate "
                "update(s) were lost to mobility. The round counter holds.",
                self.current_round,
                len(self.updates),
            )
            return

        await super()._process_reports()

    async def _select_clients(self, for_next_batch: bool = False) -> None:
        if not for_next_batch:
            self._advance_clock_to_coverage()

            self._dispatch_cycles += 1
            if self._round_was_empty:
                # super() is about to do `current_round += 1`. Cancel it here
                # rather than undoing it afterwards, so the clients dispatched
                # below are stamped with the round they will really belong to
                # (servers/base.py:534 and :684 both read current_round).
                self.current_round -= 1
                self._round_was_empty = False

        await super()._select_clients(for_next_batch=for_next_batch)

    async def process_client_info(self, client_id, sid) -> None:
        report = self.reports[sid]
        # A client answering an urgent update request reports its own ID; base.py
        # does the same re-read before using the ID (servers/base.py:1023).
        reporting_id = report.client_id

        start_time = self.training_clients[reporting_id]["start_time"]
        # Mirrors base.py's finish_time, with comm_time = 0 (checked in
        # configure()).
        finish_time = start_time + report.training_time + report.processing_time

        finish_t = finish_time - self.initial_wall_time
        # Ground truth: was the vehicle physically in range when it finished?
        # Always computed, regardless of policy -- Task 4's Stage 0
        # counterfactual (Step D) is reconstructed from this after the fact,
        # and it must be measured the same way under both policies to be
        # comparable.
        raw_in_range = self.oracle.in_coverage(reporting_id, finish_t)
        # Effective verdict: what the gates in sumo_gates.py actually act on.
        # Under "always_accept" this is unconditionally True -- the whole
        # point of that arm is that nothing is ever discarded for being out
        # of range, so the pipeline can never waste a slot on mobility alone.
        uploaded = True if self.upload_policy == "always_accept" else raw_in_range
        self.context.state.setdefault(UPLOADED_KEY, {})[reporting_id] = uploaded
        self._record_upload(reporting_id, start_time, finish_t, uploaded, raw_in_range)

        await super().process_client_info(client_id, sid)

    # ------------------------------------------------------------------
    # Waste accounting
    # ------------------------------------------------------------------

    def _tier_of(self, client_id: int) -> str:
        try:
            return gpu_tier_sleep.tier_for(client_id).tier_id
        except KeyError:
            return "unknown"

    def _record_upload(
        self,
        client_id: int,
        start_time: float,
        finish_t: float,
        uploaded: bool,
        raw_in_range: bool,
    ) -> None:
        tier = self._tier_of(client_id)
        self.dispatched_per_tier[tier] += 1
        if not uploaded:
            self.wasted_per_tier[tier] += 1

        # policy= and raw_in_range= let a later pass reconstruct the Stage 0
        # counterfactual and separate "wasted under this policy" from "was
        # physically out of range" without re-running anything -- the two
        # coincide under coverage_gated (raw_in_range == uploaded's negation)
        # but not under always_accept, where uploaded is always 1 regardless
        # of raw_in_range.
        logging.info(
            "[SumoRsu] report round=%d client=%d tier=%s dispatch_t=%.2f "
            "finish_t=%.2f uploaded=%d raw_in_range=%d policy=%s",
            self.current_round,
            client_id,
            tier,
            start_time - self.initial_wall_time,
            finish_t,
            int(uploaded),
            int(raw_in_range),
            self.upload_policy,
        )

        self._reports_seen += 1
        if self._reports_seen % SUMMARY_EVERY == 0:
            self._log_waste_summary()

    def _log_waste_summary(self) -> None:
        total_dispatched = sum(self.dispatched_per_tier.values())
        total_wasted = sum(self.wasted_per_tier.values())
        parts = []
        for tier in sorted(self.dispatched_per_tier):
            dispatched = self.dispatched_per_tier[tier]
            wasted = self.wasted_per_tier[tier]
            parts.append(f"{tier}={wasted}/{dispatched} ({100 * wasted / dispatched:.1f}%)")
        logging.info(
            "[SumoRsu] waste round=%d cycles=%d total=%d/%d (%.1f%%) %s",
            self.current_round,
            self._dispatch_cycles,
            total_wasted,
            total_dispatched,
            100 * total_wasted / max(total_dispatched, 1),
            "  ".join(parts),
        )

    def server_will_close(self) -> None:
        # The final tally has to go here rather than in wrap_up(): wrap_up()
        # calls _close(), which ends with os._exit(0) (servers/base.py:1487), so
        # nothing after the super() call in wrap_up() ever runs.
        self._log_waste_summary()
        if self.stage0_gate is not None:
            self.stage0_gate.log_summary()
        super().server_will_close()


class SumoBaselineServer(SumoRsuMixin, fedasync_server.Server):
    """The `baseline` arm: random selection among vehicles in RSU range."""

    def __init__(
        self,
        oracle,
        upload_policy: str = DEFAULT_UPLOAD_POLICY,
        stage0: Stage0Config | None = None,
        position_oracle=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.install_sumo(oracle, upload_policy, stage0, position_oracle)


class SumoPolarisServer(SumoRsuMixin, fedasync_polaris_vi_server.Server):
    """The `gpu`, `cpu` and `observed` arms: V_i scoring among vehicles in range.

    Which of the three it is comes from `[server] staleness_predictor`, resolved
    by the parent exactly as it is outside SUMO.
    """

    def __init__(
        self,
        oracle,
        upload_policy: str = DEFAULT_UPLOAD_POLICY,
        stage0: Stage0Config | None = None,
        position_oracle=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.install_sumo(oracle, upload_policy, stage0, position_oracle)
