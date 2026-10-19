"""
Stage 0 feasibility gate (prompts/STAGE5_stage0_filter.md, Step 1).

Inserted between the in-range-now gate and the arm's own scoring/lottery:

    eligible -> [busy filter] -> [in-range-now gate] -> [Stage 0] -> score -> lottery

The in-range-now gate (sumo_gates.RsuSelectionGate) and this one are not the
same predicate -- one is instantaneous, one is predictive -- and are kept as
two separate wrappers on purpose; see sumo_server.install_sumo().

R_i(t): a CONSERVATIVE estimate of how much longer client i will stay in RSU
range. Three variants (`[sumo] stage0_r_i`), all guaranteed lower bounds
since SUMO enforces speed <= v_max_mps everywhere (so a distance can never be
covered faster than at v_max, and time-in-range can never be shorter than
41.142857 s = 2*radius_m/v_max_mps):

    vmax_elapsed (default)  window_start(client_id, t) + 2*radius_m/v_max_mps - t
        Stage 5's original. Built only from the time the client entered its
        current window -- never the vehicle's own true exit time (that would
        be Task 4's oracle counterfactual, not a deployable filter) and never
        live position (not loaded at all in Stage 5 -- see sumo_mobility.py's
        module docstring). Its error is "time already spent in range assumed
        travelled at v_max", which GROWS the longer a client has been in
        range -- Stage 5b (prompts/STAGE5B_fix_Ri.md) found this had been
        acting as a hidden speed filter, not just a feasibility one.

    vmax_distance  |x_exit_boundary(direction) - x_i(t)| / v_max_mps
        Stage 5b's fix: the same worst-case assumption (still v_max, still no
        acceleration model), applied to the vehicle's ACTUAL CURRENT position
        instead of inferred elapsed time, so the error does not accumulate.
        Deployable -- position is exactly what a V2X basic safety message
        carries. Needs sumo_position.PositionOracle (`[sumo] positions_csv`),
        loaded from the per-second trace sumo_mobility.py's own docstring
        already flagged as available but unused.

    oracle_exit  window_end(client_id, t) - t
        The non-deployable ceiling: exact true remaining dwell. Task 4's own
        Stage 0 counterfactual formula, offered here as `stage0_r_i`'s upper
        bound for comparison, not as a candidate for deployment.

`vmax_elapsed` is the default, so every config written before Stage 5b
existed -- including every Stage 5 and Task 4 run -- stays byte-identical.

T_hat sources (`[sumo] stage0_t_hat`), independent of the selector arm:

    none        the gate is disabled -- Task 4's un-filtered controls
    fleet_mean  FLEET_MEAN_T_HAT_S for every client, no hardware specs at all
    tier_spec   gpu_tier_sleep.gpu_tops_predict_seconds(...), i.e. whatever
                [gpu_tier] tiers_csv this run's OWN config already points at
                -- a cpu-arm run therefore gets cpu's own wrong belief through
                the exact same call, with no separate code path
    oracle      the same function, but with [gpu_tier] oracle_sleep_csv set,
                so it returns the client's true jittered sleep time instead

Fallback: if the predicate would empty the candidate pool, dispatch proceeds
on the unfiltered pool instead and a stage0_fallback event is logged -- Stage
0 must never idle a round it could have filled (see the module prompt's
"Fallback rule").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import gpu_tier_sleep

from plato.config import Config
from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext

# Stage 5 Step 0 pre-check: the composition-weighted mean of the fleet's
# per-tier canonical training time -- (10 xavier @ 15s + 7 orin @ 1.923077s +
# 3 thor @ 1.0s) / 20 clients -- what a server could compute from the fleet's
# tier mix alone, with no hardware specs. Matches
# gpu_tiers_dataset/fleets/n20_x10-o7-t3_xa15_p2500_j0p149_s4126/manifest.json
# T_tier_s to 6 decimal places.
FLEET_MEAN_T_HAT_S = 8.323077

T_HAT_SOURCES = frozenset({"none", "fleet_mean", "tier_spec", "oracle"})
DEFAULT_T_HAT_SOURCE = "none"

# Stage 5b (prompts/STAGE5B_fix_Ri.md): which formula R_i uses. Independent of
# T_HAT_SOURCES -- this changes what "remaining range" means, not what
# "predicted training time" means.
RI_MODES = frozenset({"vmax_elapsed", "vmax_distance", "oracle_exit"})
DEFAULT_RI_MODE = "vmax_elapsed"


def resolve_t_hat_source(raw) -> str:
    if raw is None:
        return DEFAULT_T_HAT_SOURCE
    name = str(raw).strip().lower()
    if name not in T_HAT_SOURCES:
        raise ValueError(
            f"[Stage0] stage0_t_hat = {raw!r} is not recognised; expected one "
            f"of {sorted(T_HAT_SOURCES)}. Omit the key entirely to get "
            f"{DEFAULT_T_HAT_SOURCE!r} (the gate disabled)."
        )
    return name


def resolve_r_i_mode(raw) -> str:
    if raw is None:
        return DEFAULT_RI_MODE
    name = str(raw).strip().lower()
    if name not in RI_MODES:
        raise ValueError(
            f"[Stage0] stage0_r_i = {raw!r} is not recognised; expected one "
            f"of {sorted(RI_MODES)}. Omit the key entirely to get "
            f"{DEFAULT_RI_MODE!r} (Stage 5's original formula)."
        )
    return name


@dataclass(frozen=True)
class Stage0Config:
    t_hat_source: str
    radius_m: float = 0.0
    v_max_mps: float = 0.0
    r_i_mode: str = DEFAULT_RI_MODE
    rsu_x_m: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.t_hat_source != "none"


def resolve_stage0_config(sumo_cfg) -> Stage0Config:
    """Read stage0_t_hat / coverage_radius_m / v_max_mps / stage0_r_i /
    rsu_x_m from a [sumo] section.

    coverage_radius_m and v_max_mps are required together whenever the
    filter is enabled -- R_i cannot be computed from one without the other --
    and are read from the same [sumo] section coverage_csv already lives in,
    since both describe the same trace. rsu_x_m is required in addition only
    for stage0_r_i = "vmax_distance", which needs the RSU's own position to
    know which boundary a client is heading for. Missing a required key with
    the filter on RAISES rather than silently disabling it or silently
    falling back to a different R_i, following resolve_upload_policy's
    precedent in sumo_server.py: a silent fallback here would make a missing
    key an undetectable duplicate of a different, unintended run.
    """
    t_hat_source = resolve_t_hat_source(getattr(sumo_cfg, "stage0_t_hat", None))
    if t_hat_source == "none":
        return Stage0Config(t_hat_source)

    radius = getattr(sumo_cfg, "coverage_radius_m", None)
    v_max = getattr(sumo_cfg, "v_max_mps", None)
    if radius is None or v_max is None:
        raise ValueError(
            "[Stage0] [sumo] stage0_t_hat is set but coverage_radius_m and/or "
            "v_max_mps is missing -- R_i needs both to compute a crossing "
            "time. Omit stage0_t_hat entirely to run without the filter."
        )

    r_i_mode = resolve_r_i_mode(getattr(sumo_cfg, "stage0_r_i", None))
    rsu_x = getattr(sumo_cfg, "rsu_x_m", None)
    if r_i_mode == "vmax_distance" and rsu_x is None:
        raise ValueError(
            "[Stage0] stage0_r_i = 'vmax_distance' needs [sumo] rsu_x_m (the "
            "RSU's position on the road) to know which boundary a client is "
            "heading for."
        )
    return Stage0Config(
        t_hat_source, float(radius), float(v_max), r_i_mode,
        float(rsu_x) if rsu_x is not None else 0.0,
    )


def _workload() -> tuple[int, int]:
    """(partition_size, epochs) for this run, read once per prediction.

    Every client in this project's fleets trains on the same partition_size
    (gpu_tiers_dataset/build_fleet.py), so a single pair -- rather than a
    per-client dict threaded through context.state -- is enough, and it keeps
    this gate working for arms (baseline) whose own selection strategy never
    populates context.state["partition_sizes"] at all.
    """
    return (int(Config().data.partition_size), int(Config().trainer.epochs))


class Stage0FeasibilityGate(ClientSelectionStrategy):
    """R_i >= T_hat_i, wrapped around whatever selection strategy the arm uses.

    Constructed as the innermost layer, with the arm's own strategy inside it
    and sumo_gates.RsuSelectionGate outside it -- see install_sumo() in
    sumo_server.py for the exact wrapping order this depends on.
    """

    def __init__(
        self,
        inner: ClientSelectionStrategy,
        oracle,
        config: Stage0Config,
        position_oracle=None,
    ) -> None:
        self.inner = inner
        self.oracle = oracle
        self.config = config
        self.position_oracle = position_oracle
        if config.enabled and config.r_i_mode == "vmax_distance" and position_oracle is None:
            raise ValueError(
                "[Stage0] stage0_r_i = 'vmax_distance' needs a PositionOracle "
                "-- pass one in from sumo_fedasync.py's [sumo] positions_csv."
            )
        self._crossing_s = (
            2.0 * config.radius_m / config.v_max_mps if config.enabled else 0.0
        )
        # Cheap counters for the end-of-run summary (server_will_close());
        # the per-round/per-dispatch detail lives in the log lines below.
        self.candidates_seen = 0
        self.rejected_total = 0
        self.fallback_rounds = 0
        self.rounds_seen = 0

    # -- lifecycle: pure delegation --------------------------------------

    def setup(self, context: ServerContext) -> None:
        self.inner.setup(context)

    def teardown(self, context: ServerContext) -> None:
        self.inner.teardown(context)

    def on_clients_selected(self, selected_clients, context: ServerContext) -> None:
        self.inner.on_clients_selected(selected_clients, context)

    def on_reports_received(self, updates, context: ServerContext) -> None:
        self.inner.on_reports_received(updates, context)

    # -- R_i / T_hat ------------------------------------------------------

    def remaining_range_s(self, client_id: int, t: float) -> float | None:
        """Conservative lower bound on remaining dwell time in seconds, or
        None if the client is not currently in coverage (should not happen --
        this gate only ever sees clients RsuSelectionGate already passed)."""
        mode = self.config.r_i_mode
        if mode == "oracle_exit":
            end = self.oracle.window_end(client_id, t)
            return None if end is None else end - t

        if mode == "vmax_distance":
            x = self.position_oracle.x_at(client_id, t)
            direction = self.position_oracle.direction_at(client_id, t)
            if x is None or direction is None:
                return None
            x_exit = (
                self.config.rsu_x_m + self.config.radius_m if direction > 0
                else self.config.rsu_x_m - self.config.radius_m
            )
            return abs(x_exit - x) / self.config.v_max_mps

        # "vmax_elapsed" -- Stage 5's original.
        enter = self.oracle.window_start(client_id, t)
        if enter is None:
            return None
        return enter + self._crossing_s - t

    def t_hat(self, client_id: int, dataset_size: int, epochs: int) -> float:
        if self.config.t_hat_source == "fleet_mean":
            return FLEET_MEAN_T_HAT_S
        # tier_spec and oracle are the SAME call -- which belief it returns
        # depends only on whether this run's config set [gpu_tier]
        # oracle_sleep_csv. See gpu_tier_sleep.gpu_tops_predict_seconds /
        # oracle_sleeps().
        return gpu_tier_sleep.gpu_tops_predict_seconds(client_id, dataset_size, epochs)

    @staticmethod
    def _tier_of(client_id: int) -> str:
        try:
            return gpu_tier_sleep.tier_for(client_id).tier_id
        except KeyError:
            return "unknown"

    # -- the gated hook -----------------------------------------------------

    def select_clients(
        self, clients_pool: list[int], clients_count: int, context: ServerContext
    ) -> list[int]:
        if not self.config.enabled or not clients_pool:
            return self.inner.select_clients(clients_pool, clients_count, context)

        server = context.server
        t = server.sumo_time()
        round_no = getattr(server, "current_round", -1)
        dataset_size, epochs = _workload()

        r_of: dict[int, float] = {}
        t_hat_of: dict[int, float] = {}
        feasible: list[int] = []
        rejected: list[int] = []
        for cid in clients_pool:
            r_i = self.remaining_range_s(cid, t)
            t_hat_i = self.t_hat(cid, dataset_size, epochs)
            t_hat_of[cid] = t_hat_i
            if r_i is not None:
                r_of[cid] = r_i
            if r_i is not None and r_i >= t_hat_i:
                feasible.append(cid)
            else:
                rejected.append(cid)

        self.rounds_seen += 1
        self.candidates_seen += len(clients_pool)
        self.rejected_total += len(rejected)

        fallback = not feasible
        pool = clients_pool if fallback else feasible
        if fallback:
            self.fallback_rounds += 1

        logging.info(
            "[Stage0] round=%d t_hat_source=%s r_i_mode=%s "
            "eligible_after_range_gate=%d eligible_after_stage0=%d "
            "stage0_rejected_ids=%s stage0_fallback=%d",
            round_no, self.config.t_hat_source, self.config.r_i_mode,
            len(clients_pool), len(feasible), sorted(rejected), int(fallback),
        )

        selected = self.inner.select_clients(pool, clients_count, context)

        for cid in selected:
            logging.info(
                "[Stage0] round=%d selected_id=%d selected_tier=%s "
                "t_hat_used=%.6f R_i_selected=%s",
                round_no, cid, self._tier_of(cid), t_hat_of.get(cid, float("nan")),
                f"{r_of[cid]:.6f}" if cid in r_of else "nan",
            )

        return selected

    def log_summary(self) -> None:
        rate = self.fallback_rounds / self.rounds_seen if self.rounds_seen else 0.0
        reject_rate = (
            self.rejected_total / self.candidates_seen if self.candidates_seen else 0.0
        )
        logging.info(
            "[Stage0] summary t_hat_source=%s rounds=%d candidates=%d "
            "rejected=%d (%.1f%%) fallback_rounds=%d (%.1f%%)",
            self.config.t_hat_source, self.rounds_seen, self.candidates_seen,
            self.rejected_total, 100 * reject_rate, self.fallback_rounds,
            100 * rate,
        )
