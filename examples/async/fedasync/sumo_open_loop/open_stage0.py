"""
Stage 10: the Stage 0 feasibility gate for the OPEN population.

The closed-loop gate (`sumo_close_loop/stage0_filter.py`) keys position,
direction, tier and the spec `T_hat` on `client_id`.  In the open population
`client_id` is a slot, so reusing it would reproduce H3 inside the filter
(STAGE10_report.md section 0).  This gate reads:

    tier, T_hat     the OCCUPANT IDENTITY (occupant_identity.IdentityResolver):
                    the trace vehicle at p_recur = 0, the slot's fixed fleet
                    vehicle at p_recur = 1
    x_i, direction  the TRACE VEHICLE physically in the slot, in both regimes --
                    position is mobility, and at p_recur = 1 the trace is what
                    supplies the slot's coverage timing

Predicate and order are Stage 5b's, unchanged:

    eligible -> busy -> in-range-now (OpenSlotSelectionGate) -> THIS
             -> arm score -> lottery

    R_i  = |x_exit(direction) - x_i(t)| / v_max        (stage0_r_i = vmax_distance)
    pass = R_i >= T_hat_i                              (T_comm = 0)
    if nothing passes: dispatch from the unfiltered pool, log stage0_fallback=1

T_hat sources ([sumo_open] stage0_t_hat):

    none                  gate absent (default: every Stage 7-9 config)
    tier_spec             the arm's own spec file for the occupant's tier -- GPU
                          for gpu and baseline, tiers_cpu.csv for cpu (the same
                          predict_seconds_for_tier the scoring uses)
    observed_per_vehicle  the occupant identity's own mean realised training
                          time (report arrival - dispatch, on the server clock,
                          delivered reports only); for a never-seen identity,
                          the fleet running mean of every delivered duration so
                          far, starting at 8.323077 s.  No spec information.
"""

from __future__ import annotations

import csv
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CLOSED = _HERE.parent / "sumo_close_loop"
if str(_CLOSED) not in sys.path:
    sys.path.insert(0, str(_CLOSED))

from sumo_position import PositionOracle  # noqa: E402

from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext  # noqa: E402

# (10 xavier @ 15 + 7 orin @ 1.923077 + 3 thor @ 1.0) / 20 -- the closed-loop
# stage0_filter.FLEET_MEAN_T_HAT_S; the observed source's starting value.
FLEET_MEAN_T_HAT_S = 8.323077

T_HAT_SOURCES = frozenset({"none", "tier_spec", "observed_per_vehicle"})


@dataclass(frozen=True)
class OpenStage0Config:
    t_hat_source: str = "none"
    radius_m: float = 0.0
    v_max_mps: float = 0.0
    rsu_x_m: float = 0.0
    positions_csv: str = ""

    @property
    def enabled(self) -> bool:
        return self.t_hat_source != "none"


def resolve_open_stage0(open_cfg) -> OpenStage0Config:
    """Read the gate's keys from [sumo_open]; absent stage0_t_hat -> no gate.

    With the gate on, every geometry key is required and stage0_r_i must be
    vmax_distance: a missing key or another R_i would silently make a
    different run, so both RAISE (the closed loop's precedent).
    """
    raw = getattr(open_cfg, "stage0_t_hat", None)
    source = "none" if raw is None else str(raw).strip().lower()
    if source not in T_HAT_SOURCES:
        raise ValueError(f"[Stage0] stage0_t_hat={raw!r}; expected one of {sorted(T_HAT_SOURCES)}")
    if source == "none":
        return OpenStage0Config()
    r_i = str(getattr(open_cfg, "stage0_r_i", "")).strip().lower()
    if r_i != "vmax_distance":
        raise ValueError(f"[Stage0] open population supports stage0_r_i = 'vmax_distance' only, got {r_i!r}")
    missing = [k for k in ("coverage_radius_m", "v_max_mps", "rsu_x_m", "positions_csv")
               if getattr(open_cfg, k, None) is None]
    if missing:
        raise ValueError(f"[Stage0] stage0_t_hat is set but [sumo_open] lacks {missing}")
    return OpenStage0Config(
        source, float(open_cfg.coverage_radius_m), float(open_cfg.v_max_mps),
        float(open_cfg.rsu_x_m), str(open_cfg.positions_csv),
    )


def load_trace_positions(trace_csv: str | Path) -> PositionOracle:
    """1 Hz in-coverage samples of every trace vehicle, keyed on veh_id."""
    samples: dict[str, list[tuple[float, float, int]]] = {}
    with Path(trace_csv).open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["in_coverage"] != "1":
                continue
            samples.setdefault(row["veh_id"], []).append(
                (float(row["t"]), float(row["x_m"]), int(row["direction"])))
    if not samples:
        raise ValueError(f"[Stage0] {trace_csv} has no in-coverage rows.")
    logging.info("[Stage0] positions: %d vehicles, %d in-coverage samples (1 Hz) from %s",
                 len(samples), sum(len(v) for v in samples.values()), trace_csv)
    return PositionOracle(samples)


def remaining_range_s(config: OpenStage0Config, positions, veh_id: str, t: float) -> float | None:
    """vmax_distance R_i for the vehicle physically in the slot."""
    x = positions.x_at(veh_id, t)
    direction = positions.direction_at(veh_id, t)
    if x is None or direction is None:
        return None
    x_exit = config.rsu_x_m + config.radius_m if direction > 0 else config.rsu_x_m - config.radius_m
    return abs(x_exit - x) / config.v_max_mps


class OpenStage0Gate(ClientSelectionStrategy):
    """R_i >= T_hat_i on the occupant, between the slot gate and the arm."""

    def __init__(self, inner: ClientSelectionStrategy, server, config: OpenStage0Config,
                 positions) -> None:
        if not config.enabled:
            raise ValueError("[Stage0] OpenStage0Gate built with the gate disabled.")
        self.inner = inner
        self.server = server
        self.config = config
        self.positions = positions
        self.rounds_seen = self.candidates_seen = self.rejected_total = 0
        self.fallback_rounds = 0

    # -- lifecycle: pure delegation ------------------------------------------
    def setup(self, context: ServerContext) -> None:
        self.inner.setup(context)

    def teardown(self, context: ServerContext) -> None:
        self.inner.teardown(context)

    def on_clients_selected(self, selected, context: ServerContext) -> None:
        self.inner.on_clients_selected(selected, context)

    def on_reports_received(self, updates, context: ServerContext) -> None:
        self.inner.on_reports_received(updates, context)

    # -- T_hat ------------------------------------------------------------------
    def t_hat(self, identity) -> tuple[float, str]:
        """(T_hat, basis) for one occupant identity."""
        if self.config.t_hat_source == "tier_spec":
            return self.server.predict_seconds_for_tier(identity.tier), "spec"
        own = self.server.observed_duration_mean(identity.identity_id)
        if own is not None:
            return own, "own"
        return self.server.fleet_duration_mean(), "fleet"

    # -- the gated hook -----------------------------------------------------------
    def select_clients(self, clients_pool, clients_count, context: ServerContext):
        if not clients_pool:
            return self.inner.select_clients(clients_pool, clients_count, context)
        server = self.server
        t = server.sumo_time()
        rnd = getattr(server, "current_round", -1)

        feasible, rejected, used = [], [], {}
        for slot in clients_pool:
            identity = server.resolver.of_slot(slot)
            veh_id = server.timeline.vehicle_in_slot(slot)
            r_i = remaining_range_s(self.config, self.positions, veh_id, t) if veh_id else None
            t_hat, basis = self.t_hat(identity) if identity is not None else (math.nan, "none")
            ok = r_i is not None and r_i >= t_hat
            (feasible if ok else rejected).append(slot)
            used[slot] = (t_hat, r_i, basis, identity.tier if identity else "-")
            logging.info(
                "[Stage0] cand round=%d t=%.6f slot=%d veh_id=%s identity_id=%s tier=%s "
                "t_hat=%.6f t_hat_basis=%s r_i=%s pass=%d",
                rnd, t, slot, veh_id, identity.identity_id if identity else "-",
                used[slot][3], t_hat, basis, "nan" if r_i is None else f"{r_i:.6f}", int(ok))

        self.rounds_seen += 1
        self.candidates_seen += len(clients_pool)
        self.rejected_total += len(rejected)
        fallback = not feasible
        if fallback:
            self.fallback_rounds += 1
        pool = list(clients_pool) if fallback else feasible
        logging.info(
            "[Stage0] round=%d t_hat_source=%s r_i_mode=vmax_distance "
            "eligible_after_range_gate=%d eligible_after_stage0=%d "
            "stage0_rejected_ids=%s stage0_fallback=%d",
            rnd, self.config.t_hat_source, len(clients_pool), len(feasible),
            sorted(rejected), int(fallback))

        selected = self.inner.select_clients(pool, clients_count, context)
        for slot in selected:
            t_hat, r_i, basis, tier = used[slot]
            logging.info(
                "[Stage0] round=%d selected_id=%d selected_tier=%s t_hat_used=%.6f "
                "R_i_selected=%s t_hat_basis=%s",
                rnd, slot, tier, t_hat, "nan" if r_i is None else f"{r_i:.6f}", basis)
        return selected

    def log_summary(self) -> None:
        logging.info(
            "[Stage0] summary t_hat_source=%s rounds=%d candidates=%d rejected=%d (%.1f%%) "
            "fallback_rounds=%d (%.1f%%)",
            self.config.t_hat_source, self.rounds_seen, self.candidates_seen,
            self.rejected_total, 100 * self.rejected_total / max(1, self.candidates_seen),
            self.fallback_rounds, 100 * self.fallback_rounds / max(1, self.rounds_seen))
