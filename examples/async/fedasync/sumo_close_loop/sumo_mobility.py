"""
Read-only view of the SUMO trace: is client i inside RSU range at time t?

The FL server has one clock, `server.wall_time`, which PLATO seeds with
`time.time()` and then advances by simulated seconds.  Elapsed simulated time is
therefore `wall_time - initial_wall_time`, and that is the number this module
treats as a SUMO timestamp (`t = 0` is the end of the trace's warm-up).

Everything here comes from `coverage_windows.csv` -- one row per continuous stay
inside range -- rather than the per-second `trace.csv`.  The window table is
~35x smaller, and a stay is exactly the unit both questions need:

    in_coverage(client, t)     -> can this vehicle talk to the RSU right now?
    next_entry_time(t)         -> when does ANY vehicle next come into range?

Window semantics
----------------
A window `[t_enter, t_exit]` is closed at both ends: SUMO sampled the vehicle at
1 s resolution and reported it inside range at both instants.

Wrap-around
-----------
The traces cover 36 000 s.  A 10 000-round run at T_round = 2.08 s needs about
20 800 s, so it fits -- but a longer run must not crash or silently freeze.
Because the trace is a closed loop in steady state, times past the end are
wrapped back into `[0, t_end]`, with one warning the first time it happens.
"""

from __future__ import annotations

import bisect
import csv
import logging
from pathlib import Path


class CoverageOracle:
    """Per-client coverage windows, with O(log n) lookup."""

    def __init__(self, windows: dict[int, list[tuple[float, float]]]) -> None:
        # Per client: entry times (sorted, for bisect) and matching exit times.
        self._enters: dict[int, list[float]] = {}
        self._exits: dict[int, list[float]] = {}
        for client_id, spans in windows.items():
            ordered = sorted(spans)
            self._enters[client_id] = [enter for enter, _ in ordered]
            self._exits[client_id] = [exit_ for _, exit_ in ordered]

        # All entry times across all clients, for next_entry_time().
        self._all_enters: list[float] = sorted(
            enter for spans in windows.values() for enter, _ in spans
        )
        self.t_start: float = self._all_enters[0] if self._all_enters else 0.0
        self.t_end: float = max(
            (exits[-1] for exits in self._exits.values() if exits), default=0.0
        )
        self._wrap_warned = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: str | Path) -> "CoverageOracle":
        """Load a coverage table written by remap_fleet.py."""
        windows: dict[int, list[tuple[float, float]]] = {}
        with Path(path).open(newline="") as f:
            for row in csv.DictReader(f):
                client_id = int(row["client_id"])
                windows.setdefault(client_id, []).append(
                    (float(row["t_enter_s"]), float(row["t_exit_s"]))
                )
        if not windows:
            raise ValueError(f"{path} contains no coverage windows.")

        oracle = cls(windows)
        logging.info(
            "[SumoMobility] Loaded %d clients, %d windows, t in [%.1f, %.1f] s.",
            len(windows),
            sum(len(v) for v in windows.values()),
            oracle.t_start,
            oracle.t_end,
        )
        return oracle

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def client_ids(self) -> list[int]:
        return sorted(self._enters)

    def normalize(self, t: float) -> float:
        """Fold a simulated time into the span the trace actually covers."""
        if t <= self.t_end:
            return t
        if not self._wrap_warned:
            self._wrap_warned = True
            logging.warning(
                "[SumoMobility] Simulated time %.1f s is past the end of the "
                "trace (%.1f s); wrapping. The trace is a closed loop in steady "
                "state, so this is sound, but the mobility pattern now repeats.",
                t,
                self.t_end,
            )
        return t % self.t_end

    def in_coverage(self, client_id: int, t: float) -> bool:
        """Whether *client_id* is inside RSU range at simulated time *t*."""
        enters = self._enters.get(client_id)
        if not enters:
            return False
        t = self.normalize(t)
        # Rightmost window whose entry time is <= t; it is the only candidate.
        index = bisect.bisect_right(enters, t) - 1
        return index >= 0 and t <= self._exits[client_id][index]

    def clients_in_coverage(self, t: float) -> list[int]:
        """Every client inside RSU range at *t*, in ascending ID order."""
        return [cid for cid in self.client_ids if self.in_coverage(cid, t)]

    def next_entry_time(
        self, t: float, clients: "set[int] | None" = None
    ) -> float | None:
        """The first time at or after *t* when a vehicle enters range.

        *clients* restricts the question to a subset. That matters: the server
        must wait for a vehicle it can actually *dispatch*, and PLATO withholds
        both the clients still training and those whose report has arrived but
        not yet been merged. Asking about the whole fleet would answer "someone
        is in range" while every one of them is unselectable.

        Returns None when no such entry exists at or after *t*, which past the
        end of the trace normalize() already prevents.
        """
        t = self.normalize(t)

        if clients is None:
            index = bisect.bisect_left(self._all_enters, t)
            return self._all_enters[index] if index < len(self._all_enters) else None

        earliest: float | None = None
        for client_id in clients:
            enters = self._enters.get(client_id)
            if not enters:
                continue
            index = bisect.bisect_left(enters, t)
            if index < len(enters) and (earliest is None or enters[index] < earliest):
                earliest = enters[index]
        return earliest

    def _current_window_index(self, client_id: int, t: float) -> int | None:
        """Index into this client's own window lists, or None if out of range."""
        enters = self._enters.get(client_id)
        if not enters:
            return None
        t = self.normalize(t)
        index = bisect.bisect_right(enters, t) - 1
        if index < 0 or t > self._exits[client_id][index]:
            return None
        return index

    def window_end(self, client_id: int, t: float) -> float | None:
        """When the current stay ends, or None if the client is out of range."""
        index = self._current_window_index(client_id, t)
        if index is None:
            return None
        return self._exits[client_id][index]

    def window_start(self, client_id: int, t: float) -> float | None:
        """When the current stay began, or None if the client is out of range.

        Stage 5's Stage 0 filter (prompts/STAGE5_stage0_filter.md) uses this as
        the one piece of "already observed" information its conservative R_i
        estimate is built from -- see stage0_filter.py.

        Caveat: for a client already inside coverage when a run's clock starts
        at t=0, `write_coverage()` (remap_fleet.py) clips the window's true,
        earlier entry to 0.0, since a real RSU starting up at t=0 could not
        have seen it either. This makes window_start()'s answer *later* than
        the client's true physical entry for that one first window only --
        a handful of clients, once, near round 1; measured and reported
        separately by Stage 5, not silently absorbed.
        """
        index = self._current_window_index(client_id, t)
        if index is None:
            return None
        return self._enters[client_id][index]
