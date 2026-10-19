"""
Open-population mobility: the coverage timeline, keyed on string `veh_id`.

`sumo_close_loop/sumo_mobility.py` keys everything on an integer `client_id`
because its 20 vehicles own their ids for the whole run.  Here 10,523 vehicles
share 20 slots, so the timeline has to be keyed on the vehicle and the slot
mapping has to be derived from it, in simulated-time order.

The one invariant this module exists to protect
------------------------------------------------
A slot is taken when a vehicle ENTERS coverage and released when it LEAVES,
whether or not it was ever selected and whether or not it is training.  So the
whole assign/release stream is a function of the trace alone.  `advance_to()`
is therefore driven only by the clock: it never looks at who was selected, and
nothing in this file takes a selector as an argument.

`digest_at()` is what proves that held after the fact — NOT
`occupancy_digest()` compared between runs.  Each arm consumes a different
amount of the trace in 4,000 rounds (its dispatch mix sets its clock rate), so
end-of-run digests are prefixes of differing length and will differ even when
nothing is wrong.  Compare at a common simulated time instead.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path

from stage7_population import RecycleEvent, SlotAllocator


class OpenCoverageTimeline:
    """Coverage enters/exits in time order, plus the slot allocator they drive."""

    def __init__(
        self,
        events: list[tuple[float, int, str, str]],
        slots: int,
        p_recur: float,
        base_seed: int = 4126,
    ) -> None:
        # (t, is_arrival, veh_id, tier), arrivals after departures at equal t so
        # a slot freed at t is available to a vehicle arriving at t.
        self._events = sorted(events, key=lambda e: (e[0], e[1]))
        self._next = 0
        self.allocator = SlotAllocator(slots=slots, p_recur=p_recur, base_seed=base_seed)
        self.tier_of_vehicle: dict[str, str] = {v: t for _, _, v, t in events}
        self._digest = hashlib.blake2b(digest_size=16)
        self.turned_away: int = 0
        self.event_t: float = 0.0

        # Each vehicle's own coverage window, kept so a report can be judged
        # against the vehicle that was DISPATCHED even after its slot has been
        # recycled to somebody else.
        self.window_of: dict[str, tuple[float, float]] = {}
        for t, is_arrival, veh_id, _tier in self._events:
            enter, exit_ = self.window_of.get(veh_id, (None, None))
            if is_arrival:
                enter = t
            else:
                exit_ = t
            self.window_of[veh_id] = (enter, exit_)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        slots: int,
        p_recur: float,
        base_seed: int = 4126,
        start_time: float = 0.0,
    ) -> "OpenCoverageTimeline":
        """Load `coverage_windows.csv`, applying remap_fleet.py's t >= 0 clip."""
        events: list[tuple[float, int, str, str]] = []
        dropped = clipped = 0
        with Path(path).open(newline="") as fh:
            for row in csv.DictReader(fh):
                enter, exit_ = float(row["t_enter_s"]), float(row["t_exit_s"])
                if exit_ < start_time:
                    dropped += 1
                    continue
                if enter < start_time:
                    enter = start_time
                    clipped += 1
                events.append((enter, 1, row["veh_id"], row["tier"]))
                events.append((exit_, 0, row["veh_id"], row["tier"]))

        timeline = cls(events, slots, p_recur, base_seed)
        logging.info(
            "[Stage7] Loaded %d coverage windows (%d dropped before t=%.0f, "
            "%d clipped), %d vehicles, M=%d slots, p_recur=%g.",
            len(events) // 2, dropped, start_time, clipped,
            len(timeline.tier_of_vehicle), slots, p_recur,
        )
        return timeline

    # ------------------------------------------------------------------

    def advance_to(self, now: float) -> list[RecycleEvent]:
        """Apply every coverage event at or before `now`.  Clock-driven only.

        Returns the recycle events produced, for the caller to log and to
        apply the p_recur state policy to.  Vehicles that find no free slot
        are counted as turned away and simply never become clients — never
        evicting anyone, since evicting a training vehicle would truncate the
        population a second, less visible way.
        """
        recycles: list[RecycleEvent] = []
        while self._next < len(self._events) and self._events[self._next][0] <= now:
            t, is_arrival, veh_id, tier = self._events[self._next]
            self._next += 1
            self.event_t = t  # read by Stage 11's allocation observers
            if is_arrival:
                slot, event = self.allocator.assign(veh_id, tier)
                if slot is None:
                    self.turned_away += 1
                    logging.info(
                        "[Stage7] slot_exhausted t=%.2f veh_id=%s tier=%s", t, veh_id, tier
                    )
                elif event is not None:
                    recycles.append(event)
                self._digest.update(f"{t:.3f}|A|{veh_id}|{slot}\n".encode())
            else:
                slot = self.allocator.release(veh_id)
                self._digest.update(f"{t:.3f}|R|{veh_id}|{slot}\n".encode())
        return recycles

    # ------------------------------------------------------------------

    def occupied_slots(self) -> list[int]:
        """The slots currently holding an in-coverage vehicle, ascending."""
        return sorted(self.allocator.occupant)

    def vehicle_in_slot(self, slot_id: int) -> str | None:
        return self.allocator.occupant.get(slot_id)

    def tier_in_slot(self, slot_id: int) -> str | None:
        return self.allocator.tier_of_slot(slot_id)

    def peek_next_time(self) -> float | None:
        """When the next coverage event happens, or None if the trace is spent."""
        if self._next >= len(self._events):
            return None
        return self._events[self._next][0]

    def vehicle_in_coverage_at(self, veh_id: str, t: float) -> bool:
        """Was THIS vehicle still inside coverage at simulated time `t`?

        Asked of the vehicle rather than of its slot, because by the time a
        report comes back the slot may belong to someone else entirely.
        """
        window = self.window_of.get(veh_id)
        if window is None:
            return False
        enter, exit_ = window
        if enter is None or exit_ is None:
            return False
        return enter <= t <= exit_

    def occupancy_digest(self) -> str:
        """blake2b over the assign/release stream consumed SO FAR.

        Trace-determined, but a *prefix*: how far a run gets through the trace
        depends on how much simulated time 4,000 rounds take, which differs
        per arm.  So two runs' end-of-run digests are NOT directly comparable
        — measured on Stage 7 wave 1, where `baseline_open` stopped at
        t = 7,262 s, `gpu_open` at 7,463 s and `observed_open` at 6,807 s, and
        all three digests were exact prefixes of the same stream.

        To check the cross-run invariant, compare `digest_at()` at a common
        time instead.  See `digest_at`.
        """
        return self._digest.hexdigest()

    @property
    def events_consumed(self) -> int:
        return self._next


def digest_at(
    coverage_csv: str | Path,
    slots: int,
    until: float,
    p_recur: float = 0.0,
    base_seed: int = 4126,
    start_time: float = 0.0,
) -> tuple[str, int, int]:
    """The occupancy digest a run must show once it has reached `until`.

    The cross-run invariant the stage's stop condition is really asserting is
    that the assign/release stream is a function of the TRACE — not of which
    arm is running.  End-of-run digests cannot test that directly, because
    each arm consumes a different amount of the trace in 4,000 rounds.  Replay
    to a common `until` instead, and every run must agree.

    `p_recur` is accepted only to make the point that it cannot matter: slots
    are taken on coverage entry and released on coverage exit, so the identity
    policy has no way to influence the stream.

    Returns (digest, recycles, turned_away).
    """
    timeline = OpenCoverageTimeline.from_csv(
        coverage_csv, slots=slots, p_recur=p_recur,
        base_seed=base_seed, start_time=start_time,
    )
    timeline.advance_to(until)
    return (
        timeline.occupancy_digest(),
        len(timeline.allocator.recycles),
        timeline.turned_away,
    )
