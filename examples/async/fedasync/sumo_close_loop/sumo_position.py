"""
Read-only view of a client's actual road position, for Stage 5b's
`vmax_distance` R_i (prompts/STAGE5B_fix_Ri.md).

Counterpart to `sumo_mobility.CoverageOracle`, which only ever exposed
entry/exit *times*. This loads `positions_<cell>.csv` (written by
remap_positions.py from the raw per-second trace, filtered to the rows a live
Stage 0 could ever query -- a client already inside coverage) and answers:

    x_at(client_id, t)          -> where is the client right now?
    direction_at(client_id, t)  -> which way is it travelling (+1 / -1)?

Both are linearly interpolated between the two surrounding 1 Hz samples,
never extrapolated past a contiguous run: a gap bigger than the trace's own
sampling interval means the query time falls outside a window this client was
continuously observed in (which should not happen -- callers only ask about
clients the in-range-now gate already confirmed are in coverage right now --
but it is checked rather than assumed, and returns None instead of silently
interpolating across a gap that isn't really contiguous motion).
"""

from __future__ import annotations

import bisect
import csv
import logging
from pathlib import Path

# The trace samples at this resolution (config.yaml: trace_dt_s = 1.0). A gap
# larger than this between two consecutive samples for one client means they
# belong to two different coverage windows, not one continuous span.
SAMPLE_INTERVAL_S = 1.0
_GAP_TOLERANCE_S = SAMPLE_INTERVAL_S * 1.01


class PositionOracle:
    """Per-client (t, x_m, direction) samples, with O(log n) lookup."""

    def __init__(self, samples: dict[int, list[tuple[float, float, int]]]) -> None:
        # Per client: sorted (t, x_m, direction) triples, split into arrays
        # for bisect.
        self._t: dict[int, list[float]] = {}
        self._x: dict[int, list[float]] = {}
        self._dir: dict[int, list[int]] = {}
        for client_id, rows in samples.items():
            ordered = sorted(rows)
            self._t[client_id] = [t for t, _, _ in ordered]
            self._x[client_id] = [x for _, x, _ in ordered]
            self._dir[client_id] = [d for _, _, d in ordered]

    @classmethod
    def from_csv(cls, path: str | Path) -> "PositionOracle":
        samples: dict[int, list[tuple[float, float, int]]] = {}
        with Path(path).open(newline="") as f:
            for row in csv.DictReader(f):
                client_id = int(row["client_id"])
                samples.setdefault(client_id, []).append(
                    (float(row["t"]), float(row["x_m"]), int(row["direction"]))
                )
        if not samples:
            raise ValueError(f"{path} contains no position rows.")

        oracle = cls(samples)
        logging.info(
            "[SumoPosition] Loaded %d clients, %d position samples.",
            len(samples), sum(len(v) for v in samples.values()),
        )
        return oracle

    # -- shared bracket lookup ------------------------------------------

    def _bracket(self, client_id: int, t: float) -> tuple[int, int, float] | None:
        """(lo_index, hi_index, fraction) bracketing *t*, or None.

        `fraction` is 0.0 exactly on the low sample, 1.0 exactly on the high
        one. Returns None if the client is unknown, *t* is outside its
        recorded range, or the two nearest samples are more than one
        sampling interval apart (not a contiguous window).
        """
        times = self._t.get(client_id)
        if not times:
            return None
        i = bisect.bisect_left(times, t)
        if i < len(times) and times[i] == t:
            return i, i, 0.0
        if i == 0 or i == len(times):
            return None  # before the first sample or after the last
        lo, hi = i - 1, i
        span = times[hi] - times[lo]
        if span > _GAP_TOLERANCE_S:
            return None
        return lo, hi, (t - times[lo]) / span

    def x_at(self, client_id: int, t: float) -> float | None:
        """Linearly interpolated road position (m), or None -- see class doc."""
        bracket = self._bracket(client_id, t)
        if bracket is None:
            return None
        lo, hi, frac = bracket
        xs = self._x[client_id]
        return xs[lo] + (xs[hi] - xs[lo]) * frac

    def direction_at(self, client_id: int, t: float) -> int | None:
        """+1 / -1 direction of travel, or None -- see class doc.

        Direction never flips inside one contiguous coverage window (checked
        against all 10,180 windows of the radius-400 trace before this was
        written), so either bracketing sample's direction is the window's.
        """
        bracket = self._bracket(client_id, t)
        if bracket is None:
            return None
        lo, _hi, _frac = bracket
        return self._dir[client_id][lo]
