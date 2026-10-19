"""
What a vehicle reports to the RSU at decision time. Server-visible.

The ClientProfile split (ground truth / reported / estimated) is enforced by
construction: a selector or estimator receives `ReportedVehicle` objects and
nothing else. The record has no field for jitter, fading, future position or
the true training time, so there is nothing to read by accident -- and
tests/test_wen_separation.py checks that the estimator and selector modules do
not import the ground-truth modules at all.

Reported attributes (Prompt B, Step 2): current position x, constant speed V_k
(fleet_speeds.csv), the tier's datasheet spec, and the client's sample count.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReportedVehicle:
    client_id: int
    t: float                 # decision time the report describes
    x: float                 # position on the road, m
    speed: float             # V_k, m/s
    tier: str
    tflops_fp16: float
    cpu_ghz: float
    n_samples: int           # |S_i|


class ReportedView:
    """Builds ReportedVehicle records. Held by the server, not by selectors.

    It is constructed from the trace (position now) and the fleet's PUBLIC
    attributes (tier spec); `position` and `spec` are passed in as callables so
    this module never imports the ground-truth side.
    """

    def __init__(self, position, speed, spec, n_samples) -> None:
        self._position = position      # (client_id, t) -> x
        self._speed = speed            # client_id -> V_k
        self._spec = spec              # client_id -> (tier, tflops, cpu_ghz)
        self._n = n_samples            # client_id -> |S_i|

    def report(self, client_id: int, t: float) -> ReportedVehicle:
        tier, tflops, ghz = self._spec(client_id)
        return ReportedVehicle(
            client_id=client_id, t=t, x=self._position(client_id, t),
            speed=self._speed(client_id), tier=tier, tflops_fp16=tflops,
            cpu_ghz=ghz, n_samples=self._n(client_id),
        )
