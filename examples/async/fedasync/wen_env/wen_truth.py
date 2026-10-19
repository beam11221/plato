"""
Ground truth: what actually happens to one dispatch. Simulator side only.

For a dispatch of client i at clean time t_dispatch:

    T_train_true = T_train_formula(tier_i, |S_i|) * exp(eps_i)
    x_finish     = x(veh_i, t_dispatch + T_train_true)           from the trace
    d_finish     = sqrt((x_finish - 1000)^2 + 10^2)
    chi2         ~ Exp(1), RNG keyed by (seed, client_id, round, k)
    T_up_true    = D / (B log2(1 + d^-3 chi2 p / N))
    t_finish     = t_dispatch + T_train_true + T_up_true
    accepted     = S1: always
                   S2: the vehicle is inside [400, 1600] for the whole upload
                       interval [t_dispatch + T_train_true, t_finish]

`k` counts earlier dispatches of the same client within the same round number.
It is 0 except when an S2 cycle whose only update was lost holds the round
counter and the same client is dispatched again under that round number; the
prompt's key (seed, client_id, round) alone would then replay the identical
fading draw.

Nothing here is visible to a selector: the server holds a GroundTruth, and
estimators receive only `wen_reported.ReportedVehicle` objects
(tests/test_wen_separation.py).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

import wen_constants as K
import wen_formulas as F
from wen_fleet import Fleet
from wen_trace import TraceEndError, TraceIndex


@dataclass(frozen=True)
class DispatchOutcome:
    client_id: int
    veh_id: str
    tier: str
    round: int
    t_dispatch: float
    x_dispatch: float
    T_train_true: float
    x_finish: float
    d_finish: float
    chi2: float
    T_up_true: float
    t_finish_clean: float
    accepted: bool
    E_train: float
    E_up: float

    @property
    def duration(self) -> float:
        """The sleep PLATO is handed for this dispatch."""
        return self.T_train_true + self.T_up_true

    def as_dict(self) -> dict:
        return asdict(self)


class GroundTruth:
    def __init__(self, fleet: Fleet, trace: TraceIndex, flops_fwd: float,
                 model_bits: float, seed: int, scenario: str) -> None:
        if scenario not in K.SCENARIOS:
            raise ValueError(f"scenario {scenario!r} not in {K.SCENARIOS}")
        self.fleet = fleet
        self.trace = trace
        self.flops_fwd = flops_fwd
        self.model_bits = model_bits
        self.seed = seed
        self.scenario = scenario
        self._draws_per_round: dict[tuple[int, int], int] = defaultdict(int)

    def t_train_true(self, client_id: int) -> float:
        spec = self.fleet.spec(client_id)
        nominal = F.train_seconds(self.flops_fwd, K.partition_size(client_id),
                                  K.LOCAL_EPOCHS, spec.tflops_fp16)
        return nominal * self.fleet.jitter(client_id)

    def chi2(self, client_id: int, round_no: int) -> float:
        k = self._draws_per_round[(client_id, round_no)]
        self._draws_per_round[(client_id, round_no)] += 1
        rng = np.random.default_rng([self.seed, client_id, round_no, k])
        return float(rng.exponential(1.0))

    def dispatch(self, client_id: int, t_dispatch: float, round_no: int) -> DispatchOutcome:
        vid = self.fleet.veh_id(client_id)
        tier = self.fleet.tier(client_id)
        x_dispatch = self.trace.x_at(vid, t_dispatch)

        t_train = self.t_train_true(client_id)
        t_train_end = t_dispatch + t_train
        x_finish = self.trace.x_at(vid, t_train_end)
        d_finish = F.distance_to_rsu(x_finish)
        chi2 = self.chi2(client_id, round_no)
        t_up = F.upload_seconds(d_finish, chi2, self.model_bits)
        # One addition onto t_dispatch, in exactly the form PLATO uses
        # (start_time + training_time, with training_time = this duration), so
        # the online clock and PLATO's clock are bit-identical.
        t_finish = t_dispatch + (t_train + t_up)
        if t_finish > self.trace.t_end:
            raise TraceEndError(
                f"client {client_id} dispatched at t = {t_dispatch:.3f} s would "
                f"finish at {t_finish:.3f} s, past the trace end "
                f"({self.trace.t_end:.0f} s)."
            )

        if self.scenario == "S1":
            accepted = True
        else:
            lo, hi = K.S2_COVERAGE_M
            accepted = self.trace.stays_inside(vid, t_train_end, t_finish, lo, hi)

        e_train, e_up = F.energy_joules(tier, t_train, t_up)
        return DispatchOutcome(
            client_id=client_id, veh_id=vid, tier=tier, round=round_no,
            t_dispatch=t_dispatch, x_dispatch=x_dispatch, T_train_true=t_train,
            x_finish=x_finish, d_finish=d_finish, chi2=chi2, T_up_true=t_up,
            t_finish_clean=t_finish, accepted=accepted, E_train=e_train, E_up=e_up,
        )
