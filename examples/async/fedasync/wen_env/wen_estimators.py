"""
Estimators: T-hat = T-hat_train + T-hat_up, and E-hat, from reported attributes only.

One module shared by every arm (Prompt B, Step 5). Each estimator answers for
one `ReportedVehicle` at the decision time t:

    T_hat_train   arm-specific (below)
    x_hat         wrap(x + V_k * T_hat_train)       where training would end
    T_hat_up      Wen Eq. 5-6 at d(x_hat) with |chi|^2 = 1
    T_hat         T_hat_train + T_hat_up            what tau-hat is derived from
    E_hat         P_tier * T_hat_train + p * T_hat_up

Arms:
    gpu            tier formula, no jitter (the proposal)
    cpu            Wen Eq. 4: b |S| rho / f  (~23.3 / 23.3 / 19.7 s)
    gpu_oracle     the client's TRUE training time (jitter included), handed in
                   as an explicit callable by the server -- the only arm that
                   sees ground truth, and only that one number; upload is still
                   estimated with |chi|^2 = 1
    location_only  T_hat_train = constant (fleet mean of the gpu estimates), so
                   only the position term differs between vehicles

`observed` is not here: it estimates staleness in rounds from history and does
not use a T-hat (see wen_server.py for what it stores).

This module imports only wen_constants, wen_formulas and wen_reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import wen_constants as K
import wen_formulas as F
from wen_reported import ReportedVehicle

ESTIMATOR_ARMS = ("gpu", "cpu", "gpu_oracle", "location_only", "cpu_cal")


@dataclass(frozen=True)
class Estimate:
    T_train: float
    T_up: float
    x_finish: float
    E: float                # the arm family's own energy estimate (D2)

    @property
    def total(self) -> float:
        return self.T_train + self.T_up



class Estimator:
    name = "base"

    def __init__(self, flops_fwd: float, model_bits: float) -> None:
        self.flops_fwd = flops_fwd
        self.model_bits = model_bits

    def train_hat(self, rv: ReportedVehicle) -> float:
        raise NotImplementedError

    def upload_hat(self, rv: ReportedVehicle, t_train_hat: float) -> tuple[float, float]:
        x_hat = F.wrap_x(rv.x + rv.speed * t_train_hat)
        return F.upload_seconds(F.distance_to_rsu(x_hat), 1.0, self.model_bits), x_hat

    def estimate(self, rv: ReportedVehicle) -> Estimate:
        t_train = self.train_hat(rv)
        t_up, x_hat = self.upload_hat(rv, t_train)
        return Estimate(T_train=t_train, T_up=t_up, x_finish=x_hat,
                        E=self.energy_hat(rv, t_train, t_up))

    def energy_hat(self, rv: ReportedVehicle, t_train: float, t_up: float) -> float:
        """GPU family (D2): P_tier * T_train_hat + p * T_up_hat."""
        e_train, e_up = F.energy_joules(rv.tier, t_train, t_up)
        return e_train + e_up

    def gpu_formula(self, rv: ReportedVehicle) -> float:
        return F.train_seconds(self.flops_fwd, rv.n_samples, K.LOCAL_EPOCHS,
                               rv.tflops_fp16)


class GpuEstimator(Estimator):
    name = "gpu"

    def train_hat(self, rv):
        return self.gpu_formula(rv)


class CpuEstimator(Estimator):
    """Wen Eq. 4. `cpu` uses the locked rho = 1,250; `cpu_cal` (Prompt D) is the
    same code with rho = rho_cal, the fleet-mean calibration (see rho_cal())."""

    def __init__(self, flops_fwd, model_bits, rho: float = K.CYCLES_PER_BIT_RHO,
                 name: str = "cpu") -> None:
        super().__init__(flops_fwd, model_bits)
        self.rho = float(rho)
        self.name = name

    def train_hat(self, rv):
        return F.cpu_train_seconds(rv.n_samples, rv.cpu_ghz, self.rho)

    def energy_hat(self, rv, t_train, t_up):
        """CPU family (D2, Wen Eq. 7 + 8): e rho b |S| f^2 + p * T_up_hat."""
        return (F.cpu_train_energy_joules(rv.n_samples, rv.cpu_ghz, self.rho)
                + K.UPLOAD_POWER_W * t_up)


def rho_cal(flops_fwd: float, fleet_reports: list[ReportedVehicle]) -> float:
    """Prompt D: rho such that the fleet-mean CPU estimate equals the fleet-mean
    datasheet (GPU, no jitter) training time.

        rho_cal = mean_i T_train_tier(i) / mean_i (b |S_i| / f_tier(i))

    Uses fleet-mean information a deployed system would not have; `cpu_cal` is
    a CPU-favourable ablation, not a deployable estimator (ADR-0016).
    """
    gpu = GpuEstimator(flops_fwd, 0.0)
    mean_t = sum(gpu.train_hat(rv) for rv in fleet_reports) / len(fleet_reports)
    mean_cycles_per_rho = sum(K.BITS_PER_IMAGE * rv.n_samples / (rv.cpu_ghz * 1e9)
                              for rv in fleet_reports) / len(fleet_reports)
    return mean_t / mean_cycles_per_rho


class GpuOracleEstimator(Estimator):
    name = "gpu_oracle"

    def __init__(self, flops_fwd, model_bits,
                 true_train_seconds: Callable[[int], float]) -> None:
        super().__init__(flops_fwd, model_bits)
        self._true = true_train_seconds

    def train_hat(self, rv):
        return self._true(rv.client_id)


class LocationOnlyEstimator(Estimator):
    name = "location_only"

    def __init__(self, flops_fwd, model_bits, constant_train_s: float) -> None:
        super().__init__(flops_fwd, model_bits)
        self.constant_train_s = constant_train_s

    def train_hat(self, rv):
        return self.constant_train_s


def build(arm: str, flops_fwd: float, model_bits: float,
          fleet_reports: list[ReportedVehicle],
          true_train_seconds: Callable[[int], float] | None = None,
          rho: float | None = None) -> Estimator:
    """The estimator for an arm. `fleet_reports` feeds location_only's constant."""
    if arm == "gpu":
        return GpuEstimator(flops_fwd, model_bits)
    if arm == "cpu":
        return CpuEstimator(flops_fwd, model_bits)
    if arm == "cpu_cal":
        if rho is None:
            raise ValueError("cpu_cal needs rho (rho_cal from the config)")
        return CpuEstimator(flops_fwd, model_bits, rho=rho, name="cpu_cal")
    if arm == "gpu_oracle":
        if true_train_seconds is None:
            raise ValueError("gpu_oracle needs the true-training-time callable")
        return GpuOracleEstimator(flops_fwd, model_bits, true_train_seconds)
    if arm == "location_only":
        gpu = GpuEstimator(flops_fwd, model_bits)
        mean = sum(gpu.train_hat(rv) for rv in fleet_reports) / len(fleet_reports)
        return LocationOnlyEstimator(flops_fwd, model_bits, mean)
    raise ValueError(f"no estimator for arm {arm!r}; expected one of {ESTIMATOR_ARMS}")
