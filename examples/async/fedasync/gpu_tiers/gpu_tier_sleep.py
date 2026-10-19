"""
GPU-tier-aware sleep-time computation for FedAsync IoV simulation.

Loads tiers.csv and clients.csv once (lazy, at first call) and provides a
per-call sleep_time formula so that noise_sigma can inject per-round variance
without any code changes — just edit the CSV.

Public API
----------
sleep_time_seconds(client_id, dataset_size, epochs=1) -> float
    Compute simulated training time for one client on its hardware tier.

tier_for(client_id) -> TierSpec
    Return the TierSpec for inspection / logging.

Configuration (read from TOML [gpu_tier] section via Config().config.gpu_tier)
    tiers_csv    – path to tiers.csv   (default: same directory as this file)
    clients_csv  – path to clients.csv (default: same directory as this file)
    anchor_tier  – tier_id whose predicted time is pinned to anchor_sleep
    anchor_sleep – seconds that tier is pinned to (see "Predictor anchoring")

Predictor anchoring
-------------------
The raw formula produces times on the order of milliseconds, far below PLATO's
own ~2 s per-client overhead, so the sleep CSVs that drive the SIMULATOR are
generated with ``generate_sleep_times.py --anchor-tier X --anchor-sleep N``,
which rescales every tier uniformly until tier X sleeps N seconds.

The PREDICTOR must be rescaled by the same factor.  Ratios alone are not
enough: ``select_clients`` compares a predicted duration against elapsed
simulated wall-clock time,

    residual = max(0, t_full - (wall_now - start_time))

and that subtraction is only meaningful if both operands are in the same units.
Left un-anchored the two sides differ by the anchor scale (363x for the GPU
tiers at anchor 20 s, and 0.073x — the other direction — for tiers_cpu.csv), so
every residual collapses to 0 or saturates, and the residual-aware Phase 1 of
gpu_tops_predict_staleness_rounds silently stops doing its job.

Set anchor_tier/anchor_sleep to the same values passed to
generate_sleep_times.py.  Omit them and the scale is 1.0 (raw formula, the
historical behaviour).  τ̂ = T_pred / T_round is unaffected either way — the
scale cancels there, because every predictor-side time goes through the same
factor.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# FLOPs per sample per epoch for ResNet-18 on 32×32 input.  1.1 GFLOP is the
# FORWARD pass only; harmless while every run was anchored, because the anchor
# scale cancels it.  Unanchored runs set [gpu_tier] w_model_flops = 3.3e9
# (forward + backward + optimizer ≈ 3 × forward) — see w_model_flops().
# ---------------------------------------------------------------------------
W_MODEL_FLOPS: float = 1.1e9

_MODULE_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Dataclasses — one per CSV row, immutable after load
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TierSpec:
    """One row from tiers.csv."""
    tier_id: str
    gpu_tflops_fp16: float   # Peak FP16 throughput (TFLOPS)
    eta_nominal: float        # Utilisation efficiency (0, 1]; 1.0 = ideal
    noise_sigma: float        # Std-dev of multiplicative noise; 0 = no noise


@dataclass(frozen=True)
class ClientSpec:
    """One row from clients.csv."""
    client_id: int
    tier_id: str


# ---------------------------------------------------------------------------
# Loader + formula
# ---------------------------------------------------------------------------

class GpuTierSleepComputer:
    """Loads tier and client CSVs; computes per-call sleep times."""

    def __init__(self, tiers_csv: str | Path, clients_csv: str | Path) -> None:
        self._tiers: dict[str, TierSpec] = {}
        self._clients: dict[int, ClientSpec] = {}
        self._load(Path(tiers_csv), Path(clients_csv))

    def _load(self, tiers_path: Path, clients_path: Path) -> None:
        with tiers_path.open(newline="") as f:
            for row in csv.DictReader(f):
                spec = TierSpec(
                    tier_id=row["tier_id"],
                    gpu_tflops_fp16=float(row["gpu_tflops_fp16"]),
                    eta_nominal=float(row["eta_nominal"]),
                    noise_sigma=float(row["noise_sigma"]),
                )
                self._tiers[spec.tier_id] = spec

        with clients_path.open(newline="") as f:
            for row in csv.DictReader(f):
                spec = ClientSpec(
                    client_id=int(row["client_id"]),
                    tier_id=row["tier_id"],
                )
                self._clients[spec.client_id] = spec

        logging.info(
            "[GpuTierSleep] Loaded %d tiers (%s) and %d clients.",
            len(self._tiers),
            ", ".join(self._tiers),
            len(self._clients),
        )

    def tier_for(self, client_id: int) -> TierSpec:
        """Return the TierSpec for a given client_id."""
        client = self._clients.get(client_id)
        if client is None:
            raise KeyError(
                f"[GpuTierSleep] client_id={client_id} not found in clients CSV."
            )
        tier = self._tiers.get(client.tier_id)
        if tier is None:
            raise KeyError(
                f"[GpuTierSleep] tier_id='{client.tier_id}' "
                f"(client {client_id}) not found in tiers CSV."
            )
        return tier

    def tier_by_id(self, tier_id: str) -> TierSpec:
        """Return the TierSpec named *tier_id* (used to resolve the anchor tier)."""
        tier = self._tiers.get(tier_id)
        if tier is None:
            raise KeyError(
                f"[GpuTierSleep] anchor tier_id='{tier_id}' not found in tiers CSV "
                f"(available: {', '.join(self._tiers)})."
            )
        return tier

    def sleep_time_seconds(
        self,
        client_id: int,
        dataset_size: int,
        epochs: int = 1,
    ) -> float:
        """
        Compute simulated training wall-time (seconds) for one client.

        Formula:
            work_flops = W_MODEL_FLOPS × dataset_size × epochs
            sleep_time = work_flops / (gpu_tflops_fp16 × 1e12 × eta_nominal)
            if noise_sigma > 0:
                sleep_time *= (1 + N(0, noise_sigma))

        Parameters
        ----------
        client_id    : PLATO client ID (1-based as PLATO assigns them)
        dataset_size : number of samples in this client's local partition
        epochs       : local training epochs for this call (typically 1)

        Returns
        -------
        Non-negative float — seconds to sleep (or account for in wall-time).
        """
        tier = self.tier_for(client_id)
        work_flops = w_model_flops() * dataset_size * epochs
        sleep = work_flops / (tier.gpu_tflops_fp16 * 1e12 * tier.eta_nominal)
        if tier.noise_sigma > 0.0:
            # Multiplicative noise — enables per-round variance when sigma > 0.
            # Keep noise_sigma = 0 in the CSV to disable (current default).
            noise = float(np.random.normal(0.0, tier.noise_sigma))
            sleep *= 1.0 + noise
            # Multiplicative Gaussian noise: sleep *= (1 + N(0, sigma)).
            # For sigma <= 0.15 the max(0.0, ...) clip almost never fires.
            # For sigma > 0.30 the clip introduces asymmetric bias — consider switching
            # to log-normal noise (sleep *= exp(N(0, sigma))) which guarantees positivity
            # without clipping. Current default sigma=0.0 means this path is inactive.
        return max(0.0, sleep)


# ---------------------------------------------------------------------------
# Module-level singleton — lazy-initialised on first call
# ---------------------------------------------------------------------------

_computer: Optional[GpuTierSleepComputer] = None
_t_round_logged: bool = False  # log compute_pool_T_round_seconds once per process

# [gpu_tier] t_round_model — resolved once by t_round_model(), see _t_round_kernel.
_T_ROUND_MODELS = frozenset({"min", "queue"})
_t_round_model: Optional[str] = None
_predictor_scale: Optional[float] = None  # cached anchor factor; None = not resolved
_w_model_flops: Optional[float] = None    # cached [gpu_tier] w_model_flops

# [gpu_tier] comm_bandwidth_mbps / payload_mbit / comm_in_predictor — resolved
# once by comm_latency_seconds() / comm_in_predictor().  See BRIEF_comm_sweep.md.
_DEFAULT_PAYLOAD_MBIT = 341.52  # 42.69 MB, the measured ResNet-18 payload (HANDOFF § 4.8)
_comm_latency: Optional[float] = None       # cached C = D/B seconds; None = not resolved
_comm_in_predictor: Optional[bool] = None   # cached belief-side switch; None = not resolved

# [gpu_tier] oracle_sleep_csv — the gpu_oracle arm.  Resolved once by
# oracle_sleeps().  `False` is the "resolved, and there is none" sentinel, kept
# distinct from `None` = "not resolved yet", because the absent case is the
# default and must not be re-read from Config on every prediction.
_oracle_sleeps: Optional[dict | bool] = None


def _resolve_paths() -> tuple[Path, Path]:
    """Read CSV paths from TOML [gpu_tier] section, fall back to module dir."""
    try:
        from plato.config import Config  # noqa: PLC0415 — lazy import
        gpu_tier_cfg = getattr(Config().config, "gpu_tier", None)
        if gpu_tier_cfg is not None:
            tiers = Path(os.path.expanduser(str(gpu_tier_cfg.tiers_csv)))
            clients = Path(os.path.expanduser(str(gpu_tier_cfg.clients_csv)))
            return tiers, clients
    except Exception:  # noqa: BLE001 — Config not yet initialised in tests
        pass
    return _MODULE_DIR / "tiers.csv", _MODULE_DIR / "clients.csv"


def get_computer() -> GpuTierSleepComputer:
    """Return the module-level singleton, initialising it on first call."""
    global _computer
    if _computer is None:
        tiers_path, clients_path = _resolve_paths()
        _computer = GpuTierSleepComputer(tiers_path, clients_path)
    return _computer


# ---------------------------------------------------------------------------
# Predictor anchoring — see "Predictor anchoring" in the module docstring
# ---------------------------------------------------------------------------

def w_model_flops() -> float:
    """Training FLOPs per sample per epoch, from ``[gpu_tier] w_model_flops``.

    Absent -> ``W_MODEL_FLOPS``, so every config written before this key existed
    is byte-identical.  The key is for UNANCHORED runs only: with an anchor the
    scale cancels W exactly, so setting both would be a second spelling of one
    run — refused, as is a non-positive value, rather than silently ignored.
    It must equal the ``w_model_flops`` in the fleet's manifest.json.
    """
    global _w_model_flops
    if _w_model_flops is not None:
        return _w_model_flops

    value = W_MODEL_FLOPS
    anchored = False
    raw = None
    try:
        from plato.config import Config  # noqa: PLC0415 — lazy import

        gpu_tier_cfg = getattr(Config().config, "gpu_tier", None)
        if gpu_tier_cfg is not None:
            raw = getattr(gpu_tier_cfg, "w_model_flops", None)
            anchored = getattr(gpu_tier_cfg, "anchor_tier", None) is not None
    except Exception:  # noqa: BLE001 — Config not yet initialised in tests
        raw = None

    if raw is not None:
        value = float(raw)
        if value <= 0.0:
            raise ValueError(
                f"[GpuTierSleep] [gpu_tier] w_model_flops = {raw!r} must be > 0."
            )
        if anchored:
            raise ValueError(
                "[GpuTierSleep] [gpu_tier] w_model_flops is set together with "
                "anchor_tier; the anchor cancels W, so the key is only valid on "
                "an unanchored run. Remove one of them."
            )
        logging.info("[GpuTierSleep] W_model = %.6g FLOP/sample/epoch", value)

    _w_model_flops = value
    return _w_model_flops


def _raw_seconds(tier: TierSpec, dataset_size: int, local_epochs: int) -> float:
    """Noise-free training time from the tier formula, before anchoring.

    Single definition of the formula for every predictor-side caller, so the
    anchor factor cannot be applied to one of them and forgotten in another.
    """
    work_flops = w_model_flops() * dataset_size * local_epochs
    return work_flops / (tier.gpu_tflops_fp16 * 1e12 * tier.eta_nominal)


def _reference_workload() -> tuple[int, int]:
    """(partition_size, epochs) the anchor factor is defined against.

    Uses the run's configured workload rather than each call's own arguments,
    so a client with a larger partition still predicts a proportionally longer
    time instead of being renormalised back onto the anchor.
    """
    try:
        from plato.config import Config  # noqa: PLC0415 — lazy import
        return (
            int(getattr(Config().data, "partition_size", 500)),
            int(getattr(Config().trainer, "epochs", 1)),
        )
    except Exception:  # noqa: BLE001 — Config not yet initialised in tests
        return 500, 1


def predictor_scale() -> float:
    """Uniform factor putting predicted times on the simulator's wall-clock scale.

    Resolved once per process from [gpu_tier] anchor_tier / anchor_sleep:

        scale = anchor_sleep / raw_seconds(anchor_tier, partition_size, epochs)

    Returns 1.0 when either key is absent, so configs written before anchoring
    existed keep their original behaviour.
    """
    global _predictor_scale
    if _predictor_scale is not None:
        return _predictor_scale

    tier_id: Optional[str] = None
    target: Optional[float] = None
    try:
        from plato.config import Config  # noqa: PLC0415 — lazy import
        gpu_tier_cfg = getattr(Config().config, "gpu_tier", None)
        if gpu_tier_cfg is not None:
            raw_tier = getattr(gpu_tier_cfg, "anchor_tier", None)
            raw_sleep = getattr(gpu_tier_cfg, "anchor_sleep", None)
            if raw_tier is not None and raw_sleep is not None:
                tier_id, target = str(raw_tier), float(raw_sleep)
    except Exception:  # noqa: BLE001 — Config not yet initialised in tests
        tier_id, target = None, None

    if tier_id is None or target is None:
        _predictor_scale = 1.0
        return _predictor_scale

    dataset_size, local_epochs = _reference_workload()
    raw = _raw_seconds(get_computer().tier_by_id(tier_id), dataset_size, local_epochs)
    if raw <= 0.0:
        raise ValueError(
            f"[GpuTierSleep] anchor tier '{tier_id}' has non-positive raw time ({raw})."
        )

    _predictor_scale = target / raw
    logging.info(
        "[GpuTierSleep] Predictor anchored: %s = %.6f s (raw %.9f s, "
        "scale %.6gx, partition_size=%d, epochs=%d).",
        tier_id, target, raw, _predictor_scale, dataset_size, local_epochs,
    )
    return _predictor_scale


def comm_latency_seconds() -> float:
    """C = D / B, the uplink transfer time added on top of compute (seconds).

    Resolved once per process from ``[gpu_tier] comm_bandwidth_mbps`` (B, Mbps)
    and ``[gpu_tier] payload_mbit`` (D, Mbit; defaults to the measured 341.52 Mbit
    ResNet-18 payload, HANDOFF § 4.8).  Uplink only — see BRIEF_comm_sweep.md § 2.

    Absent ``comm_bandwidth_mbps`` -> C = 0.0, byte-identical to every run made
    before this feature existed.  A key that IS present but non-positive raises
    rather than silently falling back to 0 — the same discipline as
    ``t_round_model()``, because a silent fallback here would burn hours of
    compute on a run that looks like the real sweep but is not.
    """
    global _comm_latency
    if _comm_latency is not None:
        return _comm_latency

    bandwidth: Optional[float] = None
    payload = _DEFAULT_PAYLOAD_MBIT
    try:
        from plato.config import Config  # noqa: PLC0415 — lazy import

        gpu_tier_cfg = getattr(Config().config, "gpu_tier", None)
        if gpu_tier_cfg is not None:
            raw_bw = getattr(gpu_tier_cfg, "comm_bandwidth_mbps", None)
            if raw_bw is not None:
                bandwidth = float(raw_bw)
            raw_payload = getattr(gpu_tier_cfg, "payload_mbit", None)
            if raw_payload is not None:
                payload = float(raw_payload)
    except Exception:  # noqa: BLE001 — Config not yet initialised in tests
        bandwidth = None

    if bandwidth is None:
        _comm_latency = 0.0
        return _comm_latency
    if bandwidth <= 0.0:
        raise ValueError(
            f"[GpuTierSleep] [gpu_tier] comm_bandwidth_mbps = {bandwidth!r} must be > 0."
        )

    _comm_latency = payload / bandwidth
    logging.info(
        "[GpuTierSleep] Comm latency: C = %.6f s (B=%.6g Mbps, D=%.6g Mbit)",
        _comm_latency, bandwidth, payload,
    )
    return _comm_latency


def comm_in_predictor() -> bool:
    """Whether the SELECTOR's belief includes comm latency C (BRIEF_comm_sweep.md § 3).

    Resolved once per process from ``[gpu_tier] comm_in_predictor``.  Defaults to
    True — a calibrated selector whose belief matches reality — so the ordinary
    comm-sweep cells (arm A) need no extra key.  The mis-calibrated ablation
    (arm B: reality has C, belief does not) sets it to False explicitly; it is
    never inferred.  Irrelevant when ``comm_bandwidth_mbps`` is absent, since C
    is 0 either way.
    """
    global _comm_in_predictor
    if _comm_in_predictor is not None:
        return _comm_in_predictor

    value = True
    try:
        from plato.config import Config  # noqa: PLC0415 — lazy import

        gpu_tier_cfg = getattr(Config().config, "gpu_tier", None)
        if gpu_tier_cfg is not None:
            raw = getattr(gpu_tier_cfg, "comm_in_predictor", None)
            if raw is not None:
                value = bool(raw)
    except Exception:  # noqa: BLE001 — Config not yet initialised in tests
        value = True

    _comm_in_predictor = value
    return _comm_in_predictor


def _predicted_seconds(tier: TierSpec, dataset_size: int, local_epochs: int) -> float:
    """Predictor-side training time: scaled compute term, plus C if believed.

    C is added AFTER predictor_scale(), never folded into the raw compute term
    before scaling — the raw formula and the anchored/simulator scale differ by
    363x (§4.1), and C is already a real-seconds quantity in the simulator's
    units, so scaling it too would be the same bug in a new shape
    (BRIEF_comm_sweep.md § 4.2). Single definition so every predictor-side
    caller (gpu_tops_predict_seconds, compute_pool_T_round_seconds,
    compute_pool_mean_T_seconds) computes T_round from the same augmented times
    it predicts individual clients from (§ 4.4) — no second path to drift out of
    sync with this one.
    """
    t = _raw_seconds(tier, dataset_size, local_epochs) * predictor_scale()
    if comm_in_predictor():
        t += comm_latency_seconds()
    return t


def oracle_sleeps() -> Optional[dict]:
    """Per-client TRUE sleep times, or None when this is not the gpu_oracle arm.

    Resolved once per process from ``[gpu_tier] oracle_sleep_csv``.  Absent — the
    default, and what every config written before the gpu_oracle arm existed does
    — returns None and every predictor path below behaves exactly as it always
    has.  A key that IS present but unreadable RAISES, following the
    ``t_round_model()`` precedent: a silent fallback would turn the upper-bound
    arm into a duplicate of the ``gpu`` arm, and nothing in the results would
    distinguish them.

    Why the arm exists (gpu_tiers_dataset/prompts/STAGE1_code_changes.md)
    --------------------------------------------------------------------
    The sleep the simulator hands a client is ``T_tier * exp(eps_i)``, where
    ``eps_i`` is a persistent per-unit offset the datasheet cannot know.  The
    ``gpu`` arm therefore believes the TIER MEAN and is wrong by ``exp(eps_i)``
    per client — which is the point: it is the deployable method, and the thesis
    claim is that a spec-derived estimate is good enough.

    ``gpu_oracle`` believes ``sleep_i`` exactly.  It is not deployable; it exists
    to bound how much the spec-derived version gives up.

    The file this points at must be the SAME ``sleep_times.csv`` the config's
    ``[clients.simulation_distribution] csv_path`` points at — an oracle reading
    a different fleet is not an oracle, and the failure is silent.  The
    ``preflight.py`` in ``gpu_tiers_dataset/`` is what enforces it.
    """
    global _oracle_sleeps
    if _oracle_sleeps is not None:
        return _oracle_sleeps or None

    raw_path = None
    try:
        from plato.config import Config  # noqa: PLC0415 — lazy import

        gpu_tier_cfg = getattr(Config().config, "gpu_tier", None)
        if gpu_tier_cfg is not None:
            raw_path = getattr(gpu_tier_cfg, "oracle_sleep_csv", None)
    except Exception:  # noqa: BLE001 — Config not yet initialised in tests
        raw_path = None

    if raw_path is None:
        _oracle_sleeps = False
        return None

    path = Path(os.path.expanduser(str(raw_path)))
    if not path.exists():
        raise ValueError(
            f"[GpuTierSleep] [gpu_tier] oracle_sleep_csv = {path} does not exist. "
            "This key selects the gpu_oracle arm; falling back to the tier-mean "
            "prediction would make it an undetectable duplicate of the gpu arm."
        )

    with path.open(newline="") as f:
        sleeps = {
            int(row["client_id"]): float(row["sleep_time"])
            for row in csv.DictReader(f)
        }
    if not sleeps:
        raise ValueError(f"[GpuTierSleep] no rows in oracle_sleep_csv {path}.")

    _oracle_sleeps = sleeps
    logging.info(
        "[GpuTierSleep] ORACLE predictor: per-client true sleep from %s "
        "(%d clients, %.6f..%.6f s). This arm is an upper bound, not a "
        "deployable method.",
        path, len(sleeps), min(sleeps.values()), max(sleeps.values()),
    )
    return _oracle_sleeps


def _client_seconds(client_id: int, dataset_size: int, local_epochs: int) -> float:
    """Predicted training time for ONE client, on whichever belief this arm holds.

    The single place the oracle branch is taken, so the three predictor-side
    entry points — ``gpu_tops_predict_seconds``, ``compute_pool_T_round_seconds``
    and ``compute_pool_mean_T_seconds`` — cannot end up holding different
    beliefs.  That matters more than it looks: the SELECTOR computes busy-client
    residuals from ``gpu_tops_predict_seconds`` and its denominator from
    ``compute_t_round_from_times`` over the same values
    (fedasync_polaris_vi_selection.py), so an oracle applied to one and not the
    others would be a half-oracle whose behaviour no table could describe.

    The oracle value is the simulator's own sleep time and is already in
    simulator units and already carries any comm latency the fleet was built
    with, so it is NOT put through ``predictor_scale()`` and C is NOT added to
    it again.
    """
    sleeps = oracle_sleeps()
    if sleeps is None:
        return _predicted_seconds(tier_for(client_id), dataset_size, local_epochs)

    try:
        return sleeps[client_id]
    except KeyError:
        raise KeyError(
            f"[GpuTierSleep] client_id={client_id} is not in the oracle sleep "
            f"CSV. The oracle file and the fleet must be the same fleet."
        ) from None


def t_round_model() -> str:
    """Which inter-merge-interval model T_round uses: "min" or "queue".

    Resolved once per process from ``[gpu_tier] t_round_model``.  Defaults to
    ``"min"`` — the historical behaviour — so a config written before this key
    existed produces a byte-identical run.  See ``_t_round_kernel`` for what the
    two models mean and why "queue" is the correct one.
    """
    global _t_round_model
    if _t_round_model is not None:
        return _t_round_model

    choice = "min"
    try:
        from plato.config import Config  # noqa: PLC0415 — lazy import

        gpu_tier_cfg = getattr(Config().config, "gpu_tier", None)
        if gpu_tier_cfg is not None:
            raw = getattr(gpu_tier_cfg, "t_round_model", None)
            if raw is not None:
                choice = str(raw).strip().lower()
    except Exception:  # noqa: BLE001 — Config not yet initialised in tests
        choice = "min"

    if choice not in _T_ROUND_MODELS:
        # Fail loudly.  A silent fallback here would put two arms of the sweep on
        # different staleness scales with nothing in the log to say so — the same
        # class of bug as the `staleness_predictor` typo trap (HANDOFF § 4.10).
        raise ValueError(
            f"[GpuTierSleep] [gpu_tier] t_round_model = {choice!r} is not "
            f"recognised; expected one of {sorted(_T_ROUND_MODELS)}."
        )

    _t_round_model = choice
    logging.info("[GpuTierSleep] T_round model: %s", choice)
    return _t_round_model


def _t_round_kernel(per_client_T: list, clients_per_round: int) -> float:
    """Expected interval between two consecutive aggregation events (seconds).

    In async FL one "round" is one merge, so T_round is the denominator that
    turns a client's training time into a staleness in rounds:
    ``τ̂_i = T_i / T_round``.  Two models, selected by ``t_round_model()``:

    "queue" — steady-state closed queue (**this is the correct one**)
        ``C`` slots are kept full; whenever one frees it is refilled with a
        freshly drawn client.  Each slot is therefore a renewal process with
        mean cycle ``E[T]``, so it completes ``1/E[T]`` jobs per second and the
        ``C`` slots together merge at ``C/E[T]``::

            T_round = E[T] / C

        Note ``E[T]`` is the **arithmetic** mean.  The harmonic mean would be
        right only if each slot were pinned to one fixed client for the whole
        run; PLATO re-draws on every refill, so it is not.

    "min" — E[min of P i.i.d. draws], the historical behaviour
        The expected time to the *first* completion when all ``P`` clients start
        **simultaneously**.  That describes a cold start, not the steady state:
        once running, the in-flight jobs have staggered start times and their
        residual times are spread out, so the next completion arrives far sooner
        than "min of P fresh draws" implies.

    Why it matters that "min" is wrong in a *gap-dependent* way
    -----------------------------------------------------------
    The error is not a constant factor that cancels in ``τ̂``.  Measured against
    the observed seconds/round of the four finished gpu runs (P=10, C=9):

    ======  ===========  ===========  ===========
    gap G   measured     "min"        "queue"
    ======  ===========  ===========  ===========
    1        2.183 s     +816.1 %     +1.8 %
    10       1.336 s     +114.5 %     -3.3 %
    20       1.205 s      +40.9 %     -5.0 %
    30       1.147 s      +10.8 %     -7.4 %
    ======  ===========  ===========  ===========

    At G=1 every client takes the same 20 s, so ``min of 10`` *is* 20 s and the
    denominator is the full client duration — predicted staleness collapses to
    1.0 round against an observed 9.0.  As the tier spread widens the fast tier
    starts to dominate the minimum and the two models converge by accident.

    Since ``V_i = G_i / sqrt(1 + τ̂_i)``, an under-estimated τ̂ weakens the speed
    term.  Under "min" that weakening is strongest at exactly the narrow-gap end
    of the sweep and vanishes at the wide-gap end — i.e. the strength of the
    mechanism the sweep exists to measure varies with the sweep's own
    independent variable.  That is a confound on the x-axis, the same shape of
    problem as the un-anchored predictor in HANDOFF § 4.1.

    Parameters
    ----------
    per_client_T      : training times (s) of the clients this estimate is for —
                        the whole pool at startup, the live busy set at run time.
    clients_per_round : configured concurrency ``P``.  The divisor is
                        ``min(P, len(per_client_T))`` so the startup call over
                        all 100 clients still divides by the 10 that are really
                        in flight, while the live call divides by the busy set's
                        own size (9 in practice, since one slot is mid-refill).
    """
    if not per_client_T:
        return 0.0

    P = max(1, clients_per_round)

    if t_round_model() == "queue":
        return float(np.mean(per_client_T)) / min(P, len(per_client_T))

    # "min": E[min of P i.i.d. samples] via order statistics.
    #     P(min = v_k) = (1 - F_{k-1})^P - (1 - F_k)^P
    total = len(per_client_T)
    cdf = 0.0
    t_round = 0.0
    for v in sorted(set(per_client_T)):
        p_v = per_client_T.count(v) / total
        t_round += v * ((1.0 - cdf) ** P - (1.0 - cdf - p_v) ** P)
        cdf += p_v
    return t_round


def generate_client_sleep_csv(
    output_path: str | Path,
    total_clients: int,
    dataset_size: int,
    epochs: int = 1,
    tiers_csv: str | Path | None = None,
    clients_csv: str | Path | None = None,
) -> None:
    """
    Write a PLATO-compatible sleep-time CSV (client_id, sleep_time) to
    *output_path* using the GPU-tier formula.

    Call this once whenever tiers.csv, partition_size, or epochs change.
    The resulting file is read by PLATO's 'custom' simulation_distribution
    in every client subprocess via Config.simulate_client_speed().

    Parameters
    ----------
    output_path   : destination CSV path (created/overwritten)
    total_clients : number of clients (CSV will have rows 1..total_clients)
    dataset_size  : samples per client (PLATO's partition_size)
    epochs        : local training epochs per round
    tiers_csv     : override path for tiers.csv (default: module directory)
    clients_csv   : override path for clients.csv (default: module directory)
    """
    import csv as _csv  # noqa: PLC0415

    t_path = Path(tiers_csv) if tiers_csv else _MODULE_DIR / "tiers.csv"
    c_path = Path(clients_csv) if clients_csv else _MODULE_DIR / "clients.csv"
    computer = GpuTierSleepComputer(t_path, c_path)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["client_id", "sleep_time"])
        for cid in range(1, total_clients + 1):
            sleep = computer.sleep_time_seconds(cid, dataset_size, epochs)
            writer.writerow([cid, f"{sleep:.9f}"])

    logging.info(
        "[GpuTierSleep] Generated %s (%d clients, dataset_size=%d, epochs=%d).",
        out,
        total_clients,
        dataset_size,
        epochs,
    )


def tier_for(client_id: int) -> TierSpec:
    """Convenience wrapper — returns the TierSpec for *client_id*."""
    return get_computer().tier_for(client_id)


def sleep_time_seconds(
    client_id: int,
    dataset_size: int,
    epochs: int = 1,
) -> float:
    """
    Convenience wrapper around ``GpuTierSleepComputer.sleep_time_seconds``.

    Computes on every call — no caching — so noise_sigma produces fresh
    variance each round once turned on in tiers.csv.
    """
    return get_computer().sleep_time_seconds(client_id, dataset_size, epochs)


# ---------------------------------------------------------------------------
# Pool-level T_round estimation and conversion utilities
# ---------------------------------------------------------------------------

def compute_pool_T_round_seconds(
    client_ids: list,
    partition_sizes: dict,
    local_epochs: int,
    clients_per_round: int = 1,
) -> float:
    """Return the expected inter-report interval for the async round clock (seconds).

    In async FL, one "round" = one aggregation event = one client finishes.
    With ``clients_per_round`` concurrent clients, the server clock advances as
    soon as the FASTEST of those clients finishes — not when the median one does.
    T_round must therefore be the expected minimum training time across a batch
    of ``clients_per_round`` randomly-drawn clients, not the pool median.

    Why this matters for staleness prediction
    ------------------------------------------
    The selector computes τ̂_i = T_C / T_round.  If T_round is the pool median
    and clients_per_round = 10, the denominator is ~7× too large, so every
    predicted staleness is ~7× too small (e.g. Xavier predicts 1.8 rounds but
    observes 5–12).

    Formula — E[min of P i.i.d. samples from the pool distribution]:
        Sort the distinct training-time values v_1 < v_2 < ... < v_K
        with pool probabilities p_1, p_2, ..., p_K (CDF = F_k = sum_{j<=k} p_j).
        P(min = v_k) = (1 - F_{k-1})^P - (1 - F_k)^P
        E[T_round] = sum_k v_k * P(min = v_k)

    When clients_per_round = 1 the formula reduces to the pool mean (not
    median), which is very close to the median for near-symmetric distributions.

    Parameters
    ----------
    client_ids        : iterable of PLATO client IDs (1-based)
    partition_sizes   : mapping from client_id -> number of local training samples
    local_epochs      : local training epochs per round
    clients_per_round : number of clients training concurrently (P in the formula)

    Returns
    -------
    E[min of clients_per_round samples from the pool training-time distribution]

    Notes
    -----
    Do NOT substitute gpu_tier_sleep.sleep_time_seconds() here.  That path
    applies multiplicative Gaussian noise on every call (when noise_sigma > 0),
    making T_round non-reproducible across calls.  This function always uses
    the deterministic, noise-free expected value so T_round is stable across
    the entire experiment run.
    """
    global _t_round_logged
    computer = get_computer()

    per_client_T: list = []
    tier_times: dict = {}

    for cid in client_ids:
        tier = computer.tier_for(cid)
        dataset_size = partition_sizes[cid]
        # Noise-free expected value — do NOT call sleep_time_seconds() here.
        # That path applies multiplicative Gaussian noise per call, making
        # T_round non-reproducible when noise_sigma > 0 in the CSV.  Goes
        # through _client_seconds() (scale + optional comm C, or the oracle's
        # per-client truth) so T_round is derived from the same times the
        # selector predicts individual clients from.
        t = _client_seconds(cid, dataset_size, local_epochs)
        per_client_T.append(t)
        tier_times.setdefault(tier.tier_id, []).append((t, tier))

    # Expected interval between two consecutive merges, per t_round_model().
    P = max(1, clients_per_round)
    t_round = _t_round_kernel(per_client_T, P)

    t_round_median = float(np.median(per_client_T))  # kept for logging comparison

    if not _t_round_logged:
        _t_round_logged = True
        logging.info(
            "[PolarisVi] T_round = %.6f s  (model=%s, %d concurrent clients,"
            " pool median = %.6f s, speedup = %.2fx)",
            t_round,
            t_round_model(),
            P,
            t_round_median,
            t_round_median / max(t_round, 1e-12),
        )
        for tid, entries in sorted(tier_times.items()):
            times = [e[0] for e in entries]
            spec = entries[0][1]
            logging.info(
                "[PolarisVi] T_round per-tier: %-8s  median=%.6f s  "
                "n=%d  η=%.4f  σ=%.4f",
                tid,
                float(np.median(times)),
                len(times),
                spec.eta_nominal,
                spec.noise_sigma,
            )

    return t_round


def seconds_to_rounds(
    T_seconds: float,
    T_round_seconds: float,
) -> float:
    """Convert predicted training time (seconds) to predicted staleness (rounds).

    τ̂_rounds = T_seconds / T_round_seconds

    Returns a non-negative float. Raises ValueError if T_round_seconds <= 0.
    """
    if T_round_seconds <= 0.0:
        raise ValueError(
            f"T_round_seconds must be > 0, got {T_round_seconds}"
        )
    return max(0.0, T_seconds / T_round_seconds)


def compute_pool_mean_T_seconds(
    client_ids: list,
    partition_sizes: dict,
    local_epochs: int,
) -> float:
    """Return the noise-free mean single-client training time across the pool.

    Used as the replacement-client time in gpu_tops_predict_staleness_rounds:
    when a busy slot frees during the simulation, a new client with this
    expected training time joins the queue.
    """
    computer = get_computer()
    times = []
    for cid in client_ids:
        tier = computer.tier_for(cid)
        times.append(_client_seconds(cid, partition_sizes[cid], local_epochs))
    return float(np.mean(times)) if times else 0.0


def compute_t_round_from_times(
    per_client_T: list[float],
    clients_per_round: int = 1,
) -> float:
    """Compute T_round from pre-computed per-client times.

    Same model as compute_pool_T_round_seconds (see ``_t_round_kernel``), but
    accepts training times directly so callers can supply the live busy set or
    EMA-updated estimates rather than re-deriving from the GPU-tier CSV.

    Returns 0.0 if per_client_T is empty.
    """
    return _t_round_kernel(per_client_T, clients_per_round)


def gpu_tops_predict_staleness_rounds(
    t_client: float,
    busy_residuals: list,
    T_round: float,
) -> float:
    """Predict staleness using residual-aware two-phase simulation.

    Phase 1 — drain the known residuals:
        Count how many of the currently-busy clients finish before t_client.
        Each client finishing = one round increment.  Multiple clients with
        equal residuals (same tier, started together) count individually since
        PLATO increments current_round once per SequentialFedAsync step.
        Clients whose residual ≥ t_client do not finish before our client and
        are not counted.

    Phase 2 — steady-state tail:
        After the initial residuals are exhausted, replacement clients join at
        rate 1/T_round.  The remaining time is divided by T_round to get the
        expected additional events.

    The two-phase design avoids the replacement-cascade artefact that arises
    when a fixed pool_mean replacement time is used: identical-tier clients
    expire simultaneously, adding many same-valued entries that then all expire
    together in the next step.

    Parameters
    ----------
    t_client       : full training time of the newly selected client (s)
    busy_residuals : residual training times of currently-busy clients (s)
    T_round        : expected inter-report interval for the steady-state tail (s);
                     use the global compute_pool_T_round_seconds value
    """
    # Phase 1: count initial busy-client finishes that precede t_client.
    # Clients with residual >= t_client finish after us — skip them.
    early_finishes = sorted(r for r in busy_residuals if 0.0 < r < t_client)
    events = len(early_finishes)

    # Phase 2: steady-state tail after the last Phase-1 event.
    # T_round is calibrated for P concurrent clients, but after Phase 1 only
    # n_early replacement slots are "active contributors" — the slow survivors
    # (residual >= t_client) are still running but finish after us and add 0
    # events before t_client.  Scale Phase 2 by n_early / n_total so the
    # effective rate reflects only the replacement slots.
    t_last_early = early_finishes[-1] if early_finishes else 0.0
    remaining = t_client - t_last_early
    if remaining > 0.0 and T_round > 0.0:
        n_total = len(busy_residuals) or 1
        phase2_scale = max(len(early_finishes), 1) / n_total
        events += int(remaining * phase2_scale / T_round)

    return float(events)


def gpu_tops_predict_seconds(
    client_id: int,
    dataset_size: int,
    local_epochs: int,
) -> float:
    """GPU-TOPS-based prediction of training time in seconds.

    Uses the same formula as compute_pool_T_round_seconds (so the resulting
    τ̂_rounds = T_pred / T_round is dimensionally consistent), but for a
    single client.

    This is the deterministic, noise-free prediction the SELECTOR uses.
    The SIMULATOR applies noise on top of this via gpu_tier_sleep — those
    are separate code paths intentionally (selector ignores noise, simulator
    applies it). Selection bias from noise = prediction error, which is the
    quantity Exp 3 is designed to measure.

    The result carries the predictor_scale() anchor factor, so it is directly
    comparable with the simulator's wall clock (see "Predictor anchoring").  It
    also carries comm latency C when ``[gpu_tier] comm_in_predictor`` is true
    (the default) — see ``_predicted_seconds`` and BRIEF_comm_sweep.md.
    """
    return _client_seconds(client_id, dataset_size, local_epochs)
