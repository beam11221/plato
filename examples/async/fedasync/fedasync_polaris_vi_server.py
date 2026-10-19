"""
FedAsync server wired with Polaris G_i tracking and V_i client selection.

Architecture
------------
FedAsyncPolarisAggregationStrategy
    Extends SequentialFedAsyncAggregationStrategy (FedAsync weight mixing in
    arrival order) and adds a G_i update step inside aggregate_weights().

    G_i is computed as the L2 norm of conv-layer weight deltas:

        delta_i[l] = w_client_i[l] - baseline_weights[l]   (for conv layers l)
        G_i = ||{delta_i[l]}||_2

    Unexplored clients (never yet selected) receive a proxy:
        G_i = alpha * mean_norm_of_observed_clients

    Results are written to context.state["polaris"]["local_gradient_bounds"]
    so that PolarisViClientSelectionStrategy can read them next round.

Server
------
Subclasses fedasync_server.Server to inherit FedAsync-specific configure(),
_process_reports(), and _select_clients() overrides, but bypasses the parent's
__init__ to inject custom aggregation and selection strategies.

Import path for PolarisAggregationStrategy (used for setup helpers):
    examples/client_selection/polaris/polaris_aggregation_strategy.py
    Added to sys.path here so no PLATO core modifications are needed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import fedasync_aggregation
import fedasync_polaris_vi_selection
import fedasync_server
import gpu_tier_sleep

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies.base import ServerContext


# ---------------------------------------------------------------------------
# [server] staleness_predictor — the one key that decides which arm this is
# ---------------------------------------------------------------------------

STALENESS_PREDICTORS = frozenset({"gpu_tops", "cpu", "observed", "gpu_oracle"})
DEFAULT_STALENESS_PREDICTOR = "observed"

# Predictors that work in SECONDS (a spec- or truth-derived training time, turned
# into rounds by the selector) rather than in rounds (the observed-tau estimator).
SECONDS_PREDICTORS = frozenset({"gpu_tops", "cpu", "gpu_oracle"})

# Sentinel for check_oracle_wiring's injectable argument: `None` is a meaningful
# value there (the key is absent), so it cannot double as "not supplied".
_UNSET = object()


def resolve_staleness_predictor(raw) -> str:
    """Validate ``[server] staleness_predictor`` and return its canonical name.

    An ABSENT key still resolves to ``"observed"``, so every config written
    before this key existed produces a byte-identical run.  An unrecognised
    value RAISES.

    Why it raises rather than falling back (HANDOFF § 4.10 vs § 4.11).  This key
    used to be read as ``getattr(server_cfg, "staleness_predictor", "observed")``
    and then tested with ``in ("gpu_tops", "cpu")``, so anything it did not
    recognise — ``"GPU_TOPS"``, ``"gpu_top"``, a stray space — silently became
    the observed-τ predictor.  That was already a trap when ``observed`` was
    merely the default.  It became a much worse one when ``observed`` became a
    real ARM of a sweep (surgical/BRIEF_surgical.md): a typo in the gpu or cpu
    config no longer yields a visibly wrong run, it yields a run that is
    *behaviourally identical to another arm of the same experiment*, with
    nothing in the results to distinguish the duplicate.  Six hours of GPU time
    would be spent measuring the same arm twice.

    Case and surrounding whitespace are normalised, following the
    ``gpu_tier_sleep.t_round_model()`` precedent.  The generator
    (``gpu_tiers/make_configs.py``) additionally refuses to WRITE a config whose
    predictor does not match the arm implied by its own filename, so this
    function is the backstop for hand-written configs, not the only guard.
    """
    if raw is None:
        return DEFAULT_STALENESS_PREDICTOR

    name = str(raw).strip().lower()
    if name not in STALENESS_PREDICTORS:
        raise ValueError(
            f"[PolarisVi] [server] staleness_predictor = {raw!r} is not "
            f"recognised; expected one of {sorted(STALENESS_PREDICTORS)}. "
            "Omit the key entirely to get "
            f"{DEFAULT_STALENESS_PREDICTOR!r}. This is deliberately fatal: an "
            "unrecognised value used to fall back to the observed-τ predictor "
            "silently, which would now produce an undetectable duplicate of "
            "another arm of the sweep."
        )
    return name


def check_oracle_wiring(predictor_name: str, oracle_csv=_UNSET) -> None:
    """Fail unless ``[gpu_tier] oracle_sleep_csv`` and the arm agree.

    The key and the arm name are two halves of one switch, written in two
    different sections of the TOML by a generator, and either half alone produces
    a run that is silently a DIFFERENT arm of the same sweep:

      * ``gpu_oracle`` without the key -> the predictor falls back to the tier
        mean, i.e. it becomes a second copy of the ``gpu`` arm.  The results
        would show the deployable method where the upper bound should be, and
        the headroom between them would read as zero.
      * ``gpu_tops`` (the ``gpu`` arm) WITH the key -> the arm the thesis claim
        rests on quietly regains the oracle it was just split away from, which
        is the exact defect this stage exists to remove.
      * ``cpu`` or ``observed`` with the key -> same, for an arm whose whole
        definition is the belief it holds.

    Neither is visible in an accuracy curve, so both are fatal here.  This is the
    run-time backstop; ``gpu_tiers_dataset/preflight.py`` catches it before any
    GPU time is spent, and additionally checks that the oracle file is the same
    file the clients actually sleep from.
    """
    if oracle_csv is _UNSET:
        gpu_tier_cfg = getattr(Config().config, "gpu_tier", None)
        oracle_csv = getattr(gpu_tier_cfg, "oracle_sleep_csv", None)

    if predictor_name == "gpu_oracle" and oracle_csv is None:
        raise ValueError(
            "[PolarisVi] staleness_predictor = 'gpu_oracle' requires "
            "[gpu_tier] oracle_sleep_csv, which is absent. Without it this arm "
            "predicts the tier mean and is an undetectable duplicate of the gpu "
            "arm."
        )
    if predictor_name != "gpu_oracle" and oracle_csv is not None:
        raise ValueError(
            f"[PolarisVi] [gpu_tier] oracle_sleep_csv is set but "
            f"staleness_predictor = {predictor_name!r}. That key gives an arm "
            f"per-client ground truth; only 'gpu_oracle' may hold it. Remove the "
            f"key, or set the predictor to 'gpu_oracle'."
        )


# ---------------------------------------------------------------------------
# Aggregation strategy: FedAsync mixing + Polaris G_i tracking
# ---------------------------------------------------------------------------

class FedAsyncPolarisAggregationStrategy(
    fedasync_aggregation.SequentialFedAsyncAggregationStrategy
):
    """Sequential FedAsync mixing with per-round G_i updates for Polaris V_i.

    Inherits the staleness-aware sequential weight mixing from
    SequentialFedAsyncAggregationStrategy, then computes conv-layer gradient
    bounds from weight deltas and writes them to the shared polaris state dict.
    """

    def __init__(
        self,
        alpha: float = 10.0,
        initial_gradient_bound: float = 0.5,
        initial_staleness: float = 0.01,
        t_round_seconds: float = 0.0,
        partition_sizes: dict | None = None,
        local_epochs: int = 1,
        pool_mean_T_seconds: float = 0.0,
        clients_per_round: int = 1,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.initial_gradient_bound = initial_gradient_bound
        self.initial_staleness = initial_staleness
        self._t_round_seconds = t_round_seconds
        self._partition_sizes: dict = partition_sizes or {}
        self._local_epochs: int = local_epochs
        self._pool_mean_T_seconds: float = pool_mean_T_seconds
        self._clients_per_round: int = clients_per_round
        self.total_clients: int = 0
        self.unexplored_clients: set[int] | None = None
        # The proxy G_i most recently given to unexplored clients.  Read by
        # Stage 7b's recycle reset so a newly arrived identity carries the same
        # prior as every other never-seen client; nothing here reads it.
        self.last_proxy: float | None = None
        # The mean measured norm the proxy was built from (proxy = alpha x this).
        # Recorded for Stage 9's pre-flight; nothing here reads it.
        self.last_proxy_basis: float | None = None

    def setup(self, context: ServerContext) -> None:
        super().setup(context)
        self.total_clients = context.total_clients
        self.unexplored_clients = set(range(self.total_clients))

        polaris_state = context.state.setdefault("polaris", {})
        polaris_state.setdefault(
            "local_gradient_bounds",
            np.full(self.total_clients, self.initial_gradient_bound, dtype=float),
        )
        polaris_state.setdefault(
            "local_stalenesses",
            np.full(self.total_clients, self.initial_staleness, dtype=float),
        )
        polaris_state.setdefault(
            "aggregation_weights",
            np.full(
                self.total_clients,
                1.0 / max(1, self.total_clients),
                dtype=float,
            ),
        )
        # Write shared config so the selection strategy can read it from
        # context.state without needing a direct reference to this object.
        context.state["T_round_seconds"] = self._t_round_seconds
        context.state["partition_sizes"] = self._partition_sizes
        context.state["local_epochs"] = self._local_epochs
        context.state["pool_mean_T_seconds"] = self._pool_mean_T_seconds
        context.state["clients_per_round"] = self._clients_per_round

    async def aggregate_weights(
        self,
        updates: list[SimpleNamespace],
        baseline_weights: dict,
        weights_received: list[dict],
        context: ServerContext,
    ) -> dict:
        # Step 1: FedAsync sequential weight mixing (staleness-aware)
        result = await super().aggregate_weights(
            updates, baseline_weights, weights_received, context
        )

        # Step 2: Compute G_i from conv-layer weight deltas
        if updates and weights_received and baseline_weights:
            self._update_gradient_bounds(
                updates, baseline_weights, weights_received, context
            )

        return result

    def _update_gradient_bounds(
        self,
        updates: list[SimpleNamespace],
        baseline_weights: dict,
        weights_received: list[dict],
        context: ServerContext,
    ) -> None:
        """Compute G_i = ||conv_delta_i||_2 for each reporting client.

        Clients that have never reported receive a proxy:
            G_i = alpha * mean_G_of_observed
        """
        polaris_state = context.state.setdefault("polaris", {})
        local_gradient_bounds: np.ndarray = polaris_state["local_gradient_bounds"]

        norm_deltas = np.zeros(self.total_clients, dtype=float)
        sum_norms = 0.0
        count = 0

        for update, w_client in zip(updates, weights_received):
            client_index = update.client_id - 1
            squared_delta = 0.0

            for layer_name, w_val in w_client.items():
                if "conv" not in layer_name:
                    continue
                if layer_name not in baseline_weights:
                    continue

                delta = w_val - baseline_weights[layer_name]
                if isinstance(delta, torch.Tensor):
                    delta_np = delta.detach().cpu().numpy()
                else:
                    delta_np = np.asarray(delta, dtype=float)

                squared_delta += float(np.sum(np.square(delta_np)))

            norm_delta = float(np.sqrt(max(squared_delta, 0.0)))
            norm_deltas[client_index] = norm_delta

            if (
                self.unexplored_clients is not None
                and client_index in self.unexplored_clients
            ):
                self.unexplored_clients.remove(client_index)

            sum_norms += norm_delta
            count += 1

        # Assign proxy to unexplored clients
        if count > 0 and self.unexplored_clients:
            avg_norm = sum_norms / count
            proxy = self.alpha * avg_norm
            self.last_proxy = proxy
            self.last_proxy_basis = avg_norm
            for idx in self.unexplored_clients:
                norm_deltas[idx] = proxy

        # Write back (skip zero entries — clients not in this batch and not proxied)
        for idx, bound in enumerate(norm_deltas):
            if bound != 0.0:
                local_gradient_bounds[idx] = bound

        polaris_state["local_gradient_bounds"] = local_gradient_bounds

        logging.debug(
            "[PolarisVi/Agg] G_i updated for %d client(s); "
            "G_i stats: min=%.4f max=%.4f mean=%.4f",
            count,
            float(np.min(local_gradient_bounds)),
            float(np.max(local_gradient_bounds)),
            float(np.mean(local_gradient_bounds)),
        )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class Server(fedasync_server.Server):
    """FedAsync server with Polaris G_i tracking and V_i client selection.

    Bypasses fedasync_server.Server.__init__ to inject custom strategies, but
    inherits configure(), _process_reports(), and _select_clients() from the
    fedasync_server.Server parent so all FedAsync-specific behaviour is kept.
    """

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        server_cfg = Config().server

        # --- Compute T_round once at startup -----------------------------------
        # Pull config values; fall back to safe defaults if not yet initialised.
        total_clients = int(getattr(Config().clients, "total_clients", 100))
        local_epochs = int(getattr(Config().trainer, "epochs", 1))

        # PLATO assigns equal partition_size to every client (set in
        # [data] partition_size in the TOML).  Non-uniform sizes would require
        # per-client accounting from the datasource; document that assumption
        # here so it is visible if partition sizes ever become non-uniform.
        partition_size = int(getattr(Config().data, "partition_size", 500))
        partition_sizes = {cid: partition_size for cid in range(1, total_clients + 1)}

        clients_per_round = int(getattr(Config().clients, "per_round", 10))
        client_ids = list(partition_sizes.keys())
        t_round_seconds = gpu_tier_sleep.compute_pool_T_round_seconds(
            client_ids, partition_sizes, local_epochs,
            clients_per_round=clients_per_round,
        )
        pool_mean_T_seconds = gpu_tier_sleep.compute_pool_mean_T_seconds(
            client_ids, partition_sizes, local_epochs
        )
        logging.info(
            "[PolarisVi] T_round = %.6f s  (E[min of %d concurrent clients],"
            " pool mean = %.6f s, %d total clients, epochs=%d)",
            t_round_seconds,
            clients_per_round,
            pool_mean_T_seconds,
            total_clients,
            local_epochs,
        )
        # -----------------------------------------------------------------------

        aggregation_strategy = FedAsyncPolarisAggregationStrategy(
            alpha=getattr(server_cfg, "polaris_alpha", 10.0),
            initial_gradient_bound=getattr(
                server_cfg, "polaris_initial_gradient_bound", 0.5
            ),
            initial_staleness=getattr(server_cfg, "polaris_initial_staleness", 0.01),
            t_round_seconds=t_round_seconds,
            partition_sizes=partition_sizes,
            local_epochs=local_epochs,
            pool_mean_T_seconds=pool_mean_T_seconds,
            clients_per_round=clients_per_round,
        )

        # Choose staleness predictor from [server] staleness_predictor in the TOML.
        # Omitting the key keeps the existing observed-τ behaviour, so old configs
        # are unaffected; an unrecognised value now RAISES (see
        # resolve_staleness_predictor).
        # "gpu_tops": uses GPU TFLOPS from tiers.csv for proactive staleness prediction.
        # "cpu": identical code path; tiers_csv in [gpu_tier] should point to
        #         tiers_cpu.csv whose gpu_tflops_fp16 column holds CPU-based throughput
        #         (cores × GHz / 1000) — lower heterogeneity than GPU tiers.
        #         Clients still sleep for the same durations as gpu_tops (same sleep CSV).
        # "observed": the client's own last measured staleness, in rounds — Polaris's
        #         reactive estimator.  Never computes T_round or a residual.
        # "gpu_oracle": identical code path to gpu_tops, but [gpu_tier]
        #         oracle_sleep_csv makes gpu_tier_sleep return each client's TRUE
        #         sleep — jitter included — instead of its tier's datasheet mean.
        #         Not deployable: it is the upper bound that says how much the
        #         spec-derived gpu arm gives up.  See gpu_tier_sleep.oracle_sleeps().
        predictor_name = resolve_staleness_predictor(
            getattr(server_cfg, "staleness_predictor", None)
        )
        check_oracle_wiring(predictor_name)
        if predictor_name in SECONDS_PREDICTORS:
            _local_epochs = local_epochs  # capture for closure
            def _gpu_tops_predictor(cid: int, ctx) -> float:
                sizes = ctx.state.get("partition_sizes", partition_sizes)
                return gpu_tier_sleep.gpu_tops_predict_seconds(
                    cid, sizes.get(cid, partition_size), _local_epochs
                )
            client_selection_strategy = fedasync_polaris_vi_selection.PolarisViClientSelectionStrategy(
                predict_staleness=_gpu_tops_predictor,
                predict_staleness_unit="seconds",
            )
            logging.info(
                "[PolarisVi] Staleness predictor: %s"
                " (proactive, T_round recomputed each round from live busy pool)",
                predictor_name,
            )
        else:  # "observed" — the only remaining value resolve_() can return
            client_selection_strategy = fedasync_polaris_vi_selection.PolarisViClientSelectionStrategy()
            logging.info("[PolarisVi] Staleness predictor: observed-τ (rounds, default)")

        logging.info("[PolarisVi] Server initialised with FedAsync+Polaris aggregation "
                     "and V_i client selection.")

        # Call fedavg.Server.__init__ directly to inject our strategies.
        # This bypasses fedasync_server.Server.__init__ (which hard-codes its
        # own aggregation and selection), while still inheriting its FedAsync
        # overrides for configure(), _process_reports(), and _select_clients().
        fedavg.Server.__init__(
            self,
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=aggregation_strategy,
            client_selection_strategy=client_selection_strategy,
        )
