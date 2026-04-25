"""
A federated learning server using FedAsync.

Reference:

Xie, C., Koyejo, S., Gupta, I. "Asynchronous federated optimization,"
in Proc. 12th Annual Workshop on Optimization for Machine Learning (OPT 2020).

https://opt-ml.org/papers/2020/paper_28.pdf
"""

import logging

# --- FedAsync example fixes (see companion modules for details) ------------
# Two new sibling modules drive the bug fixes and the batched-update
# workaround. They are imported here so this server can wire them in.
#   * fedasync_selection: RandomSelectionStrategy subclass that filters out
#       clients still in `self.training_clients`. Fixes the KeyError in
#       process_client_info (plato/servers/base.py:1025) caused by the same
#       client being re-selected while its prior round's report was still in
#       flight.
#   * fedasync_aggregation: FedAsyncAggregationStrategy subclass that walks
#       the full `updates` list in arrival order instead of keeping only
#       updates[0] and discarding the rest. See the "Issue 2 - batched
#       updates" notes for semantics.
import fedasync_aggregation
import fedasync_selection

from plato.config import Config
from plato.servers import fedavg

# Original import kept for reference; we now instantiate the subclass from
# fedasync_aggregation instead:
# from plato.servers.strategies import FedAsyncAggregationStrategy


class Server(fedavg.Server):
    """A federated learning server using the FedAsync algorithm."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        # Previous wiring used the stock FedAsyncAggregationStrategy, which
        # keeps only `updates[0]` and logs "discarding N other update(s)".
        # That discarded most client work whenever several reports arrived in
        # the same periodic tick.
        # aggregation_strategy = FedAsyncAggregationStrategy()
        aggregation_strategy = fedasync_aggregation.SequentialFedAsyncAggregationStrategy()

        # New: custom selection strategy skips clients still training. Without
        # this, in `simulate_wall_time=false` mode the base.py filter at
        # base.py:557-595 is bypassed and busy clients can be re-selected.
        client_selection_strategy = fedasync_selection.FedAsyncClientSelectionStrategy()

        logging.info("FedAsync Server: Initialized with FedAsync aggregation strategy.")
        logging.info(f"model: {model}, datasource: {datasource}, algorithm: {algorithm}, trainer: {trainer}, callbacks: {callbacks}")
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=aggregation_strategy,
            # Newly passed through so base.py uses our filtering selection.
            client_selection_strategy=client_selection_strategy,
        )
        logging.info(f"[After super] model: {self.model}, datasource: {self.datasource}, algorithm: {self.algorithm}, trainer: {self.trainer}, callbacks: {self.callbacks}")

    def configure(self) -> None:
        """Configure the mixing hyperparameter for the server, as well as
        other parameters from the configuration file.
        """
        super().configure()

        if not hasattr(Config().server, "mixing_hyperparameter"):
            logging.warning(
                "FedAsync: Variable mixing hyperparameter is required for the FedAsync server."
            )
        else:
            try:
                mixing_hyperparam = float(Config().server.mixing_hyperparameter)
            except (TypeError, ValueError):
                logging.warning(
                    "FedAsync: Invalid mixing hyperparameter. "
                    "Unable to cast %s to float.",
                    Config().server.mixing_hyperparameter,
                )
                return

            if 0 < mixing_hyperparam < 1:
                logging.info(
                    "FedAsync: Mixing hyperparameter is set to %s.",
                    mixing_hyperparam,
                )
            else:
                logging.warning(
                    "FedAsync: Invalid mixing hyperparameter. "
                    "The hyperparameter needs to be between 0 and 1 (exclusive)."
                )

    async def _process_reports(self):
        """Aggregate weights every round, but only run the expensive server-side
        model test every `server.eval_interval` rounds (default: every round)."""
        eval_interval = getattr(Config().server, "eval_interval", 1)
        if self.current_round % eval_interval != 0:
            logging.info(
                "[FedAsync] Round %d: skipping evaluation (next at round %d).",
                self.current_round,
                self.current_round + (eval_interval - self.current_round % eval_interval),
            )
            trainer = self.require_trainer()
            _orig_test = trainer.test
            # Return last known accuracy so logging/callbacks still work
            trainer.test = lambda *a, **kw: self.accuracy
            try:
                await super()._process_reports()
            finally:
                trainer.test = _orig_test
        else:
            await super()._process_reports()

    # NOTE: the original fedasync example wrapped _select_clients with a
    # defensive guard that raised RuntimeError if the previously-selected
    # `clients_per_round` exceeded the number of free client processes. That
    # guard was over-eager and was reading stale `selected_clients` from the
    # prior round before `super()._select_clients` could overwrite it. It
    # broke the documented `simulate_wall_time=true` mode where Plato is
    # meant to chunk a large `per_round` through a smaller pool of physical
    # processes (see plato/servers/base.py:642-648, which slices
    # `selected_clients[len(trained):len(trained)+len(clients)]` for the
    # single-GPU case). We rely on that built-in chunking instead.
