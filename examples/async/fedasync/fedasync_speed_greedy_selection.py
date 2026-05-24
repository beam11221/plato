"""
Speed-greedy client selection strategy for FedAsync.

Selects the top-fastest available clients from a configurable set of GPU tiers,
skipping clients still training and capping at the number of free process slots
(same slot-cap semantics as FedAsyncClientSelectionStrategy).

Configuration (TOML)
--------------------
[server.client_selection]
strategy      = "speed_greedy"
allowed_tiers = ["orin", "thor"]   # any subset of tier_ids in tiers.csv

Defaults to {"orin", "thor"} when the section is absent.

Selection algorithm
-------------------
1. Remove clients currently in ``server.training_clients`` (busy filter).
2. Remove clients whose tier_id is not in ``allowed_tiers``.
3. Sort survivors by ``gpu_tflops_fp16`` descending (fastest hardware first).
4. Take the top min(clients_count, open_slots, eligible) clients — deterministic,
   no random sampling.
"""

from __future__ import annotations

import logging

import gpu_tier_sleep

from fedasync_selection import FedAsyncClientSelectionStrategy
from plato.servers.strategies.base import ServerContext


class SpeedGreedyClientSelectionStrategy(FedAsyncClientSelectionStrategy):
    """Deterministic top-speed selection restricted to configured GPU tiers."""

    _DEFAULT_TIERS: frozenset[str] = frozenset({"orin", "thor"})

    def _allowed_tiers(self) -> frozenset[str]:
        """Return the set of allowed tier_ids from TOML config, or the default."""
        try:
            from plato.config import Config  # noqa: PLC0415 — lazy import
            sel_cfg = getattr(Config().server, "client_selection", None)
            if sel_cfg is not None:
                raw = getattr(sel_cfg, "allowed_tiers", None)
                if raw:
                    return frozenset(str(t).lower() for t in raw)
        except Exception:  # noqa: BLE001
            pass
        return self._DEFAULT_TIERS

    def select_clients(
        self,
        clients_pool: list[int],
        clients_count: int,
        context: ServerContext,
    ) -> list[int]:
        server = context.server
        busy = set(getattr(server, "training_clients", {}) or {})
        clients_per_round = getattr(server, "clients_per_round", clients_count)
        open_slots = max(0, clients_per_round - len(busy))

        allowed = self._allowed_tiers()

        eligible: list[int] = []
        for cid in clients_pool:
            if cid in busy:
                continue
            try:
                if gpu_tier_sleep.tier_for(cid).tier_id.lower() in allowed:
                    eligible.append(cid)
            except KeyError:
                pass  # client not in clients.csv — skip silently

        effective_count = min(clients_count, open_slots, len(eligible))

        if effective_count == 0:
            logging.info(
                "[SpeedGreedy] No eligible clients "
                "(allowed_tiers=%s, busy=%d, open_slots=%d).",
                sorted(allowed), len(busy), open_slots,
            )
            return []

        if effective_count < clients_count:
            logging.info(
                "[SpeedGreedy] %d busy, %d open slot(s), %d eligible in %s; "
                "selecting %d of %d requested.",
                len(busy), open_slots, len(eligible),
                sorted(allowed), effective_count, clients_count,
            )

        def _tflops(cid: int) -> float:
            try:
                return gpu_tier_sleep.tier_for(cid).gpu_tflops_fp16
            except KeyError:
                return 0.0

        selected = sorted(eligible, key=_tflops, reverse=True)[:effective_count]
        logging.info("[SpeedGreedy] Selected clients: %s", selected)
        return selected
