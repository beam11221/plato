"""
Selection rule D1 (Prompt C, 16 Oct 2026): epsilon-greedy on V_i.

One class, `WenSelector`, for EVERY arm, so that eligibility, slot counting and
the RNG are identical across arms:

    random        egreedy, epsilon = 1     (no scorer; every pick explores)
    gpu, cpu, gpu_oracle, observed, location_only
                  egreedy, epsilon = 0.2   (default)
    gpu ablations egreedy, epsilon = 0     (pure top-1)
                  proportional             (the previous rule: sample
                                            proportional to V_i, no replacement --
                                            delegated unchanged to the Polaris
                                            selector)

Per pick (one decision per free slot):
    u ~ U(0,1) from default_rng([seed, round, k, SALT])
    u < epsilon  -> explore: uniform over the candidates (second draw, same stream)
    otherwise    -> exploit: highest V_i, ties to the lower client_id

`k` counts decisions already taken under the same round number, the same
convention as the fading key (wen_truth.py): a round held by a lost S2 update
does not replay the draw.

Scores come from the Polaris selector's `score_candidates` once per selection
call; when one call fills several slots (the first cycle, or a refill after a
lost slot), the later picks use the same snapshot minus the clients already
picked.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext

SELECTION_RULES = ("egreedy", "proportional")
DEFAULT_EPSILON = 0.2
_SALT = 0xE6EED


@dataclass(frozen=True)
class PickRecord:
    selection_rule: str
    epsilon: float | None
    coin_u: float | None
    explored: bool | None
    n_candidates: int
    rank_of_pick: int | None
    V_max: float | None
    V_min: float | None
    V_pick: float | None


def egreedy_pick(candidates: list[int], scores: dict[int, float] | None,
                 epsilon: float, rng: np.random.Generator) -> tuple[int, PickRecord]:
    """One decision. Pure: candidates, scores and an RNG in, a pick out."""
    if not candidates:
        raise ValueError("no candidates")
    ordered = sorted(candidates)
    u = float(rng.random())
    explored = u < epsilon
    if explored:
        pick = ordered[int(rng.integers(len(ordered)))]
    else:
        if scores is None:
            raise ValueError("an exploit pick needs scores (epsilon < 1 without a scorer)")
        pick = min(ordered, key=lambda c: (-scores[c], c))

    if scores is not None:
        ranking = sorted(ordered, key=lambda c: (-scores[c], c))
        values = [scores[c] for c in ordered]
        record = PickRecord("egreedy", epsilon, u, explored, len(ordered),
                            ranking.index(pick) + 1, max(values), min(values),
                            scores[pick])
    else:
        record = PickRecord("egreedy", epsilon, u, explored, len(ordered),
                            None, None, None, None)
    return pick, record


class WenSelector(ClientSelectionStrategy):
    """The arm's selection strategy. `scorer` is the Polaris selector (or None).

    Exposes the scorer as `.inner`, so the server's lookup for the Polaris
    selector (predictor wiring, tau_hat logging) still finds it.
    """

    def __init__(self, scorer, rule: str, epsilon: float | None, seed: int) -> None:
        if rule not in SELECTION_RULES:
            raise ValueError(f"selection rule {rule!r} not in {SELECTION_RULES}")
        if rule == "egreedy" and not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError(f"epsilon {epsilon} outside [0, 1]")
        if rule == "egreedy" and scorer is None and float(epsilon) != 1.0:
            raise ValueError("without a scorer only epsilon = 1 (random) is defined")
        if rule == "proportional" and scorer is None:
            raise ValueError("proportional needs the Polaris scorer")
        self.inner = scorer
        self.rule = rule
        self.epsilon = None if rule == "proportional" else float(epsilon)
        self.seed = seed
        self._decisions_per_round: dict[int, int] = defaultdict(int)
        self.last_picks: dict[int, PickRecord] = {}

    # -- lifecycle: delegate to the scorer when there is one -----------------

    def setup(self, context: ServerContext) -> None:
        if self.inner is not None:
            self.inner.setup(context)

    def teardown(self, context: ServerContext) -> None:
        if self.inner is not None:
            self.inner.teardown(context)

    def on_clients_selected(self, selected_clients, context: ServerContext) -> None:
        if self.inner is not None:
            self.inner.on_clients_selected(selected_clients, context)

    def on_reports_received(self, updates, context: ServerContext) -> None:
        if self.inner is not None:
            self.inner.on_reports_received(updates, context)

    # -- selection -----------------------------------------------------------

    def _rng(self, round_no: int) -> np.random.Generator:
        k = self._decisions_per_round[round_no]
        self._decisions_per_round[round_no] += 1
        return np.random.default_rng([self.seed, round_no, k, _SALT])

    def select_clients(self, clients_pool, clients_count, context):
        self.last_picks = {}
        server = context.server
        busy = set(getattr(server, "training_clients", {}) or {})
        candidates = sorted(c for c in clients_pool if c not in busy)
        count = min(clients_count, len(candidates))
        if count == 0:
            return []

        if self.rule == "proportional":
            selected = self.inner.select_clients(candidates, count, context)
            for cid in selected:
                self.last_picks[cid] = PickRecord("proportional", None, None, None,
                                                  len(candidates), None, None, None, None)
            return selected

        scores = None
        if self.inner is not None:
            scores, tau_hats, _G, _total = self.inner.score_candidates(candidates, context)
        round_no = int(getattr(server, "current_round", 0))
        selected: list[int] = []
        remaining = list(candidates)
        for _ in range(count):
            pick, record = egreedy_pick(remaining, scores, self.epsilon,
                                        self._rng(round_no))
            selected.append(pick)
            remaining.remove(pick)
            self.last_picks[pick] = record

        if self.inner is not None:
            # Keep the Polaris prediction-accuracy log working, as select_clients
            # would have.
            predictions = context.state.setdefault("polaris", {}).setdefault(
                "staleness_predictions", {})
            for cid in selected:
                predictions[cid] = tau_hats[cid]
        return selected
