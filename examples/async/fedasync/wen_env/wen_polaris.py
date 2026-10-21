"""
`polaris` (Prompt J): Kang & Li, IEEE TCC 12(2), 2024, as a host in S1.

Fidelity rule: the paper first; where it is silent, the authors' Plato code
(examples/client_selection/polaris/); where the code contradicts the paper, the
paper. Decisions PL1-PL8 and the code-vs-paper differences are in ADR-0020.

Per selection call, over the eligible pool P (S1: idle vehicles), problem P4
(Eq. 23):

    min_q  sum_i f1_i / q_i + sum_i f2_i q_i    s.t. sum_i q_i = 1, q_i > 0
    f1_i = p_i^2 G_i^2,   f2_i = A p_i^2 tau_i G_i   (tau linear, as in the paper)

KKT: q_i(mu) = sqrt(f1_i / (f2_i + mu)), mu > -min f2; sum q is strictly
decreasing in mu, so mu is found by bisection. No GP solver is needed.

Inputs, all observed by the server:
    p_i    n_i / sum over the 30 clients of n_j (reported partition sizes)
    G_i    context.state["polaris"]["local_gradient_bounds"], written by
           FedAsyncPolarisAggregationStrategy (conv-layer update norm; unseen
           clients alpha x the latest aggregation's mean norm; initial 0.5)
    tau_i  staleness of the client's last merged update + 0.1; initial 0.01

One draw per free slot from q renormalised over the candidates not yet picked
(PL5, PL8), with the RNG keyed by (seed, round, k, salt) as in wen_egreedy.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import wen_constants as K
from sumo_gates import delivered

from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext

A_CONST = K.POLARIS_A
TAU_OFFSET = K.POLARIS_TAU_OFFSET
INITIAL_TAU = K.POLARIS_INITIAL_TAU
_SALT = 0x90A715
BISECTION_TOL = 1e-12
BISECTION_MAX_ITERS = 400

DECISION_FIELDS = ["round", "t", "pool", "G", "tau", "p", "f1", "f2", "mu", "q",
                   "max_q", "eff_pool", "n_zero_G", "picks", "q_picks"]


def solve_p4(G, tau, p, A: float = A_CONST) -> tuple[np.ndarray, float]:
    """(q, mu) for P4 over one pool. G_i = 0 -> q_i = 0."""
    G, tau, p = (np.asarray(a, dtype=float) for a in (G, tau, p))
    f1 = p * p * G * G
    f2 = A * p * p * tau * G
    pos = f1 > 0
    if not pos.any():
        raise ValueError("P4: every candidate has G = 0")
    q = np.zeros(len(G))
    if np.all(f1[pos] == f1[pos][0]) and np.all(f2[pos] == f2[pos][0]):
        q[pos] = 1.0 / pos.sum()                    # identical states -> uniform
        return q, float(f1[pos][0] * pos.sum() ** 2 - f2[pos][0])

    f1p, f2p = f1[pos], f2[pos]

    def total(mu):
        return float(np.sqrt(f1p / (f2p + mu)).sum())

    lo = -float(f2p.min())                          # sum q -> inf as mu -> lo+
    hi = max(lo, 0.0) + 1.0
    while total(hi) > 1.0:
        hi = lo + 2.0 * (hi - lo)
    mu = 0.5 * (lo + hi)
    for _ in range(BISECTION_MAX_ITERS):
        mu = 0.5 * (lo + hi)
        s = total(mu)
        if abs(s - 1.0) < BISECTION_TOL or not lo < mu < hi:
            break
        lo, hi = (mu, hi) if s > 1.0 else (lo, mu)
    qp = np.sqrt(f1p / (f2p + mu))
    q[pos] = qp / qp.sum()
    return q, mu


class PolarisSelection(ClientSelectionStrategy):
    """The `polaris` host, in the WenSelector position (wen_server.install_wen)."""

    def __init__(self, seed: int, out_dir: Path, A: float = A_CONST) -> None:
        self.seed = seed
        self.A = A
        ids = range(1, K.N_CLIENTS + 1)
        sizes = np.array([K.partition_size(c) for c in ids], dtype=float)
        self.p = sizes / sizes.sum()                  # PL3: fleet-static
        self.tau = np.full(K.N_CLIENTS, INITIAL_TAU)
        self._decisions_per_round: dict[int, int] = defaultdict(int)
        self.last_picks: dict = {}                    # the D1 columns stay blank
        self.last_q: dict[int, float] = {}            # dispatch log `score` (PL7)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._fh = (out_dir / "polaris_decisions.csv").open("w", newline="")
        self._log = csv.DictWriter(self._fh, fieldnames=DECISION_FIELDS)
        self._log.writeheader()

    def on_reports_received(self, updates, context: ServerContext) -> None:
        for u in delivered(updates, context):
            self.tau[u.client_id - 1] = float(getattr(u, "staleness", 0.0)) + TAU_OFFSET

    def _rng(self, round_no: int) -> np.random.Generator:
        k = self._decisions_per_round[round_no]
        self._decisions_per_round[round_no] += 1
        return np.random.default_rng([self.seed, round_no, k, _SALT])

    def select_clients(self, clients_pool, clients_count, context):
        self.last_q = {}
        server = context.server
        busy = set(getattr(server, "training_clients", {}) or {})
        pool = sorted(c for c in clients_pool if c not in busy)
        count = min(clients_count, len(pool))
        if count == 0:
            return []

        G_all = context.state["polaris"]["local_gradient_bounds"]
        idx = [c - 1 for c in pool]
        G, tau, p = np.asarray(G_all, dtype=float)[idx], self.tau[idx], self.p[idx]
        q, mu = solve_p4(G, tau, p, self.A)
        self.last_q = {c: float(q[j]) for j, c in enumerate(pool)}

        round_no = int(getattr(server, "current_round", 0))
        remaining = list(pool)
        picks = []
        for _ in range(count):
            w = np.array([self.last_q[c] for c in remaining])
            pick = remaining[int(self._rng(round_no).choice(len(remaining), p=w / w.sum()))]
            picks.append(pick)
            remaining.remove(pick)

        f1 = p * p * G * G
        f2 = self.A * p * p * tau * G
        self._log.writerow({
            "round": round_no, "t": repr(server.sumo_time()), "pool": json.dumps(pool),
            "G": json.dumps(G.tolist()), "tau": json.dumps(tau.tolist()),
            "p": json.dumps(p.tolist()), "f1": json.dumps(f1.tolist()),
            "f2": json.dumps(f2.tolist()), "mu": repr(mu), "q": json.dumps(q.tolist()),
            "max_q": repr(float(q.max())), "eff_pool": repr(float(1.0 / np.sum(q * q))),
            "n_zero_G": int(np.sum(G == 0)), "picks": json.dumps(picks),
            "q_picks": json.dumps([self.last_q[c] for c in picks]),
        })
        self._fh.flush()
        return picks
