"""
Wen-faithful selector, async adaptation (Prompt E, decisions W1-W10).

Wen et al., Information Sciences 664 (2024), RA-MaOVSM, picking one vehicle per
free FedAsync slot. Four objectives per idle eligible candidate k, all minimised:

    F1 = T_com + T_up      Wen's CPU time model: b |S_k| rho / f, rho = 1,250, plus the
                           upload time at Wen's predicted upload position
    F2 = E_com + E_up      Wen's CPU energy model: e rho b |S_k| f^2 + p T_up
    F3 = last reported training loss (never-trained -> mean of all losses reported)
    F4 = KL(P_k || uniform 0.1), P_k = the client's label histogram / |S_k|

F1 and F2 come from `CpuEstimator` (|chi|^2 = 1, wrap via wen_formulas.wrap_x).

Selection (W3, W5), deterministic, no RNG:
    1. feasible = F1 <= T_max and F2 <= E_max; none feasible -> use all (fallback)
    2. exact non-dominated set of the feasible candidates
    3. min-max normalise each objective within that set (range 0 -> 0)
    4. pick the smallest mean of the four normalised values; ties -> lower client_id

The functions below are pure; `WenFaithfulSelection` only wires them to the server
and writes two side logs (one row per decision, one row per scored candidate).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sumo_gates import delivered
from wen_estimators import CpuEstimator

from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext

UNIFORM_Q = 0.1
TIERS = ("xavier", "orin", "thor")


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def kl_to_uniform(counts) -> float:
    """KL(P || uniform 0.1) with P = counts / sum; 0 ln 0 = 0, no smoothing."""
    counts = np.asarray(counts, dtype=float)
    p = counts / counts.sum()
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / UNIFORM_Q)))


def dominates(a, b) -> bool:
    """a dominates b: no worse in every objective, strictly better in one."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def non_dominated(points: dict[int, tuple]) -> list[int]:
    """Client ids whose objective vector no other candidate dominates (sorted)."""
    return sorted(c for c, p in points.items()
                  if not any(dominates(q, p) for d, q in points.items() if d != c))


def normalise(points: dict[int, tuple]) -> dict[int, tuple]:
    """Min-max per objective within `points`; an objective with range 0 -> 0."""
    ids = sorted(points)
    cols = list(zip(*(points[c] for c in ids)))
    out_cols = []
    for col in cols:
        lo, hi = min(col), max(col)
        out_cols.append([0.0 if hi == lo else (v - lo) / (hi - lo) for v in col])
    return {c: tuple(col[i] for col in out_cols) for i, c in enumerate(ids)}


@dataclass(frozen=True)
class Decision:
    pick: int
    feasible: list[int]
    fallback: bool
    pareto: list[int]
    normalised: dict[int, tuple]
    score: dict[int, float]


def decide(objectives: dict[int, tuple], t_max: float, e_max: float) -> Decision:
    """One Wen decision over {client_id: (F1, F2, F3, F4)}."""
    feasible = sorted(c for c, f in objectives.items() if f[0] <= t_max and f[1] <= e_max)
    fallback = not feasible
    pool = {c: objectives[c] for c in (sorted(objectives) if fallback else feasible)}
    pareto = non_dominated(pool)
    norm = normalise({c: pool[c] for c in pareto})     # one member -> all zeros
    score = {c: sum(v) / len(v) for c, v in norm.items()}
    pick = min(pareto, key=lambda c: (score[c], c))
    return Decision(pick, feasible, fallback, pareto, norm, score)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

DECISION_FIELDS = (
    ["round", "t", "pick", "pick_tier", "n_candidates", "n_feasible",
     "n_cut_T_only", "n_cut_E_only", "n_cut_both"]
    + [f"n_cut_{k}_{t}" for k in ("T_only", "E_only", "both") for t in TIERS]
    + [f"n_candidates_{t}" for t in TIERS]
    + ["fallback", "n_pareto", "pick_rank_in_pareto",
       "F1", "F2", "F3", "F4", "nF1", "nF2", "nF3", "nF4", "x_hat_upload"]
)
CANDIDATE_FIELDS = ["round", "t", "decision", "client_id", "tier", "x",
                    "F1", "F2", "F3", "F4", "feasible", "pareto", "picked"]


class WenFaithfulSelection(ClientSelectionStrategy):
    decision_fields = DECISION_FIELDS
    candidate_fields = CANDIDATE_FIELDS

    def __init__(self, estimator: CpuEstimator, t_max: float, e_max: float,
                 kl: dict[int, float], out_dir: Path) -> None:
        self.estimator = estimator
        self.t_max, self.e_max = t_max, e_max
        self.kl = kl
        self.last_loss: dict[int, float] = {}
        self._loss_sum, self._loss_n = 0.0, 0
        self.last_picks: dict = {}          # the dispatch log's D1 columns stay blank
        self._decisions = 0
        out_dir.mkdir(parents=True, exist_ok=True)
        self._dec_fh = (out_dir / "wen_decisions.csv").open("w", newline="")
        self._dec = csv.DictWriter(self._dec_fh, fieldnames=self.decision_fields)
        self._dec.writeheader()
        self._cand_fh = (out_dir / "wen_candidates.csv").open("w", newline="")
        self._cand = csv.DictWriter(self._cand_fh, fieldnames=self.candidate_fields)
        self._cand.writeheader()

    # -- F3: what the RSU has actually received ------------------------------

    def on_reports_received(self, updates, context: ServerContext) -> None:
        for u in delivered(updates, context):          # lost S2 uploads teach nothing
            loss = getattr(u.report, "train_loss", None)
            if loss is not None and math.isfinite(loss):
                self.last_loss[u.client_id] = float(loss)
                self._loss_sum += float(loss)
                self._loss_n += 1

    def f3(self, client_id: int) -> float:
        if client_id in self.last_loss:
            return self.last_loss[client_id]
        return self._loss_sum / self._loss_n if self._loss_n else 0.0

    # -- selection ---------------------------------------------------------------

    def objectives(self, server, candidates, t):
        out, x_hat, tier, xs = {}, {}, {}, {}
        for c in candidates:
            rv = server.wen_report(c, t)
            e = self.estimator.estimate(rv)
            out[c] = (e.total, e.E, self.f3(c), self.kl[c])
            x_hat[c], tier[c], xs[c] = e.x_finish, rv.tier, rv.x
        return out, x_hat, tier, xs

    def select_clients(self, clients_pool, clients_count, context):
        server = context.server
        busy = set(getattr(server, "training_clients", {}) or {})
        remaining = sorted(c for c in clients_pool if c not in busy)
        t = server.sumo_time()
        round_no = int(getattr(server, "current_round", 0))
        picks = []
        for _ in range(min(clients_count, len(remaining))):
            obj, x_hat, tier, xs = self.objectives(server, remaining, t)
            d = decide(obj, self.t_max, self.e_max)
            self._log(round_no, t, obj, d, x_hat, tier, xs)
            picks.append(d.pick)
            remaining.remove(d.pick)
        return picks

    def _log(self, round_no, t, obj, d: Decision, x_hat, tier, xs) -> None:
        self._decisions += 1
        cut_t = {c for c, f in obj.items() if f[0] > self.t_max}
        cut_e = {c for c, f in obj.items() if f[1] > self.e_max}
        row = {"round": round_no, "t": repr(t), "pick": d.pick, "pick_tier": tier[d.pick],
               "n_candidates": len(obj), "n_feasible": len(d.feasible),
               "n_cut_T_only": len(cut_t - cut_e), "n_cut_E_only": len(cut_e - cut_t),
               "n_cut_both": len(cut_t & cut_e),
               "fallback": int(d.fallback), "n_pareto": len(d.pareto),
               "pick_rank_in_pareto": 1 + sum(s < d.score[d.pick] for s in d.score.values()),
               "x_hat_upload": repr(x_hat[d.pick])}
        for name, group in (("T_only", cut_t - cut_e), ("E_only", cut_e - cut_t),
                            ("both", cut_t & cut_e)):
            for tr in TIERS:
                row[f"n_cut_{name}_{tr}"] = sum(tier[c] == tr for c in group)
        for tr in TIERS:
            row[f"n_candidates_{tr}"] = sum(tier[c] == tr for c in obj)
        for i in range(4):
            row[f"F{i + 1}"] = repr(obj[d.pick][i])
            row[f"nF{i + 1}"] = repr(d.normalised[d.pick][i])
        row.update(self.log_extra())
        self._dec.writerow(row)
        self._dec_fh.flush()
        pareto = set(d.pareto)
        feasible = set(d.feasible)
        for c, f in obj.items():
            self._cand.writerow({
                "round": round_no, "t": repr(t), "decision": self._decisions, "client_id": c,
                "tier": tier[c], "x": repr(xs[c]), "F1": repr(f[0]), "F2": repr(f[1]),
                "F3": repr(f[2]), "F4": repr(f[3]), "feasible": int(c in feasible),
                "pareto": int(c in pareto), "picked": int(c == d.pick),
                **{k: v for k, v in self.log_extra().items() if k in self.candidate_fields}})
        self._cand_fh.flush()

    def log_extra(self) -> dict:
        """Extra columns a subclass adds to both side logs (none for `wen`)."""
        return {}


def label_kl(targets, data_seed: int) -> dict[int, float]:
    """F4 per client, from the label histogram each client reports at registration.

    The histogram is the client's own partition (wen_partition), the same split
    every client process trains on.
    """
    import wen_partition

    part = wen_partition.build(targets, data_seed)
    return {cid: kl_to_uniform(part.counts[i]) for i, cid in enumerate(sorted(part.indices))}


def cifar10_train_targets(data_path: Path) -> list[int]:
    """The 50,000 CIFAR-10 training labels, read from the torchvision batch files."""
    import pickle

    labels: list[int] = []
    for i in range(1, 6):
        with (Path(data_path) / "cifar-10-batches-py" / f"data_batch_{i}").open("rb") as fh:
            labels += pickle.load(fh, encoding="latin1")["labels"]
    return labels
