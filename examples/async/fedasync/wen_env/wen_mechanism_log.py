"""
Mechanism logging (Prompt M, Part A1): one row per candidate per decision.

Server-side and behaviour-neutral: it reads the selector's snapshot of the
call it has just made, the server's reported records and estimates, and its own
passive trackers. It draws no random number, writes nothing a selector reads,
and no selector imports it (tests/test_wen_separation.py).

Files, in the run's base_path:

    <arm>_candidates.csv   our score (`gpu` and the other egreedy-with-scorer arms):
                           common + passive + T_hat / tau_hat / G / V / rank / coin
    polaris_candidates.csv the polaris host: common + passive + P4 inputs, q, draw
    passive_candidates.csv every other arm (random, wen, wen_fresh): common + passive;
                           Wen's own columns are in wen_candidates.csv, joined on
                           decision_id == wen_candidates.decision
    gpu_g_updates.csv      (`gpu`) / passive_g_updates.csv (others): one row per
                           merged update of the passive G tracker

Passive common axes, the same for every arm: what Ours' scorer WOULD see --
tau_hat_ours from the gpu estimator's T_hat_total through the queue kernel
(startup T_round, live T_round from the busy set's T_hat frozen at dispatch, the
two-phase residual count), and G_ours from a tracker with Ours' formula
(conv-layer update norm against the global model at merge, unseen clients
alpha = 1 x the batch's mean norm, initial 0.5). In the `gpu` arm both equal the
arm's own values; tests/test_wen_mechanism_log.py checks it.

Ground truth stays in dispatch_log.csv; nothing here reads it.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import gpu_tier_sleep
import numpy as np
import torch
import wen_constants as K
import wen_formulas as F
from sumo_gates import UploadGate, _uploaded
from wen_estimators import GpuEstimator

PASSIVE_ALPHA = 1.0
PASSIVE_INITIAL_G = 0.5

COMMON_FIELDS = ["run", "arm", "seed", "round", "decision_id", "slot_k", "client",
                 "tier", "x_dispatch", "V_k", "picked"]
PASSIVE_FIELDS = ["tau_hat_ours", "T_hat_train_gpu", "T_hat_up_gpu", "x_hat_finish_gpu",
                  "T_hat_total_gpu", "G_ours", "V_ours", "n_merged_client"]
OURS_FIELDS = ["T_hat_train", "T_hat_up", "x_hat_finish", "d_hat", "T_hat_total",
               "T_round_used", "tau_hat", "G", "G_source", "V", "rank_by_V",
               "coin_u", "explored", "is_max_G", "is_min_tau_hat"]
POLARIS_FIELDS = ["G_i", "G_source", "tau_i", "p_i", "f1_i", "f2_i", "mu", "q_i",
                  "rank_by_q", "q_renormalised_at_pick", "draw_u"]
G_UPDATE_FIELDS = ["round", "client", "update_norm_conv", "G_new", "staleness",
                   "proxy"]


def _r(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return repr(value)
    return "" if value is None else value


class PassiveG:
    """Ours' G_i tracker (FedAsyncPolarisAggregationStrategy._update_gradient_bounds,
    the start_weights=None path), fed from every merged batch, alpha = 1."""

    def __init__(self, n: int = K.N_CLIENTS, alpha: float = PASSIVE_ALPHA) -> None:
        self.alpha = alpha
        self.G = np.full(n, PASSIVE_INITIAL_G, dtype=float)
        self.unexplored = set(range(n))
        self.merged = np.zeros(n, dtype=int)

    def update(self, updates, baseline_weights, weights_received) -> list[dict]:
        norms = np.zeros(len(self.G), dtype=float)
        rows, total, count = [], 0.0, 0
        for update, w_client in zip(updates, weights_received):
            idx = update.client_id - 1
            squared = 0.0
            for name, w_val in w_client.items():
                if "conv" not in name or name not in baseline_weights:
                    continue
                delta = w_val - baseline_weights[name]
                if isinstance(delta, torch.Tensor):
                    delta_np = delta.detach().cpu().numpy()
                else:
                    delta_np = np.asarray(delta, dtype=float)
                squared += float(np.sum(np.square(delta_np)))
            norm = float(np.sqrt(max(squared, 0.0)))
            norms[idx] = norm
            self.unexplored.discard(idx)
            self.merged[idx] += 1
            total += norm
            count += 1
            rows.append({"client": update.client_id, "update_norm_conv": norm,
                         "staleness": getattr(update, "staleness", ""), "proxy": 0})
        if count and self.unexplored:
            proxy = self.alpha * total / count
            for idx in self.unexplored:
                norms[idx] = proxy
        for idx, bound in enumerate(norms):
            if bound != 0.0:
                self.G[idx] = bound
        for row in rows:
            row["G_new"] = float(self.G[row["client"] - 1])
        return rows


class ObservedUploadGate(UploadGate):
    """UploadGate that also hands each delivered batch to the passive G tracker,
    after the real aggregation, with the same objects. `.inner` is unchanged."""

    def __init__(self, inner, on_merge) -> None:
        super().__init__(inner)
        self._on_merge = on_merge

    async def aggregate_weights(self, updates, baseline_weights, weights_received, context):
        result = await super().aggregate_weights(updates, baseline_weights,
                                                 weights_received, context)
        delivered = [(u, w) for u, w in zip(updates, weights_received)
                     if _uploaded(u, context)]
        if delivered:
            self._on_merge([u for u, _ in delivered], baseline_weights,
                           [w for _, w in delivered], context)
        return result


class MechanismLogger:
    def __init__(self, server, out_dir: Path, arm: str, host: str, seed: int,
                 run_id: str, model_bits: float) -> None:
        self.server = server
        self.arm, self.host, self.seed, self.run_id = arm, host, seed, run_id
        self.estimator = GpuEstimator(K.FLOPS_FWD_PER_IMAGE, model_bits)
        self.g = PassiveG()
        self._frozen: dict[int, float] = {}
        self._startup_t_round: float | None = None
        self._decision = 0
        out_dir = Path(out_dir)
        selector = server.wen_selector
        if host == "polaris":
            self.layout, name, extra = "polaris", "polaris_candidates.csv", POLARIS_FIELDS
        elif getattr(selector, "inner", None) is not None and \
                hasattr(selector, "last_scores") and server.wen_estimator is not None:
            self.layout, name, extra = "ours", f"{arm}_candidates.csv", OURS_FIELDS
        else:
            self.layout, name, extra = "passive", "passive_candidates.csv", []
        self._fh = (out_dir / name).open("w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=COMMON_FIELDS + PASSIVE_FIELDS + extra)
        self._w.writeheader()
        g_name = "gpu_g_updates.csv" if arm == "gpu" else "passive_g_updates.csv"
        self._gfh = (out_dir / g_name).open("w", newline="")
        self._gw = csv.DictWriter(self._gfh, fieldnames=G_UPDATE_FIELDS)
        self._gw.writeheader()

    # -- passive trackers ------------------------------------------------------

    def on_merge(self, updates, baseline_weights, weights_received, context) -> None:
        round_no = int(getattr(context, "current_round", self.server.current_round))
        for row in self.g.update(updates, baseline_weights, weights_received):
            self._gw.writerow({k: _r(v) for k, v in {"round": round_no, **row}.items()})
        self._gfh.flush()

    def _passive_estimate(self, cid: int, t: float):
        return self.estimator.estimate(self.server.wen_report(cid, t))

    def _passive_tau(self, pool, t) -> dict[int, float]:
        """tau_hat (rounds) for every candidate, the gpu arm's computation
        (fedasync_polaris_vi_selection.score_candidates, unit = seconds)."""
        server = self.server
        C = int(server.clients_per_round)
        if self._startup_t_round is None:
            totals = [self._passive_estimate(c, 0.0).total
                      for c in range(1, K.N_CLIENTS + 1)]
            self._startup_t_round = gpu_tier_sleep.compute_t_round_from_times(totals, C)
        training = getattr(server, "training_clients", {}) or {}
        busy = set(training)
        t_round = self._startup_t_round
        residuals = []
        if busy:
            live = gpu_tier_sleep.compute_t_round_from_times(
                [self._frozen[c] for c in busy], C)
            if live > 0.0:
                t_round = live
        wall_now = float(server.wall_time)
        if wall_now > 0.0 and training:
            for c in busy:
                start = float((training.get(c) or {}).get("start_time", wall_now))
                residuals.append(max(0.0, self._frozen[c] - (wall_now - start)))
        out = {}
        for c in pool:
            total = self._passive_estimate(c, t).total
            out[c] = (gpu_tier_sleep.gpu_tops_predict_staleness_rounds(total, residuals,
                                                                       t_round)
                      if residuals else gpu_tier_sleep.seconds_to_rounds(total, t_round))
        return out

    # -- one selection call ----------------------------------------------------

    def record(self, clients_pool, selected, context) -> None:
        server = self.server
        t = server.sumo_time()
        busy = set(getattr(server, "training_clients", {}) or {})
        pool = sorted(c for c in clients_pool if c not in busy)
        gate = getattr(server, "wen_stage0", None)
        if gate is not None and not gate.last_stats.get("fallback"):
            # S2 with Stage 0: the arm saw only the vehicles Stage 0 passed.
            pool = [c for c in pool if gate.last_pass.get(c)]
        if not selected:
            return
        tau = self._passive_tau(pool, t)
        G = self.g.G.copy()
        reports = {c: server.wen_report(c, t) for c in pool}
        passive = {c: self._passive_estimate(c, t) for c in pool}
        arm_rows = self._arm_columns(pool, selected, t, context)

        remaining = list(pool)
        for k, pick in enumerate(selected):
            self._decision += 1
            for c in remaining:
                rv, pe = reports[c], passive[c]
                row = {"run": self.run_id, "arm": self.arm, "seed": self.seed,
                       "round": server.current_round, "decision_id": self._decision,
                       "slot_k": k, "client": c, "tier": rv.tier, "x_dispatch": rv.x,
                       "V_k": rv.speed, "picked": c == pick,
                       "tau_hat_ours": tau[c], "T_hat_train_gpu": pe.T_train,
                       "T_hat_up_gpu": pe.T_up, "x_hat_finish_gpu": pe.x_finish,
                       "T_hat_total_gpu": pe.total, "G_ours": float(G[c - 1]),
                       "V_ours": float(G[c - 1]) / math.sqrt(1.0 + max(tau[c], 0.0)),
                       "n_merged_client": int(self.g.merged[c - 1])}
                if arm_rows is not None:
                    row.update(arm_rows(k, c, remaining))
                self._w.writerow({key: _r(v) for key, v in row.items()})
            remaining.remove(pick)
        self._fh.flush()
        for c in selected:
            self._frozen[c] = passive[c].total

    def _arm_columns(self, pool, selected, t, context):
        server = self.server
        sel = server.wen_selector
        agg = getattr(server.aggregation_strategy, "inner", None)
        unexplored = getattr(agg, "unexplored_clients", None) or set()
        if self.layout == "ours":
            scores, taus = sel.last_scores, sel.last_call_tau_hats
            G_state = np.asarray(context.state["polaris"]["local_gradient_bounds"])
            t_round = getattr(server.polaris_selector(), "last_t_round", None)
            ests = {c: server.wen_estimate(c, t) for c in pool}

            def cols(k, c, remaining):
                rank = sorted(remaining, key=lambda d: (-scores[d], d))
                gs = [float(G_state[d - 1]) for d in remaining]
                tmin = min(taus[d] for d in remaining)
                rec = sel.last_picks.get(selected[k])
                e = ests[c]
                return {"T_hat_train": e.T_train, "T_hat_up": e.T_up,
                        "x_hat_finish": e.x_finish,
                        "d_hat": F.distance_to_rsu(e.x_finish), "T_hat_total": e.total,
                        "T_round_used": t_round, "tau_hat": taus[c],
                        "G": float(G_state[c - 1]),
                        "G_source": "prior" if (c - 1) in unexplored else "measured",
                        "V": scores[c], "rank_by_V": rank.index(c) + 1,
                        "coin_u": rec.coin_u if rec else None,
                        "explored": rec.explored if rec else None,
                        "is_max_G": float(G_state[c - 1]) == max(gs),
                        "is_min_tau_hat": taus[c] == tmin}
            return cols
        if self.layout == "polaris":
            call = sel.last_call
            index = {c: j for j, c in enumerate(call["pool"])}
            q = call["q"]
            rank = sorted(call["pool"], key=lambda d: (-q[index[d]], d))
            seen_proxy = getattr(agg, "last_proxy", None) is not None

            def cols(k, c, remaining):
                j = index[c]
                draw = call["draws"][k]
                source = ("seen" if (c - 1) not in unexplored else
                          "alpha-proxy" if seen_proxy else "initial")
                return {"G_i": call["G"][j], "G_source": source, "tau_i": call["tau"][j],
                        "p_i": call["p"][j], "f1_i": call["f1"][j], "f2_i": call["f2"][j],
                        "mu": call["mu"], "q_i": q[j], "rank_by_q": rank.index(c) + 1,
                        "q_renormalised_at_pick": draw["q_renorm"][c],
                        "draw_u": draw["draw_u"]}
            return cols
        return None

    def close(self) -> None:
        self._fh.close()
        self._gfh.close()
