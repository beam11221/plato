"""
Disjoint Dirichlet partition of the 50,000 CIFAR-10 training images.

Locked: 20 clients x 1,667 + 10 x 1,666 = 50,000, Dirichlet alpha = 0.8, no image
reused. PLATO's `noniid` sampler cannot do this -- it draws every client's
subset independently from the whole training set, so images repeat across
clients and the total is not tied to the dataset size.

Construction (deterministic in `seed`):

  1. Each client draws label proportions p_i ~ Dir(0.8 * 1_10).
  2. The 30 x 10 matrix of desired counts |S_i| * p_i is scaled by
     iterative proportional fitting (Sinkhorn) so its row sums are exactly
     |S_i| and its column sums exactly the 5,000 images per class. This is
     the minimum-KL adjustment of the drawn proportions that makes a disjoint
     split of the whole dataset possible; it shrinks over-subscribed classes
     and tops up under-subscribed ones in proportion to each client's own draw.
  3. The fitted matrix is rounded to integers with both marginals preserved
     (floor, then hand the remaining units to the largest fractional parts,
     never past a column's remaining stock).
  4. Within each class, the image indices are shuffled once and dealt out in
     client order.

`summary()` reports how far step 2 moved each client from its own Dirichlet
draw (total variation distance), so the cost of the "no reuse" rule is visible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

import wen_constants as K

SINKHORN_ITERS = 10_000
SINKHORN_TOL = 1e-9


@dataclass(frozen=True)
class Partition:
    indices: dict[int, np.ndarray]    # client_id -> image indices (sorted)
    counts: np.ndarray                # [client, class] integer counts
    drawn: np.ndarray                 # [client, class] Dirichlet proportions

    def fingerprint(self, client_id: int) -> str:
        return hashlib.sha256(self.indices[client_id].tobytes()).hexdigest()[:12]

    def summary(self) -> dict:
        realised = self.counts / self.counts.sum(axis=1, keepdims=True)
        tv = 0.5 * np.abs(realised - self.drawn).sum(axis=1)
        return {
            "sizes": {cid: int(len(ix)) for cid, ix in self.indices.items()},
            "tv_from_draw_mean": float(tv.mean()),
            "tv_from_draw_max": float(tv.max()),
            "max_class_share_mean": float(realised.max(axis=1).mean()),
        }


def _sinkhorn(m: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    m = m.copy()
    for _ in range(SINKHORN_ITERS):
        m *= (rows / m.sum(axis=1))[:, None]
        m *= (cols / m.sum(axis=0))[None, :]
        if np.abs(m.sum(axis=1) - rows).max() < SINKHORN_TOL:
            return m
    raise RuntimeError("Sinkhorn did not converge")


def _round_preserving(m: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    counts = np.floor(m).astype(int)
    frac = m - counts
    row_left = rows - counts.sum(axis=1)
    col_left = cols - counts.sum(axis=0)
    # Pass 1: largest fractional parts first, at most one extra unit per cell.
    # Pass 2: whatever is left goes to any cell whose column still has stock,
    # still in fractional order. Row and column totals are equal, so pass 2
    # always completes.
    order = np.argsort(-frac, axis=None, kind="stable")
    for unbounded in (False, True):
        for flat in order:
            i, j = divmod(int(flat), m.shape[1])
            while row_left[i] > 0 and col_left[j] > 0:
                counts[i, j] += 1
                row_left[i] -= 1
                col_left[j] -= 1
                if not unbounded:
                    break
        if row_left.sum() == 0:
            break
    if row_left.any() or col_left.any():
        raise RuntimeError("integer rounding could not meet both marginals")
    return counts


def build(targets, seed: int, alpha: float = K.DIRICHLET_ALPHA) -> Partition:
    targets = np.asarray(targets)
    classes = np.unique(targets)
    client_ids = list(range(1, K.N_CLIENTS + 1))
    rows = np.array([K.partition_size(c) for c in client_ids], dtype=float)
    cols = np.array([(targets == c).sum() for c in classes], dtype=float)
    if rows.sum() != cols.sum() or rows.sum() != K.TRAIN_IMAGES:
        raise ValueError(f"sizes sum to {rows.sum()}, dataset has {cols.sum()}")

    rng = np.random.default_rng([seed, 0x5EED])
    drawn = rng.dirichlet(np.full(len(classes), alpha), size=len(client_ids))
    fitted = _sinkhorn(drawn * rows[:, None] + 1e-12, rows, cols)
    counts = _round_preserving(fitted, rows.astype(int), cols.astype(int))

    pools = {c: rng.permutation(np.flatnonzero(targets == c)) for c in classes}
    taken = {c: 0 for c in classes}
    indices: dict[int, np.ndarray] = {}
    for i, cid in enumerate(client_ids):
        parts = []
        for j, c in enumerate(classes):
            n = int(counts[i, j])
            parts.append(pools[c][taken[c]:taken[c] + n])
            taken[c] += n
        indices[cid] = np.sort(np.concatenate(parts))
    return Partition(indices=indices, counts=counts, drawn=drawn)
