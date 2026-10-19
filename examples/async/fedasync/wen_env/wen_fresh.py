"""
`wen_fresh` (Prompt H): the Wen-faithful selector with a fresh F3.

Same selector as `wen` (wen_faithful.WenFaithfulSelection: F1, F2, F4, the
constraint limits, fallback, exact Pareto set, min-max normalisation, pick rule
and tie-break are inherited unchanged). Only F3 differs:

    F3_k = mean cross-entropy of the CURRENT global model on vehicle k's whole
           disjoint partition (wen_partition), eval mode, no augmentation, the
           CIFAR-10 test transform (ToTensor + the training Normalize).

All 30 vehicles are evaluated once per global-model version (the server's
`global_version` counts aggregations that changed the model). The evaluation
runs at the first decision after the version changes and is cached until the
next change, so every decision uses the model current at that decision. There
is no prior and no reported training loss.

The evaluation is not charged to the online clock or to energy: it happens on
the host between PLATO events and never touches the simulated clock. Its host
wall time is logged per version in `wen_fresh_eval.csv` (`fresh_eval_s`).

`stale_f3_test` (Prompt H Step 3, test 1) keeps the fresh evaluation running and
logged, but decides with the `wen` arm's stale F3, so the `wen_fresh` code path
can be checked against an existing `wen` run.
"""

from __future__ import annotations

import copy
import csv
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF

import wen_constants as K
import wen_partition
from wen_faithful import CANDIDATE_FIELDS, DECISION_FIELDS, WenFaithfulSelection

# The CIFAR-10 Normalize of plato/datasources/torchvision.py (train and test).
CIFAR10_MEAN = (0.485, 0.456, 0.406)
CIFAR10_STD = (0.229, 0.224, 0.225)
EVAL_BATCH = 1000


def cifar10_train_arrays(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(images uint8 [50000, 3, 32, 32], labels int64), in torchvision's order."""
    images, labels = [], []
    for i in range(1, 6):
        with (Path(data_path) / "cifar-10-batches-py" / f"data_batch_{i}").open("rb") as fh:
            batch = pickle.load(fh, encoding="latin1")
        images.append(np.asarray(batch["data"], dtype=np.uint8).reshape(-1, 3, 32, 32))
        labels += batch["labels"]
    return np.concatenate(images), np.asarray(labels, dtype=np.int64)


class FreshLoss:
    """Per-vehicle mean cross-entropy of a model over the 50,000 training images."""

    def __init__(self, data_path: Path, data_seed: int, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        images, labels = cifar10_train_arrays(data_path)
        part = wen_partition.build(labels.tolist(), data_seed)
        self.client_ids = sorted(part.indices)
        owner = np.full(len(labels), -1, dtype=np.int64)
        for j, cid in enumerate(self.client_ids):
            owner[part.indices[cid]] = j
        if (owner < 0).any() or len(labels) != K.TRAIN_IMAGES:
            raise ValueError("the partition does not cover the 50,000 training images")
        self.images = torch.from_numpy(images).to(self.device)
        self.labels = torch.from_numpy(labels).to(self.device)
        self.owner_np = owner
        self.counts_np = np.array([len(part.indices[c]) for c in self.client_ids], dtype=float)
        self.mean = torch.tensor(CIFAR10_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR10_STD, device=self.device).view(1, 3, 1, 1)

    def evaluate(self, model: torch.nn.Module) -> dict[int, float]:
        """{client_id: mean cross-entropy on its partition}; model must be on self.device."""
        was_training = model.training
        model.eval()
        flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        losses = []
        try:
            with torch.no_grad():
                for i in range(0, len(self.labels), EVAL_BATCH):
                    x = self.images[i:i + EVAL_BATCH].float().div(255)
                    x = (x - self.mean) / self.std
                    losses.append(TF.cross_entropy(model(x), self.labels[i:i + EVAL_BATCH],
                                                   reduction="none"))
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags
            model.train(was_training)
        # Per-vehicle sums on the CPU, in image order: CUDA index_add_ uses atomics,
        # whose order (and so the last bit of the sum) varies between runs.
        per_image = torch.cat(losses).double().cpu().numpy()
        sums = np.bincount(self.owner_np, weights=per_image, minlength=len(self.client_ids))
        means = (sums / self.counts_np).tolist()
        return dict(zip(self.client_ids, means))


# Study 2 (Prompt I Step 2.5): Stage 0's view of the same selection call, on
# each decision row of the `_s0` arms.
STAGE0_DECISION_FIELDS = ["n_eligible", "n_stage0_rejected", "n_stage0_rej_time",
                          "n_stage0_rej_energy", "n_stage0_rej_both", "n_stage0_rej_xavier",
                          "n_stage0_rej_orin", "n_stage0_rej_thor", "stage0_fallback"]
FRESH_EVAL_FIELDS = (["global_version", "round", "t", "fresh_eval_s"]
                     + [f"F3_{c}" for c in range(1, K.N_CLIENTS + 1)])


class WenFreshSelection(WenFaithfulSelection):
    decision_fields = DECISION_FIELDS + ["global_version"]
    candidate_fields = CANDIDATE_FIELDS + ["global_version"]

    def __init__(self, fresh: FreshLoss, stale_f3_test: bool = False,
                 stage0_columns: bool = False, **kwargs) -> None:
        if stage0_columns:
            self.decision_fields = self.decision_fields + STAGE0_DECISION_FIELDS
        self.stage0_columns = stage0_columns
        self._server = None
        super().__init__(**kwargs)
        self.fresh = fresh
        self.stale_f3_test = stale_f3_test
        self.version: int | None = None
        self.fresh_f3: dict[int, float] = {}
        self._model: torch.nn.Module | None = None
        out_dir = Path(kwargs["out_dir"])
        self._eval_fh = (out_dir / "wen_fresh_eval.csv").open("w", newline="")
        self._eval = csv.DictWriter(self._eval_fh, fieldnames=FRESH_EVAL_FIELDS)
        self._eval.writeheader()

    def refresh(self, server) -> None:
        """Re-evaluate all vehicles if the global model changed since the last call."""
        version = int(server.global_version)
        if version == self.version:
            return
        start = time.perf_counter()
        source = server.require_trainer().model
        if self._model is None:
            self._model = copy.deepcopy(source).to(self.fresh.device)
        self._model.load_state_dict(source.state_dict())
        self.fresh_f3 = self.fresh.evaluate(self._model)
        if self.fresh.device.type == "cuda":
            torch.cuda.synchronize(self.fresh.device)
        elapsed = time.perf_counter() - start
        self.version = version
        self._eval.writerow({
            "global_version": version, "round": int(server.current_round),
            "t": repr(server.sumo_time()), "fresh_eval_s": repr(elapsed),
            **{f"F3_{c}": repr(v) for c, v in self.fresh_f3.items()}})
        self._eval_fh.flush()

    def f3(self, client_id: int) -> float:
        if self.stale_f3_test:
            return super().f3(client_id)
        return self.fresh_f3[client_id]

    def select_clients(self, clients_pool, clients_count, context):
        self._server = context.server
        self.refresh(context.server)
        return super().select_clients(clients_pool, clients_count, context)

    def log_extra(self) -> dict:
        extra = {"global_version": self.version}
        gate = getattr(self._server, "wen_stage0", None)
        if self.stage0_columns and gate is not None:
            st = gate.last_stats
            extra.update(n_eligible=st["n_eligible"], n_stage0_rejected=st["n_rejected"],
                         n_stage0_rej_time=st["n_rej_time"],
                         n_stage0_rej_energy=st["n_rej_energy"],
                         n_stage0_rej_both=st["n_rej_both"],
                         n_stage0_rej_xavier=st["n_rej_xavier"],
                         n_stage0_rej_orin=st["n_rej_orin"],
                         n_stage0_rej_thor=st["n_rej_thor"],
                         stage0_fallback=st["fallback"])
        return extra
