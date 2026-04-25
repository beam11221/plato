"""
YOLOP loss strategy for Plato's ComposableTrainer.

The strategy wraps YOLOP's MultiHeadLoss (lib/core/loss.py). The loss is
constructed lazily on the first compute_loss() call so we can pin its
device to wherever the model actually lives, regardless of how Plato's
strategy setup ordering interacts with model.to(device).
"""

from __future__ import annotations

import sys
import pathlib

import torch

# Ensure YOLOP imports resolve. trainers/yolop_loss.py → parents[4] is /workspace.
_YOLOP_ROOT = pathlib.Path(__file__).resolve().parents[4] / "YOLOP"
if str(_YOLOP_ROOT) not in sys.path:
    sys.path.insert(0, str(_YOLOP_ROOT))

from lib.core.loss import get_loss

from plato.trainers.strategies.base import LossCriterionStrategy, TrainingContext

from .multitask_labels import MultiTaskLabels


class YOLOPLossStrategy(LossCriterionStrategy):
    """Compute YOLOP's six-component multi-task loss."""

    def __init__(self, yacs_cfg):
        super().__init__()
        self._cfg = yacs_cfg
        self._criterion = None

    def setup(self, context: TrainingContext) -> None:
        """No-op; criterion is built lazily on first compute_loss."""
        # We could build it here using context.device, but that device may
        # be set before model.to(device) is actually called. Lazy init in
        # compute_loss() reads the model's *actual* parameter device.
        self._criterion = None

    def _ensure_criterion(self, context: TrainingContext) -> None:
        if self._criterion is not None:
            return
        # Use the model's actual device — most reliable source of truth.
        try:
            device = next(context.model.parameters()).device
        except StopIteration:
            device = context.device or torch.device("cpu")
        self._criterion = get_loss(self._cfg, device)

    def compute_loss(
        self, outputs, labels: MultiTaskLabels, context: TrainingContext
    ) -> torch.Tensor:
        """Forward through MultiHeadLoss, return scalar total loss."""
        self._ensure_criterion(context)

        # `outputs` is the model's forward return: [det_out, da_seg, ll_seg].
        # In train mode (set by ComposableTrainer.train), Detect.forward
        # returns the raw 3-scale list, which is exactly what the loss expects.
        head_targets = [labels.det, labels.da_seg, labels.ll_seg]

        total_loss, _head_losses = self._criterion(
            outputs,
            head_targets,
            labels.shapes,
            context.model,
        )

        # Stash per-head losses for callbacks/loggers if anyone cares.
        # Order: (lbox, lobj, lcls, lseg_da, lseg_ll, liou_ll, total).
        context.state["yolop_head_losses"] = _head_losses

        return total_loss
