"""
YOLOP trainer: ComposableTrainer wired with YOLOP-specific strategies.

Plato instantiates this as `custom_trainer(model=..., callbacks=...)` from
DefaultLifecycleStrategy.create_trainer (clients/strategies/defaults.py:88-95).
"""

from __future__ import annotations

import sys
import pathlib

# Ensure YOLOP imports resolve when the loss strategy needs the YACS bridge.
_YOLOP_ROOT = pathlib.Path(__file__).resolve().parents[3] / "YOLOP"
if str(_YOLOP_ROOT) not in sys.path:
    sys.path.insert(0, str(_YOLOP_ROOT))

from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies import (
    CustomCollateFnDataLoaderStrategy,
    DefaultTrainingStepStrategy,
)

from datasources.bdd100k import _build_yacs_cfg
from trainers.yolop_collate import yolop_collate_fn
from trainers.yolop_loss import YOLOPLossStrategy
from trainers.yolop_testing import YOLOPTestingStrategy


def _resolve_num_workers() -> int:
    """Read num_workers from [trainer] section, default 0 for safety."""
    return int(getattr(Config().trainer, "num_workers", 0))


def _resolve_persistent_workers() -> bool:
    """Read persistent_workers from [trainer] section, default False.

    When True, DataLoader worker processes stay alive between rounds so their
    spawn startup cost (~minutes with spawn mode) is paid only once per client
    process lifetime rather than once per FL round. Only effective when
    num_workers > 0.
    """
    return bool(getattr(Config().trainer, "persistent_workers", False))


def _resolve_pin_memory() -> bool:
    """Pin memory only if a CUDA device is available."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


class Trainer(ComposableTrainer):
    """ComposableTrainer with YOLOP-specific strategies wired in."""

    def __init__(self, model=None, callbacks=None):
        # Build the YACS cfg once so the loss strategy gets a frozen,
        # consistent view of cfg.LOSS regardless of how often it is
        # constructed during a session.
        yacs_cfg = _build_yacs_cfg(Config())

        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=YOLOPLossStrategy(yacs_cfg),
            # Default standard precision: YOLOP's loss uses fp32 BCE
            # pos_weights and FocalLoss math; mixed precision can produce
            # NaNs here. Switch to MixedPrecisionStepStrategy after the
            # baseline is stable.
            training_step_strategy=DefaultTrainingStepStrategy(),
            data_loader_strategy=CustomCollateFnDataLoaderStrategy(
                collate_fn=yolop_collate_fn,
                num_workers=_resolve_num_workers(),
                pin_memory=_resolve_pin_memory(),
                drop_last=False,
                persistent_workers=_resolve_persistent_workers(),
            ),
            testing_strategy=YOLOPTestingStrategy(),
        )
