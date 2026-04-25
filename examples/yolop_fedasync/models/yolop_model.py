"""
YOLOP model factory for Plato.

Plato's lifecycle calls our factory as a zero-argument callable
(see composable.py:121-128 — `module = model()` if `model` is callable).
We expose `Model.get` as the entry point.

Important: YOLOP's loss code reads `model.gr` at runtime (see
YOLOP/lib/core/loss.py:100). This attribute is NOT set by MCnet.__init__;
every YOLOP training script (tools/train.py:251, tools/test.py:152)
sets `model.gr = 1.0` manually after construction. We do the same here
to avoid AttributeError on the first training step.
"""

from __future__ import annotations

import sys
import pathlib
import logging

# Ensure YOLOP imports resolve. models/yolop_model.py → parents[4] is /workspace.
_YOLOP_ROOT = pathlib.Path(__file__).resolve().parents[4] / "YOLOP"
if str(_YOLOP_ROOT) not in sys.path:
    sys.path.insert(0, str(_YOLOP_ROOT))

from lib.config.default import _C as _YOLOP_CFG_DEFAULTS
from lib.models.YOLOP import get_net

from plato.config import Config


# Default IoU-objectness gain factor used by the loss; matches every
# YOLOP training script.
_DEFAULT_GR = 1.0


def _build_model_cfg(plato_cfg):
    """Build a YACS cfg for model construction.

    Currently `get_net(cfg)` only consumes the YOLOP block list (a
    module-level constant) and does not read from `cfg`. We still pass
    a properly-formed cfg with overridden IMAGE_SIZE for forward
    consistency in case later YOLOP versions use it.
    """
    cfg = _YOLOP_CFG_DEFAULTS.clone()
    cfg.defrost()

    parameters = getattr(plato_cfg, "parameters", None)
    if parameters is not None:
        ds = getattr(parameters, "dataset", None)
        if ds is not None and hasattr(ds, "img_size"):
            cfg.MODEL.IMAGE_SIZE = list(ds.img_size)

    cfg.freeze()
    return cfg


class Model:
    """Factory wrapper exposing `get(model_name=None, **kwargs)`."""

    @staticmethod
    def get(model_name: str | None = None, **kwargs):
        """Build and return a YOLOP MCnet ready for federated training."""
        cfg = _build_model_cfg(Config())
        model = get_net(cfg)

        # Required by MultiHeadLoss._forward_impl. NOT set by MCnet itself.
        model.gr = _DEFAULT_GR

        logging.info(
            "[YOLOP] Model constructed: nc=%d, detector_index=%d, "
            "seg_out_idx=%s, gr=%.3f",
            model.nc, model.detector_index, model.seg_out_idx, model.gr,
        )
        return model
