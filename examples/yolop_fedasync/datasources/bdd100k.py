"""
BDD100K DataSource for Plato.

Loads the *full* BDD100K dataset (no per-client split). Per-client
partitioning is delegated to Plato's IID sampler, which subsets the
training indices uniformly across clients.

The bridge function `_build_yacs_cfg(plato_cfg)` translates Plato's TOML
[parameters.dataset] / [parameters.loss] / [parameters.model] sections
into a YACS config object that YOLOP's BddDataset and MultiHeadLoss
already understand.
"""

from __future__ import annotations

import sys
import pathlib
import logging
from typing import Any

import torchvision.transforms as transforms
from yacs.config import CfgNode as CN

# YOLOP imports go through sys.path mutation. Done once at module load.
# datasources/bdd100k.py → parents[4] is /workspace.
_YOLOP_ROOT = pathlib.Path(__file__).resolve().parents[4] / "YOLOP"
if str(_YOLOP_ROOT) not in sys.path:
    sys.path.insert(0, str(_YOLOP_ROOT))

from lib.config.default import _C as _YOLOP_CFG_DEFAULTS
from lib.dataset.bdd import BddDataset

from plato.config import Config
from plato.datasources import base


# YOLOP standard ImageNet normalization (matches tools/train.py:257-259).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _namespace_to_dict(ns: Any) -> dict:
    """Convert a Plato Config Namespace (or dict) to a plain dict.

    Plato's TOML loader yields argparse.Namespace objects for nested
    sections; nested namespaces show up as Namespace attributes. We walk
    them recursively so the rest of the bridge logic can use dict access.
    """
    if ns is None:
        return {}
    if isinstance(ns, dict):
        return {k: _namespace_to_dict(v) for k, v in ns.items()}
    if hasattr(ns, "__dict__"):
        return {k: _namespace_to_dict(v) for k, v in vars(ns).items()}
    return ns


def _build_yolop_transform() -> transforms.Compose:
    """Standard YOLOP transform: ToTensor + ImageNet Normalize."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _build_yacs_cfg(plato_cfg) -> CN:
    """Bridge Plato TOML config → frozen YACS config for YOLOP.

    Reads:
        plato_cfg.parameters.dataset.{dataroot,labelroot,maskroot,laneroot,
                                       train_set,test_set,img_size,org_img_size,
                                       num_seg_class}
        plato_cfg.parameters.loss.{multi_head_lambda, cls_pos_weight,
                                    obj_pos_weight, seg_pos_weight, fl_gamma,
                                    cls_gain, obj_gain, box_gain, da_seg_gain,
                                    ll_seg_gain, ll_iou_gain}

    Any missing key falls back to the YOLOP defaults in
    YOLOP/lib/config/default.py.
    """
    cfg = _YOLOP_CFG_DEFAULTS.clone()
    cfg.defrost()

    parameters = getattr(plato_cfg, "parameters", None)

    # ----- DATASET section -----
    ds = _namespace_to_dict(getattr(parameters, "dataset", None)) if parameters else {}
    if "dataroot" in ds:
        cfg.DATASET.DATAROOT = str(ds["dataroot"])
    if "labelroot" in ds:
        cfg.DATASET.LABELROOT = str(ds["labelroot"])
    if "maskroot" in ds:
        cfg.DATASET.MASKROOT = str(ds["maskroot"])
    if "laneroot" in ds:
        cfg.DATASET.LANEROOT = str(ds["laneroot"])
    if "train_set" in ds:
        cfg.DATASET.TRAIN_SET = str(ds["train_set"])
    if "test_set" in ds:
        cfg.DATASET.TEST_SET = str(ds["test_set"])
    if "img_size" in ds:
        cfg.MODEL.IMAGE_SIZE = list(ds["img_size"])
    if "org_img_size" in ds:
        cfg.DATASET.ORG_IMG_SIZE = list(ds["org_img_size"])
    if "num_seg_class" in ds:
        cfg.num_seg_class = int(ds["num_seg_class"])

    # ----- LOSS section -----
    loss = _namespace_to_dict(getattr(parameters, "loss", None)) if parameters else {}
    if "multi_head_lambda" in loss and loss["multi_head_lambda"] is not None:
        cfg.LOSS.MULTI_HEAD_LAMBDA = list(loss["multi_head_lambda"])
    for key, yacs_key in [
        ("cls_pos_weight", "CLS_POS_WEIGHT"),
        ("obj_pos_weight", "OBJ_POS_WEIGHT"),
        ("seg_pos_weight", "SEG_POS_WEIGHT"),
        ("fl_gamma", "FL_GAMMA"),
        ("cls_gain", "CLS_GAIN"),
        ("obj_gain", "OBJ_GAIN"),
        ("box_gain", "BOX_GAIN"),
        ("da_seg_gain", "DA_SEG_GAIN"),
        ("ll_seg_gain", "LL_SEG_GAIN"),
        ("ll_iou_gain", "LL_IOU_GAIN"),
    ]:
        if key in loss:
            setattr(cfg.LOSS, yacs_key, float(loss[key]))

    cfg.freeze()
    return cfg


class DataSource(base.DataSource):
    """BDD100K data source.

    Loads the full BDD100K dataset on every client. Plato's IID sampler
    subsets the indices per client at data-loader creation time, so we
    don't need to do anything client-specific here.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # `client_id` may be passed by Plato's registry path, but we ignore it
        # because per-client partitioning is the sampler's job.
        cfg = _build_yacs_cfg(Config())
        transform = _build_yolop_transform()

        logging.info(
            "[BDD100K] Loading dataset from %s (train_set=%s, test_set=%s)",
            cfg.DATASET.DATAROOT, cfg.DATASET.TRAIN_SET, cfg.DATASET.TEST_SET,
        )
        # client_id=None → BddDataset reads cfg.DATASET.* (the full dataset).
        # inputsize accepts a list (BddDataset takes max() of it).
        self.trainset = BddDataset(
            cfg=cfg,
            is_train=True,
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transform,
            client_id=None,
        )
        self.testset = BddDataset(
            cfg=cfg,
            is_train=False,
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transform,
            client_id=None,
        )
        logging.info(
            "[BDD100K] trainset=%d, testset=%d",
            len(self.trainset), len(self.testset),
        )

    def targets(self):
        """Stub targets — IID sampler only uses len()."""
        return [0] * len(self.trainset)

    def num_train_examples(self) -> int:
        return len(self.trainset)

    def num_test_examples(self) -> int:
        return len(self.testset)
