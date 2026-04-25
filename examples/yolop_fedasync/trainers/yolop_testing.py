"""
YOLOP testing strategy for Plato's ComposableTrainer.

Ports the canonical eval flow from YOLOP/lib/core/function.py:validate(),
returning mAP@0.5 as the primary metric (stored as ComposableTrainer's
`accuracy`) while logging DA mIoU and LL IoU into context.state for
optional callback consumption.

Reuses YOLOP utilities verbatim:
  - non_max_suppression, scale_coords, xywh2xyxy, box_iou
  - SegmentationMetric, ap_per_class
"""

from __future__ import annotations

import sys
import pathlib
import logging
from typing import Any

import numpy as np
import torch
import torch.utils.data

# Ensure YOLOP imports resolve. trainers/yolop_testing.py → parents[4] is /workspace.
_YOLOP_ROOT = pathlib.Path(__file__).resolve().parents[4] / "YOLOP"
if str(_YOLOP_ROOT) not in sys.path:
    sys.path.insert(0, str(_YOLOP_ROOT))

from lib.core.evaluate import SegmentationMetric, ap_per_class
from lib.core.general import (
    box_iou,
    non_max_suppression,
    scale_coords,
    xywh2xyxy,
)

from plato.config import Config
from plato.trainers.strategies.base import TestingStrategy, TrainingContext

from .multitask_labels import MultiTaskLabels
from .yolop_collate import yolop_collate_fn


# Mirror YOLOP's defaults from lib/config/default.py:172-174.
_DEFAULT_NMS_CONF = 0.001
_DEFAULT_NMS_IOU = 0.6


def _resolve_sampler(sampler):
    """Convert a Plato sampler (or None) to something a DataLoader accepts."""
    if sampler is None:
        return None
    if hasattr(sampler, "get"):
        return sampler.get()
    return sampler


def _resolve_test_batch_size() -> int:
    cfg = Config()
    bs = getattr(getattr(cfg, "tester", object()), "batch_size", None)
    if bs is None:
        bs = getattr(cfg.trainer, "batch_size", 16)
    return int(bs)


def _resolve_num_workers() -> int:
    cfg = Config()
    nw = getattr(cfg.trainer, "num_workers", 0)
    return int(nw)


class YOLOPTestingStrategy(TestingStrategy):
    """Run YOLOP-style multi-task evaluation; return mAP@0.5."""

    def __init__(
        self,
        conf_thres: float = _DEFAULT_NMS_CONF,
        iou_thres: float = _DEFAULT_NMS_IOU,
    ):
        super().__init__()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def test_model(
        self,
        model,
        config,  # dict-like (Plato passes a Munch / dict)
        testset,
        sampler,
        context: TrainingContext,
    ) -> float:
        device = context.device
        if device is None:
            device = next(model.parameters()).device

        batch_size = _resolve_test_batch_size()
        loader_sampler = _resolve_sampler(sampler)

        loader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            sampler=loader_sampler,
            num_workers=_resolve_num_workers(),
            pin_memory=False,
            collate_fn=yolop_collate_fn,
        )

        # Per-batch averaging accumulators (YOLOP convention).
        # We keep totals and sample counts to compute weighted means at end.
        da_iou_sum = 0.0
        da_miou_sum = 0.0
        ll_iou_sum = 0.0
        ll_miou_sum = 0.0
        sample_count = 0

        # IoU thresholds for AP@0.5:0.95 (10 thresholds, 0.05 step).
        iouv = torch.linspace(0.5, 0.95, 10).to(device)
        niou = iouv.numel()
        nc = int(getattr(model, "nc", 1))

        # YOLOP's BddDataset is hardcoded for binary masks (two classes:
        # foreground/background) regardless of cfg.num_seg_class — the
        # __getitem__ stacks (seg2, seg1) producing shape (2, H, W).
        da_metric = SegmentationMetric(2)
        ll_metric = SegmentationMetric(2)
        stats: list[tuple] = []

        model.to(device)
        was_training = model.training
        model.eval()

        try:
            with torch.no_grad():
                for imgs, lbl in loader:
                    if not isinstance(lbl, MultiTaskLabels):
                        raise TypeError(
                            "YOLOPTestingStrategy expects MultiTaskLabels; got "
                            f"{type(lbl).__name__}. Make sure the testset uses "
                            "yolop_collate_fn."
                        )
                    imgs = imgs.to(device, non_blocking=True)
                    lbl = lbl.to(device)
                    nb, _, height, width = imgs.shape

                    # `shapes` is a tuple of length B; YOLOP assumes uniform
                    # padding across the batch and uses shapes[0].
                    pad_w, pad_h = lbl.shapes[0][1][1]
                    pad_w = int(pad_w)
                    pad_h = int(pad_h)

                    # Forward — eval mode means Detect returns (inf_out, raw_list).
                    out = model(imgs)
                    det_out, da_seg_out, ll_seg_out = out
                    inf_out, _train_out = det_out  # eval-mode tuple

                    # ---- DA segmentation: argmax, crop padding, addBatch ----
                    _, da_pred = torch.max(da_seg_out, 1)
                    _, da_gt = torch.max(lbl.da_seg, 1)
                    da_pred = da_pred[:, pad_h:height - pad_h, pad_w:width - pad_w]
                    da_gt = da_gt[:, pad_h:height - pad_h, pad_w:width - pad_w]
                    da_metric.reset()
                    da_metric.addBatch(da_pred.cpu(), da_gt.cpu())
                    da_iou_sum += da_metric.IntersectionOverUnion() * nb
                    da_miou_sum += da_metric.meanIntersectionOverUnion() * nb

                    # ---- LL segmentation: argmax, crop padding, addBatch ----
                    _, ll_pred = torch.max(ll_seg_out, 1)
                    _, ll_gt = torch.max(lbl.ll_seg, 1)
                    ll_pred = ll_pred[:, pad_h:height - pad_h, pad_w:width - pad_w]
                    ll_gt = ll_gt[:, pad_h:height - pad_h, pad_w:width - pad_w]
                    ll_metric.reset()
                    ll_metric.addBatch(ll_pred.cpu(), ll_gt.cpu())
                    ll_iou_sum += ll_metric.IntersectionOverUnion() * nb
                    ll_miou_sum += ll_metric.meanIntersectionOverUnion() * nb

                    sample_count += nb

                    # ---- Detection: NMS + per-image TP matching ----
                    # Convert normalized xywh targets to pixel xywh for matching.
                    # NOTE: this mutates lbl.det in place — fine since we don't
                    # reuse it after this batch.
                    target_det = lbl.det.clone()  # avoid mutating wrapper
                    if target_det.numel():
                        target_det[:, 2:] *= torch.tensor(
                            [width, height, width, height], device=device
                        )

                    output = non_max_suppression(
                        inf_out,
                        conf_thres=self.conf_thres,
                        iou_thres=self.iou_thres,
                    )

                    # Per-image stats accumulation (matches function.py:328-410).
                    for si, pred in enumerate(output):
                        labels = target_det[target_det[:, 0] == si, 1:]
                        nl = labels.shape[0]
                        tcls = labels[:, 0].tolist() if nl else []

                        if pred is None or len(pred) == 0:
                            if nl:
                                stats.append((
                                    torch.zeros(0, niou, dtype=torch.bool),
                                    torch.empty(0),
                                    torch.empty(0),
                                    tcls,
                                ))
                            continue

                        # Native-space predictions (rescaled to original image).
                        predn = pred.clone()
                        scale_coords(
                            imgs[si].shape[1:],
                            predn[:, :4],
                            lbl.shapes[si][0],
                            lbl.shapes[si][1],
                        )

                        correct = torch.zeros(
                            pred.shape[0], niou, dtype=torch.bool, device=device
                        )

                        if nl:
                            detected: list[torch.Tensor] = []
                            tcls_tensor = labels[:, 0]

                            tbox = xywh2xyxy(labels[:, 1:5])
                            scale_coords(
                                imgs[si].shape[1:],
                                tbox,
                                lbl.shapes[si][0],
                                lbl.shapes[si][1],
                            )

                            for cls in torch.unique(tcls_tensor):
                                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                                pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)

                                if pi.shape[0]:
                                    ious, i = box_iou(
                                        predn[pi, :4], tbox[ti]
                                    ).max(1)
                                    detected_set: set[int] = set()
                                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                        d = ti[i[j]]
                                        if d.item() not in detected_set:
                                            detected_set.add(d.item())
                                            detected.append(d)
                                            correct[pi[j]] = ious[j] > iouv
                                            if len(detected) == nl:
                                                break

                        stats.append((
                            correct.cpu(),
                            pred[:, 4].cpu(),
                            pred[:, 5].cpu(),
                            tcls,
                        ))
        finally:
            if was_training:
                model.train()

        # Compute mAP from accumulated stats.
        map50 = 0.0
        if stats:
            np_stats = [np.concatenate(x, 0) for x in zip(*stats)]
            if len(np_stats) and np_stats[0].any():
                p, r, ap, f1, ap_class = ap_per_class(
                    *np_stats, plot=False, save_dir=".", names={}
                )
                # ap shape: (nc, 10) — column 0 is IoU=0.5
                ap50 = ap[:, 0]
                map50 = float(ap50.mean()) if ap50.size else 0.0

        # Final per-batch-averaged segmentation metrics.
        if sample_count:
            da_iou = da_iou_sum / sample_count
            da_miou = da_miou_sum / sample_count
            ll_iou = ll_iou_sum / sample_count
            ll_miou = ll_miou_sum / sample_count
        else:
            da_iou = da_miou = ll_iou = ll_miou = 0.0

        # Stash secondary metrics for callbacks/loggers.
        context.state["yolop_da_iou"] = float(da_iou)
        context.state["yolop_da_miou"] = float(da_miou)
        context.state["yolop_ll_iou"] = float(ll_iou)
        context.state["yolop_ll_miou"] = float(ll_miou)
        context.state["yolop_map50"] = float(map50)

        logging.info(
            "[YOLOP eval] mAP@0.5=%.4f  DA-IoU=%.4f  DA-mIoU=%.4f  "
            "LL-IoU=%.4f  LL-mIoU=%.4f  (n=%d)",
            map50, da_iou, da_miou, ll_iou, ll_miou, sample_count,
        )

        return float(map50)
