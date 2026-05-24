"""
YOLOP testing strategy for Plato's ComposableTrainer.

Mirrors YOLOP's canonical validate() from lib/core/function.py in a single
forward pass that computes:
  - Multi-task validation loss (stored as context.state["valid_loss"])
  - Per-head losses: lbox, lobj, lcls, lseg_da, lseg_ll, liou_ll
  - mAP@0.5 (stored as the returned accuracy value)
  - DA-seg IoU / mIoU and LL-seg IoU / mIoU

When enable_plots=True, saves per-image overlays for the first batch of
each eval round to <base_path>/visualization/:
  batch_{round}_{i}_da_segresult.png  /  _da_seg_gt.png
  batch_{round}_{i}_ll_segresult.png  /  _ll_seg_gt.png
  batch_{round}_{i}_det_pred.png      /  _det_gt.png
"""

from __future__ import annotations

import ctypes
import gc
import os
import sys
import pathlib
import logging
from typing import Any

import cv2
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
    plot_one_box,
    scale_coords,
    xywh2xyxy,
)
from lib.core.loss import get_loss
from lib.utils import show_seg_result

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


def _resolve_test_num_workers() -> int:
    # Inference is GPU-bound, not data-loading bound, so a small worker count
    # is sufficient. Using the full training num_workers spawns unnecessary
    # forked processes that compete for RAM during server-side evaluation.
    # Use at most 2 workers regardless of the training setting.
    cfg = Config()
    nw = getattr(cfg.trainer, "num_workers", 0)
    return min(int(nw), 2)


class YOLOPTestingStrategy(TestingStrategy):
    """
    Single-pass YOLOP evaluation: loss + seg metrics + mAP in one loop,
    with optional first-batch visualization to disk.
    """

    def __init__(
        self,
        conf_thres: float = _DEFAULT_NMS_CONF,
        iou_thres: float = _DEFAULT_NMS_IOU,
        yacs_cfg=None,
        enable_plots: bool = False,
    ):
        super().__init__()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self._yacs_cfg = yacs_cfg
        self._criterion = None
        self.enable_plots = enable_plots
        self._eval_round = 0  # incremented each test_model() call for file naming

    def _ensure_criterion(self, device) -> None:
        """Lazily build MultiHeadLoss once we know the device."""
        if self._criterion is not None or self._yacs_cfg is None:
            return
        self._criterion = get_loss(self._yacs_cfg, device)

    def test_model(
        self,
        model,
        config,
        testset,
        sampler,
        context: TrainingContext,
    ) -> float:
        device = context.device
        if device is None:
            device = next(model.parameters()).device

        self._ensure_criterion(device)
        self._eval_round += 1

        batch_size = _resolve_test_batch_size()
        loader_sampler = _resolve_sampler(sampler)

        test_workers = _resolve_test_num_workers()
        loader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            sampler=loader_sampler,
            num_workers=test_workers,
            pin_memory=False,
            collate_fn=yolop_collate_fn,
            prefetch_factor=2 if test_workers > 0 else None,
        )

        iouv = torch.linspace(0.5, 0.95, 10).to(device)
        niou = iouv.numel()

        da_metric = SegmentationMetric(2)
        ll_metric = SegmentationMetric(2)
        stats: list[tuple] = []

        da_iou_sum = 0.0
        da_miou_sum = 0.0
        ll_iou_sum = 0.0
        ll_miou_sum = 0.0
        val_loss_sum = 0.0
        head_loss_sums = [0.0] * 6  # lbox, lobj, lcls, lseg_da, lseg_ll, liou_ll
        sample_count = 0

        # Resolve visualization save directory once before the loop.
        if self.enable_plots:
            try:
                _base_path = Config().params["base_path"]
            except (AttributeError, KeyError):
                _base_path = "."
            _save_dir = os.path.join(_base_path, "visualization")
            os.makedirs(_save_dir, exist_ok=True)
            _model_names = getattr(model, "names", ["vehicle"])
        _visualized = False  # draw first batch only

        model.to(device)
        was_training = model.training
        model.eval()

        try:
            with torch.no_grad():
                for imgs, lbl in loader:
                    if not isinstance(lbl, MultiTaskLabels):
                        raise TypeError(
                            "YOLOPTestingStrategy expects MultiTaskLabels; got "
                            f"{type(lbl).__name__}."
                        )
                    imgs = imgs.to(device, non_blocking=True)
                    lbl = lbl.to(device)
                    nb, _, height, width = imgs.shape

                    pad_w, pad_h = lbl.shapes[0][1][1]
                    pad_w = int(pad_w)
                    pad_h = int(pad_h)

                    # Forward — eval mode: Detect returns (inf_out, train_out).
                    out = model(imgs)
                    det_out, da_seg_out, ll_seg_out = out
                    inf_out, train_out = det_out

                    # ---- Multi-task validation loss ----
                    if self._criterion is not None:
                        head_targets = [lbl.det, lbl.da_seg, lbl.ll_seg]
                        total_loss, head_losses = self._criterion(
                            (train_out, da_seg_out, ll_seg_out),
                            head_targets,
                            lbl.shapes,
                            model,
                        )
                        val_loss_sum += total_loss.item() * nb
                        for k, v in enumerate(head_losses[:6]):
                            head_loss_sums[k] += v * nb

                    # ---- DA segmentation metrics ----
                    _, da_pred = torch.max(da_seg_out, 1)
                    _, da_gt = torch.max(lbl.da_seg, 1)
                    da_pred = da_pred[:, pad_h:height - pad_h, pad_w:width - pad_w]
                    da_gt = da_gt[:, pad_h:height - pad_h, pad_w:width - pad_w]
                    da_metric.reset()
                    da_metric.addBatch(da_pred.cpu(), da_gt.cpu())
                    da_iou_sum += da_metric.IntersectionOverUnion() * nb
                    da_miou_sum += da_metric.meanIntersectionOverUnion() * nb

                    # ---- LL segmentation metrics ----
                    _, ll_pred = torch.max(ll_seg_out, 1)
                    _, ll_gt = torch.max(lbl.ll_seg, 1)
                    ll_pred = ll_pred[:, pad_h:height - pad_h, pad_w:width - pad_w]
                    ll_gt = ll_gt[:, pad_h:height - pad_h, pad_w:width - pad_w]
                    ll_metric.reset()
                    ll_metric.addBatch(ll_pred.cpu(), ll_gt.cpu())
                    ll_iou_sum += ll_metric.IntersectionOverUnion() * nb
                    ll_miou_sum += ll_metric.meanIntersectionOverUnion() * nb

                    sample_count += nb

                    # ---- Detection: NMS + per-image stats ----
                    target_det = lbl.det.clone()
                    if target_det.numel():
                        target_det[:, 2:] *= torch.tensor(
                            [width, height, width, height], device=device
                        )

                    output = non_max_suppression(
                        inf_out,
                        conf_thres=self.conf_thres,
                        iou_thres=self.iou_thres,
                    )

                    # ---- Visualization: first batch only ----
                    if self.enable_plots and not _visualized:
                        _visualized = True
                        paths = lbl.paths or []
                        for img_i in range(min(nb, len(paths))):
                            orig = cv2.imread(str(paths[img_i]))
                            if orig is None:
                                continue  # file inaccessible — skip silently
                            orig_h, orig_w = orig.shape[:2]

                            # Resize cropped mask to original image dimensions.
                            # INTER_NEAREST preserves integer class labels.
                            def _resize_mask(mask_tensor):
                                arr = mask_tensor[img_i].cpu().numpy().astype(np.uint8)
                                return cv2.resize(arr, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                            # DA segmentation pred + gt
                            show_seg_result(orig.copy(), _resize_mask(da_pred), img_i, self._eval_round, _save_dir)
                            show_seg_result(orig.copy(), _resize_mask(da_gt),   img_i, self._eval_round, _save_dir, is_gt=True)

                            # LL segmentation pred + gt
                            show_seg_result(orig.copy(), _resize_mask(ll_pred), img_i, self._eval_round, _save_dir, is_ll=True)
                            show_seg_result(orig.copy(), _resize_mask(ll_gt),   img_i, self._eval_round, _save_dir, is_ll=True, is_gt=True)

                            # Detection predictions
                            det_img = orig.copy()
                            if output[img_i] is not None and len(output[img_i]):
                                det = output[img_i].clone()
                                scale_coords(imgs[img_i].shape[1:], det[:, :4], lbl.shapes[img_i][0], lbl.shapes[img_i][1])
                                for *xyxy, conf, cls in reversed(det.tolist()):
                                    label = f"{_model_names[int(cls)]} {conf:.2f}"
                                    plot_one_box(xyxy, det_img, label=label, line_thickness=2)
                            cv2.imwrite(os.path.join(_save_dir, f"batch_{self._eval_round}_{img_i}_det_pred.png"), det_img)

                            # Detection ground truth
                            gt_img = orig.copy()
                            img_gt = target_det[target_det[:, 0] == img_i, 1:]  # [cls, xc, yc, w, h] in pixels
                            if len(img_gt):
                                tbox = xywh2xyxy(img_gt[:, 1:5].clone())
                                scale_coords(imgs[img_i].shape[1:], tbox, lbl.shapes[img_i][0], lbl.shapes[img_i][1])
                                for row, box in zip(img_gt.tolist(), tbox.tolist()):
                                    plot_one_box(box, gt_img, label=_model_names[int(row[0])], line_thickness=2)
                            cv2.imwrite(os.path.join(_save_dir, f"batch_{self._eval_round}_{img_i}_det_gt.png"), gt_img)

                    # Per-image stats accumulation.
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
                                    ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)
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

        # Explicitly release the DataLoader and its worker process before
        # aggregating metrics. The worker is a fork of the server process; if
        # not killed promptly it holds a copy of the server's heap while the
        # server then allocates the stats tensors below, doubling peak RSS.
        del loader
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass

        # ---- Aggregate metrics ----
        map50 = 0.0
        if stats:
            np_stats = [np.concatenate(x, 0) for x in zip(*stats)]
            if len(np_stats) and np_stats[0].any():
                p, r, ap, f1, ap_class = ap_per_class(
                    *np_stats, plot=False, save_dir=".", names={}
                )
                ap50 = ap[:, 0]
                map50 = float(ap50.mean()) if ap50.size else 0.0

        if sample_count:
            da_iou = da_iou_sum / sample_count
            da_miou = da_miou_sum / sample_count
            ll_iou = ll_iou_sum / sample_count
            ll_miou = ll_miou_sum / sample_count
            valid_loss = val_loss_sum / sample_count if self._criterion is not None else None
            head_losses_avg = [v / sample_count for v in head_loss_sums] if self._criterion is not None else None
        else:
            da_iou = da_miou = ll_iou = ll_miou = 0.0
            valid_loss = None
            head_losses_avg = None

        context.state["valid_loss"] = valid_loss
        context.state["yolop_da_iou"] = float(da_iou)
        context.state["yolop_da_miou"] = float(da_miou)
        context.state["yolop_ll_iou"] = float(ll_iou)
        context.state["yolop_ll_miou"] = float(ll_miou)
        context.state["yolop_map50"] = float(map50)
        if head_losses_avg is not None:
            context.state["yolop_head_losses_val"] = {
                "lbox": head_losses_avg[0],
                "lobj": head_losses_avg[1],
                "lcls": head_losses_avg[2],
                "lseg_da": head_losses_avg[3],
                "lseg_ll": head_losses_avg[4],
                "liou_ll": head_losses_avg[5],
            }

        logging.info(
            "[YOLOP eval] mAP@0.5=%.4f  DA-IoU=%.4f  DA-mIoU=%.4f  "
            "LL-IoU=%.4f  LL-mIoU=%.4f  valid_loss=%s  (n=%d)",
            map50, da_iou, da_miou, ll_iou, ll_miou,
            f"{valid_loss:.4f}" if valid_loss is not None else "N/A",
            sample_count,
        )

        return float(map50)
