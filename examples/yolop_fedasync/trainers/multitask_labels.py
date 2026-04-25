"""
MultiTaskLabels — a label container that satisfies Plato's single-tensor
label assumption while carrying YOLOP's three task-specific tensors plus
the per-image shape metadata that the loss needs for coord rescaling.

Plato's training loop calls `labels.to(device)` and `labels.size(0)` on
whatever the data loader yields as the second tuple element. By exposing
those two methods we slip a structured payload through unchanged.
"""

from __future__ import annotations

import torch


class MultiTaskLabels:
    """Container for YOLOP's three label tensors.

    Attributes:
        det:    Detection labels, shape (N, 6) where columns are
                [batch_idx, class, x_center, y_center, w, h] (normalized).
                N is the total number of bounding boxes across the batch.
        da_seg: Drivable-area segmentation masks, shape (B, 2, H, W).
        ll_seg: Lane-line segmentation masks, shape (B, 2, H, W).
        shapes: Per-image padding/scale metadata from the dataset; used by
                the loss for lane-IoU coordinate cropping. CPU-only —
                never moved to the device.
    """

    __slots__ = ("det", "da_seg", "ll_seg", "shapes")

    def __init__(
        self,
        det: torch.Tensor,
        da_seg: torch.Tensor,
        ll_seg: torch.Tensor,
        shapes=None,
    ) -> None:
        self.det = det
        self.da_seg = da_seg
        self.ll_seg = ll_seg
        self.shapes = shapes

    def to(self, device, *args, **kwargs) -> "MultiTaskLabels":
        """Move all tensor components to `device`. Shapes stays on CPU.

        We accept *args/**kwargs so calls like .to(device, non_blocking=True)
        from Plato or DataLoader pin-memory workers don't break.
        """
        return MultiTaskLabels(
            self.det.to(device, *args, **kwargs),
            self.da_seg.to(device, *args, **kwargs),
            self.ll_seg.to(device, *args, **kwargs),
            self.shapes,
        )

    def size(self, dim=None):
        """Mirror torch.Tensor.size(): return batch dim from da_seg.

        Plato calls labels.size(0) at composable.py:555 to update the loss
        tracker's running average. da_seg has shape (B, 2, H, W), so
        size(0) returns B — the actual batch size.
        """
        if dim is None:
            return self.da_seg.size()
        return self.da_seg.size(dim)

    def pin_memory(self) -> "MultiTaskLabels":
        """Required by DataLoader when pin_memory=True. Pin tensor parts only."""
        return MultiTaskLabels(
            self.det.pin_memory(),
            self.da_seg.pin_memory(),
            self.ll_seg.pin_memory(),
            self.shapes,
        )

    def __repr__(self) -> str:
        return (
            f"MultiTaskLabels(det={tuple(self.det.shape)}, "
            f"da_seg={tuple(self.da_seg.shape)}, "
            f"ll_seg={tuple(self.ll_seg.shape)})"
        )
