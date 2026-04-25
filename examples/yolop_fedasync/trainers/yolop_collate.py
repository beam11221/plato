"""
Collate function bridging YOLOP's 4-tuple batch format into the
(examples, labels) 2-tuple that Plato's training loop unpacks.

YOLOP's BddDataset.collate_fn returns:
    (imgs, [det_labels, da_seg, ll_seg], paths, shapes)

Plato's loop iterates `for examples, labels in train_loader:` so we must
collapse to two elements. The path strings are dropped (never used by the
loss or metrics); `shapes` is smuggled through MultiTaskLabels because the
loss reads `shapes[0][1][1]` for lane-IoU padding.
"""

from __future__ import annotations

import sys
import pathlib

# Ensure YOLOP's `lib.*` packages are importable. Done once at module load.
# This file lives at examples/yolop_fedasync/trainers/yolop_collate.py;
# parents[4] resolves to /workspace, so YOLOP sits at parents[4]/YOLOP.
_YOLOP_ROOT = pathlib.Path(__file__).resolve().parents[4] / "YOLOP"
if str(_YOLOP_ROOT) not in sys.path:
    sys.path.insert(0, str(_YOLOP_ROOT))

from lib.dataset.AutoDriveDataset import AutoDriveDataset

from .multitask_labels import MultiTaskLabels


def yolop_collate_fn(batch):
    """Wrap YOLOP's collate output into Plato's (examples, labels) shape.

    Reuses AutoDriveDataset.collate_fn verbatim — no risk of subtle bugs
    in the variable-length detection-label batching (it prepends the
    image index to column 0 of each row).
    """
    imgs, labels, _paths, shapes = AutoDriveDataset.collate_fn(batch)
    det, da_seg, ll_seg = labels
    return imgs, MultiTaskLabels(det, da_seg, ll_seg, shapes=shapes)
