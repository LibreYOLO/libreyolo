"""Loss functions for YOLOv9 end-to-end (NMS-free) training.

Provenance: each branch is scored by one instance of the MIT-derived
:class:`~libreyolo.models.yolo9.loss.YOLO9Loss` (MultimediaTechLab/YOLO,
https://github.com/MultimediaTechLab/YOLO, commit c4cb5f6f, MIT License).
The one-to-many branch uses matcher ``topk: 10`` from MultimediaTechLab's
``yolo/config/task/train.yaml``; the one-to-one branch uses the same matcher
with top-1. The two branch losses are summed, like the two heads in
MultimediaTechLab's ``DualLoss`` (``yolo/tools/loss_functions.py``). Training
a one-to-many and a one-to-one predictor together and summing their losses
follows DATE (https://github.com/YiqunChen1999/date, commit 5daf092c,
Apache-2.0).
"""

from typing import Dict, List, Optional

import torch
from torch import Tensor

from ..yolo9.loss import YOLO9Loss


class YOLO9E2ELoss:
    """Combined one-to-many + one-to-one loss for NMS-free training.

    Two :class:`YOLO9Loss` instances with the same weights and matching
    metric: the dense (one-to-many) branch lets the ``BoxMatcher`` keep the
    top 10 anchors per ground-truth box, the exclusive (one-to-one) branch
    keeps the top 1, so each box is claimed by exactly one prediction and
    inference can use top-K selection instead of NMS. The totals and the
    box / DFL / class components are summed across the two branches.
    """

    def __init__(
        self,
        num_classes: int,
        reg_max: int,
        strides: List[int],
        image_size: Optional[List[int]],
        device: torch.device,
        box_weight: float = 7.5,
        dfl_weight: float = 1.5,
        cls_weight: float = 0.5,
        topk_many: int = 10,
        topk_one: int = 1,
        iou_factor: float = 6.0,
        cls_factor: float = 0.5,
        distributed_normalize: bool = True,
    ):
        self.dense_loss = YOLO9Loss(
            num_classes=num_classes,
            reg_max=reg_max,
            strides=strides,
            image_size=image_size,
            device=device,
            box_weight=box_weight,
            dfl_weight=dfl_weight,
            cls_weight=cls_weight,
            topk=topk_many,
            iou_factor=iou_factor,
            cls_factor=cls_factor,
            distributed_normalize=distributed_normalize,
        )
        self.exclusive_loss = YOLO9Loss(
            num_classes=num_classes,
            reg_max=reg_max,
            strides=strides,
            image_size=image_size,
            device=device,
            box_weight=box_weight,
            dfl_weight=dfl_weight,
            cls_weight=cls_weight,
            topk=topk_one,
            iou_factor=iou_factor,
            cls_factor=cls_factor,
            distributed_normalize=distributed_normalize,
        )

    def update_anchors(self, image_size: List[int]):
        """Update anchor grids for both branches."""
        self.dense_loss.update_anchors(image_size)
        self.exclusive_loss.update_anchors(image_size)

    def __call__(
        self,
        dense_preds,
        exclusive_preds,
        targets,
    ) -> Dict[str, Tensor]:
        """Compute the summed dual-branch loss."""
        loss_many = self.dense_loss(dense_preds, targets)
        loss_one = self.exclusive_loss(exclusive_preds, targets)

        total_loss = loss_many["total_loss"] + loss_one["total_loss"]
        box_loss = loss_many["box_loss"] + loss_one["box_loss"]
        dfl_loss = loss_many["dfl_loss"] + loss_one["dfl_loss"]
        cls_loss = loss_many["cls_loss"] + loss_one["cls_loss"]

        # Logging values stay detached 0-dim tensors: the per-key ``.item()``
        # calls here were GPU pipeline drains every step (issue #763). Float
        # conversion happens once, batched, in ``get_loss_components``.
        num_fg = loss_many.get("num_fg", 0) + loss_one.get("num_fg", 0)
        if isinstance(num_fg, Tensor):
            num_fg = num_fg.detach()

        return {
            "total_loss": total_loss,
            "box_loss": box_loss,
            "dfl_loss": dfl_loss,
            "cls_loss": cls_loss,
            "box": box_loss.detach(),
            "dfl": dfl_loss.detach(),
            "cls": cls_loss.detach(),
            "num_fg": num_fg,
        }
