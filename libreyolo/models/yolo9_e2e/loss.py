"""Loss functions for YOLOv9 end-to-end (NMS-free) training."""

from typing import Dict, List, Optional

import torch
from torch import Tensor

from ..yolo9.loss import YOLO9Loss


class YOLO9E2ELoss:
    """Combined one-to-many + one-to-one loss for NMS-free training.

    The one-to-many branch uses TaskAlignedAssigner with topk=10 (dense
    supervision). The one-to-one branch uses topk=1 (exclusive assignment)
    so that each ground-truth box is claimed by exactly one prediction,
    enabling NMS-free inference via top-K selection.
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
    ):
        self.one2many = YOLO9Loss(
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
        )
        self.one2one = YOLO9Loss(
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
        )

    def update_anchors(self, image_size: List[int]):
        """Update anchor grids for both branches."""
        self.one2many.update_anchors(image_size)
        self.one2one.update_anchors(image_size)

    def __call__(
        self,
        preds_one2many,
        preds_one2one,
        targets,
    ) -> Dict[str, Tensor]:
        """Compute the summed dual-branch loss."""
        loss_many = self.one2many(preds_one2many, targets)
        loss_one = self.one2one(preds_one2one, targets)

        total_loss = loss_many["total_loss"] + loss_one["total_loss"]
        box_loss = loss_many["box_loss"] + loss_one["box_loss"]
        dfl_loss = loss_many["dfl_loss"] + loss_one["dfl_loss"]
        cls_loss = loss_many["cls_loss"] + loss_one["cls_loss"]

        num_fg = loss_many.get("num_fg", 0) + loss_one.get("num_fg", 0)
        if isinstance(num_fg, Tensor):
            num_fg = num_fg.item()

        return {
            "total_loss": total_loss,
            "box_loss": box_loss,
            "dfl_loss": dfl_loss,
            "cls_loss": cls_loss,
            "box": box_loss.item() if isinstance(box_loss, Tensor) else box_loss,
            "dfl": dfl_loss.item() if isinstance(dfl_loss, Tensor) else dfl_loss,
            "cls": cls_loss.item() if isinstance(cls_loss, Tensor) else cls_loss,
            "num_fg": num_fg,
        }
