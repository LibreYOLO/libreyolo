"""
Loss functions for YOLOX training.
"""

import torch
import torch.nn as nn


# =============================================================================
# YOLOX Losses
# =============================================================================


class IoULoss(nn.Module):
    """IoU loss for bounding box regression (YOLOX)."""

    def __init__(self, reduction="none", loss_type="iou"):
        """
        Args:
            reduction: Reduction mode ('none', 'mean', 'sum')
            loss_type: Type of IoU loss ('iou' or 'giou')
        """
        super().__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        """
        Compute IoU loss.

        Args:
            pred: Predicted boxes in cxcywh format (N, 4)
            target: Target boxes in cxcywh format (N, 4)

        Returns:
            IoU loss tensor
        """
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        # Convert cxcywh to xyxy for intersection calculation
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou**2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
