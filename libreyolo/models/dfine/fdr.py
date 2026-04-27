"""Fine-grained Distribution Refinement (FDR) primitives.

Ported from D-FINE (https://github.com/Peterande/D-FINE).
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

This module holds the numerical core of D-FINE's regression head:
- ``weighting_function``: non-uniform W(n) sequence over ``reg_max+1`` bins.
- ``Integral``: softmax-over-bins integrated against ``W(n)``.
- ``distance2bbox``: decode predicted per-edge offsets into a box.

Kept isolated from the decoder so it can be unit-tested for numerical parity
against the reference implementation independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_ops import box_xyxy_to_cxcywh


def weighting_function(reg_max, up, reg_scale, deploy=False):
    """Generate the non-uniform Weighting Function W(n) for box regression.

    Args:
        reg_max: Max number of the discrete bins (produces ``reg_max + 1`` values).
        up: Tensor controlling upper bounds of the sequence; maximum offset is
            ``±up * H/W``.
        reg_scale: Controls the curvature. Larger values -> flatter weights near
            the center W(reg_max/2)=0 and steeper weights at both ends.
        deploy: If True, pre-compute values as Python floats for tracing.
    """
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.tensor(values, dtype=up.dtype, device=up.device)
    else:
        upper_bound1 = abs(up[0]) * abs(reg_scale)
        upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.cat(values, 0)


def distance2bbox(points, distance, reg_scale):
    """Decode edge-distances into cxcywh boxes.

    Args:
        points: (B, N, 4) or (N, 4) representing [x, y, w, h] (center + size).
        distance: (B, N, 4) or (N, 4) distances from the point to left/top/right/bottom.
        reg_scale: curvature parameter of W(n) (scalar).

    Returns: boxes in cxcywh format, same leading shape as ``points``.
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (
        points[..., 2] / reg_scale
    )
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (
        points[..., 3] / reg_scale
    )
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (
        points[..., 2] / reg_scale
    )
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (
        points[..., 3] / reg_scale
    )

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    return box_xyxy_to_cxcywh(bboxes)


class Integral(nn.Module):
    """Softmax-weighted sum over bins: ``sum_n Pr(n) * W(n)``.

    Input: ``(..., 4 * (reg_max + 1))`` logits.
    Output: ``(..., 4)`` continuous offsets.
    """

    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


def translate_gt(gt, reg_max, reg_scale, up):
    """Decode GT distances to (closest_bin, weight_left, weight_right) bin indices.

    Used by the FGL loss to convert continuous edge-distance targets into a
    soft assignment over two adjacent bins of W(n).
    """
    gt = gt.reshape(-1)
    function_values = weighting_function(reg_max, up, reg_scale)

    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    mask = diffs <= 0
    closest_left_indices = torch.sum(mask, dim=1) - 1

    indices = closest_left_indices.float()

    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)

    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid_idx_mask].long()

    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]

    left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
    right_diffs = torch.abs(right_values - gt[valid_idx_mask])

    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

    invalid_idx_mask_neg = indices < 0
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0

    invalid_idx_mask_pos = indices >= reg_max
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1

    return indices, weight_right, weight_left


def bbox2distance(points, bbox, reg_max, reg_scale, up, eps=0.1):
    """Encode boxes as per-edge distance distributions for FGL targets.

    Args:
        points: (n, 4) [x, y, w, h] reference points.
        bbox: (n, 4) target boxes in xyxy format.
        reg_max: Max bin count.
        reg_scale: W(n) curvature.
        up: W(n) upper-bound parameter.
        eps: Small offset to ensure target < reg_max.

    Returns:
        (target_corners flat, weight_right, weight_left) — all 1-D tensors.
    """
    reg_scale = abs(reg_scale)
    left = (points[:, 0] - bbox[:, 0]) / (
        points[..., 2] / reg_scale + 1e-16
    ) - 0.5 * reg_scale
    top = (points[:, 1] - bbox[:, 1]) / (
        points[..., 3] / reg_scale + 1e-16
    ) - 0.5 * reg_scale
    right = (bbox[:, 2] - points[:, 0]) / (
        points[..., 2] / reg_scale + 1e-16
    ) - 0.5 * reg_scale
    bottom = (bbox[:, 3] - points[:, 1]) / (
        points[..., 3] / reg_scale + 1e-16
    ) - 0.5 * reg_scale
    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(
        four_lens, reg_max, reg_scale, up
    )
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max - eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()
