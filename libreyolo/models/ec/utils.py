"""EC decoder helpers.

Ported from EdgeCrafter (Apache-2.0). Functions kept byte-equivalent to upstream
to lock in numerical parity for the decoder + criterion. Only the import of
``box_xyxy_to_cxcywh`` is retargeted to LibreYOLO's D-FINE box_ops (already
proven equivalent).
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...kernels.attention.ms_deform_attn import (
    maybe_ms_deform_attn_v2,
    ms_deform_attn_available,
    spatial_shapes_tensor,
)
from ..dfine.box_ops import box_xyxy_to_cxcywh


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    return float(-math.log((1 - prior_prob) / prior_prob))


def _grid_sample_bilinear_manual(feat, grid, padding_mode="zeros", align_corners=False):
    """ONNX/TRT-safe bilinear grid_sample: gather-based, emits NO GridSample op.

    Numerically matches F.grid_sample(mode='bilinear'); sidesteps the TensorRT
    GridSample kernel that miscomputes boundary samples for deformable attention.
    Batch is kept symbolic (expand(-1)/unflatten) so dynamic-batch exports trace
    correctly.
    """
    _, channels, height, width = feat.shape
    grid_h, grid_w = grid.shape[1], grid.shape[2]
    if align_corners:
        ix = (grid[..., 0] + 1) * (width - 1) / 2
        iy = (grid[..., 1] + 1) * (height - 1) / 2
    else:
        ix = (grid[..., 0] + 1) * width / 2 - 0.5
        iy = (grid[..., 1] + 1) * height / 2 - 0.5
    ix0 = ix.floor().long()
    iy0 = iy.floor().long()
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    wx1 = (ix - ix0.float()).to(feat.dtype).unsqueeze(1)
    wy1 = (iy - iy0.float()).to(feat.dtype).unsqueeze(1)
    one = wx1.new_tensor(1.0)
    wx0 = one - wx1
    wy0 = one - wy1
    if padding_mode == "border":
        ix0 = ix0.clamp(0, width - 1)
        iy0 = iy0.clamp(0, height - 1)
        ix1 = ix1.clamp(0, width - 1)
        iy1 = iy1.clamp(0, height - 1)
    else:
        in_x0 = (ix0 >= 0) & (ix0 < width)
        in_x1 = (ix1 >= 0) & (ix1 < width)
        in_y0 = (iy0 >= 0) & (iy0 < height)
        in_y1 = (iy1 >= 0) & (iy1 < height)
        ix0 = ix0.clamp(0, width - 1)
        iy0 = iy0.clamp(0, height - 1)
        ix1 = ix1.clamp(0, width - 1)
        iy1 = iy1.clamp(0, height - 1)
    flat = feat.flatten(2)

    def _gather(iy_, ix_):
        idx = (iy_ * width + ix_).flatten(1).unsqueeze(1).expand(-1, channels, -1)
        return flat.gather(2, idx).unflatten(2, (grid_h, grid_w))

    v00 = _gather(iy0, ix0)
    v10 = _gather(iy0, ix1)
    v01 = _gather(iy1, ix0)
    v11 = _gather(iy1, ix1)
    if padding_mode == "zeros":
        v00 = v00 * (in_x0 & in_y0).unsqueeze(1)
        v10 = v10 * (in_x1 & in_y0).unsqueeze(1)
        v01 = v01 * (in_x0 & in_y1).unsqueeze(1)
        v11 = v11 * (in_x1 & in_y1).unsqueeze(1)
    return wx0 * wy0 * v00 + wx1 * wy0 * v10 + wx0 * wy1 * v01 + wx1 * wy1 * v11


def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: List[int],
    method: str = "default",
    value_shape: str = "default",
):
    """Multi-scale deformable attention aggregator (grid_sample variant).

    The loop below is the default and the export path. When the optional
    accelerated ``ms_deform_attn`` slot resolves and the flat point layout
    reshapes onto the slot's ``(n_levels, n_points)`` one, it takes over
    (see ``libreyolo/kernels/attention/ms_deform_attn.py``);
    ``method='discrete'`` never does, its integer-index sampling is a
    different equation.
    """
    hint_value = value[0] if value_shape == "default" else value
    if method == "default" and ms_deform_attn_available(hint_value):
        if value_shape == "default":
            # Per-level (bs, n_head, c, H*W) -> the slot's (bs, Len_in, n_head, c).
            flat_value = torch.cat(list(value), dim=-1).permute(0, 3, 1, 2)
        else:
            flat_value = value
        accelerated = maybe_ms_deform_attn_v2(
            flat_value,
            spatial_shapes_tensor(value_spatial_shapes, flat_value.device),
            sampling_locations,
            attention_weights,
            num_points_list,
        )
        if accelerated is not None:
            return accelerated
    if value_shape == "default":
        bs, n_head, c, _ = value[0].shape
    elif value_shape == "reshape":
        bs, _, n_head, c = value.shape
        split_shape = [h * w for h, w in value_spatial_shapes]
        value = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)

    _, Len_q, _, _, _ = sampling_locations.shape

    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l = sampling_locations_list[level]

        if method == "default":
            if torch.onnx.is_in_onnx_export():
                sampling_value_l = _grid_sample_bilinear_manual(
                    value_l, sampling_grid_l, padding_mode="zeros", align_corners=False
                )
            else:
                sampling_value_l = F.grid_sample(
                    value_l,
                    sampling_grid_l,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
        elif method == "discrete":
            sampling_coord = (
                sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5
            ).to(torch.int64)
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(
                bs * n_head, Len_q * num_points_list[level], 2
            )
            s_idx = (
                torch.arange(sampling_coord.shape[0], device=value_l.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l = value_l[
                s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]
            ]
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                bs * n_head, c, Len_q, num_points_list[level]
            )

        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        bs * n_head, 1, Len_q, sum(num_points_list)
    )
    weighted = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted.sum(-1).reshape(bs, n_head * c, Len_q)
    return output.permute(0, 2, 1)


def weighting_function(reg_max: int, up: torch.Tensor, reg_scale, deploy: bool = False):
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


def distance2pose(
    points: torch.Tensor, distance: torch.Tensor, reg_scale
) -> torch.Tensor:
    """Decode per-keypoint distance offsets onto reference points.

    Used by the pose decoder. Mirrors upstream's
    ``ecpose/engine/edgecrafter/detrpose_transformer.distance2pose``.
    """
    reg_scale = abs(reg_scale)
    x = points[..., 0] + distance[..., 0] / reg_scale
    y = points[..., 1] + distance[..., 1] / reg_scale
    return torch.stack([x, y], -1)


def distance2bbox(
    points: torch.Tensor, distance: torch.Tensor, reg_scale
) -> torch.Tensor:
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
    return box_xyxy_to_cxcywh(torch.stack([x1, y1, x2, y2], -1))


def get_activation(act, inplace: bool = True):
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act

    act = act.lower()
    if act in ("silu", "swish"):
        m = nn.SiLU()
    elif act == "relu":
        m = nn.ReLU()
    elif act == "leaky_relu":
        m = nn.LeakyReLU()
    elif act == "gelu":
        m = nn.GELU()
    elif act == "hardsigmoid":
        m = nn.Hardsigmoid()
    else:
        raise RuntimeError(f"unknown act: {act}")
    if hasattr(m, "inplace"):
        m.inplace = inplace
    return m
