"""ECDet decoder helpers.

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

from ..dfine.box_ops import box_xyxy_to_cxcywh


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    return float(-math.log((1 - prior_prob) / prior_prob))


def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: List[int],
    method: str = "default",
    value_shape: str = "default",
):
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
