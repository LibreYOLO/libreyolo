"""Helpers ported from RT-DETRv2's ``src/zoo/rtdetr/utils.py``."""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    return float(-math.log((1 - prior_prob) / prior_prob))


def get_activation(act, inplace: bool = True):
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act
    name = act.lower()
    if name in ("silu", "swish"):
        m = nn.SiLU()
    elif name == "relu":
        m = nn.ReLU()
    elif name == "leaky_relu":
        m = nn.LeakyReLU()
    elif name == "gelu":
        m = nn.GELU()
    elif name == "hardsigmoid":
        m = nn.Hardsigmoid()
    else:
        raise RuntimeError(f"Unknown activation: {act!r}")
    if hasattr(m, "inplace"):
        m.inplace = inplace
    return m


def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: List[int],
    method: str = "default",
):
    """v2 deformable attention core.

    Differs from v1 in: (a) flat ``sum(num_points_list)`` layout instead of
    ``[L, P]``; (b) per-level split via ``num_points_list``; (c) optional
    ``method='discrete'`` integer-index sampling for TensorRT <8.5.
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, _, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)

    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations
    else:
        raise ValueError(f"unknown sampling method: {method!r}")

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l = sampling_locations_list[level]

        if method == "default":
            sampling_value_l = F.grid_sample(
                value_l,
                sampling_grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        else:  # discrete
            sampling_coord = (
                sampling_grid_l * torch.tensor([[w, h]], device=value.device) + 0.5
            ).to(torch.int64)
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(
                bs * n_head, Len_q * num_points_list[level], 2
            )
            s_idx = torch.arange(
                sampling_coord.shape[0], device=value.device
            ).unsqueeze(-1).repeat(1, sampling_coord.shape[1])
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
