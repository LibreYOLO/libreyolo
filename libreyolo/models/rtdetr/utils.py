"""
Utility functions for LibreYOLO RTDETR.

Provides preprocessing, postprocessing, and core attention functions.
"""

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Core Utilities (used by nn.py, denoising.py, loss.py)
# =============================================================================

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse of sigmoid function, clamped for numerical stability."""
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    """Initialize conv/fc bias value according to a given probability."""
    return float(-math.log((1 - prior_prob) / prior_prob))


def get_activation(act: str, inplace: bool = True) -> nn.Module:
    """Get activation module by name."""
    if act is None:
        return nn.Identity()
    
    act = act.lower()
    if act == 'silu':
        m = nn.SiLU()
    elif act == 'relu':
        m = nn.ReLU()
    elif act == 'leaky_relu':
        m = nn.LeakyReLU()
    elif act == 'gelu':
        m = nn.GELU()
    else:
        raise RuntimeError(f'Unknown activation: {act}')

    if hasattr(m, 'inplace'):
        m.inplace = inplace

    return m


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Pure PyTorch implementation of multi-scale deformable attention core.
    
    Uses F.grid_sample for bilinear interpolation at sampling locations.
    
    Args:
        value: [bs, value_length, n_head, c]
        value_spatial_shapes: List of [h, w] for each level
        sampling_locations: [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights: [bs, query_length, n_head, n_levels, n_points]
    
    Returns:
        output: [bs, query_length, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # [bs, H*W, n_head, c] -> [bs, n_head*c, H, W]
        value_l_ = value_list[level].flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)
        # [bs, Lq, n_head, n_points, 2] -> [bs*n_head, Lq, n_points, 2]
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
        # [bs*n_head, c, Lq, n_points]
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)

    # [bs, Lq, n_head, n_levels, n_points] -> [bs*n_head, 1, Lq, n_levels*n_points]
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points)
    
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


# =============================================================================
# Preprocessing for backends (used by model.py)
# =============================================================================

def preprocess_numpy(img_rgb_hwc: np.ndarray, input_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Resize and normalize image for RTDETR inference. Returns float32 NCHW array.

    Args:
        img_rgb_hwc: Input image in RGB format, HWC layout
        input_size: Target input size (square)

    Returns:
        Tuple of (preprocessed image as float32 NCHW array, original size (h, w))
    """
    import cv2
    h, w = img_rgb_hwc.shape[:2]
    img = cv2.resize(img_rgb_hwc, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW
    return img, (h, w)