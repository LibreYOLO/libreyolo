# =============================================================================
# Copyright 2025 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import GroupNorm2d, LayerNorm2d, BatchNorm2d
from .local_timm.gelu import GELU


def compute_pyramid_levels(x):
    img_diag = math.sqrt((x.shape[-2] ** 2) + (x.shape[-1] ** 2))
    input_factor = max(
        1, img_diag / 1100
    )  # 1100 ~= math.sqrt((960 ** 2) + (540 ** 2)), i.e., 1K resolution
    pyr_levels = int(round(math.log2(input_factor))) + 3
    return pyr_levels


def bilinear_sampler(img, coords, mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def get_activation(name):
    if name == "relu":
        return nn.ReLU
    elif name == "gelu":
        return GELU
    elif name == "silu":
        return nn.SiLU
    elif name == "mish":
        return nn.Mish
    elif name == "linear":
        return nn.Identity
    else:
        return None


def get_norm(name, affine=False, num_groups=8):
    if name == "group":
        return partial(GroupNorm2d, affine=affine, num_groups=num_groups)
    elif name == "layer":
        return partial(LayerNorm2d, affine=affine)
    elif name == "batch":
        return BatchNorm2d
    else:
        return None
