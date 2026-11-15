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
#
# Code based on VAN: https://github.com/Visual-Attention-Network/VAN-Classification/blob/main/models/van.py
# =============================================================================

import math
from functools import partial

import torch
import torch.nn as nn

from .local_timm.drop import DropPath
from .local_timm.layer_helpers import to_2tuple
from .norm import GroupNorm2d
from .local_timm.weight_init import trunc_normal_
from .conv import Conv2dBlock
from .utils import get_activation


class DWConv(nn.Module):
    def __init__(self, dim=768, kernel_size=3):
        super(DWConv, self).__init__()
        self.dwconv = Conv2dBlock(
            dim, dim, kernel_size, 1, kernel_size // 2, bias=True, groups=dim
        )

    def forward(self, x):
        x = self.dwconv(x)
        return x


class ActGLU(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation_function=None,
        drop=0.0,
        mlp_use_dw_conv=True,
        mlp_dw_kernel_size=3,
        mlp_in_kernel_size=1,
        mlp_out_kernel_size=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_g = Conv2dBlock(
            in_features,
            hidden_features,
            mlp_in_kernel_size,
            padding=mlp_in_kernel_size // 2,
        )
        self.fc1_x = Conv2dBlock(
            in_features,
            hidden_features,
            mlp_in_kernel_size,
            padding=mlp_in_kernel_size // 2,
        )
        self.dwconv_g = None
        self.dwconv_x = None
        if mlp_use_dw_conv:
            self.dwconv_g = DWConv(hidden_features, mlp_dw_kernel_size)
            self.dwconv_x = DWConv(hidden_features, mlp_dw_kernel_size)
        act = (
            get_activation("gelu")
            if activation_function is None
            else activation_function
        )
        self.act = act(inplace=True)
        self.fc2 = Conv2dBlock(
            hidden_features,
            out_features,
            mlp_out_kernel_size,
            padding=mlp_out_kernel_size // 2,
        )
        self.drop = nn.Dropout(drop)

        self.in_hid_factor = float(hidden_features) / in_features
        self.hid_out_factor = float(out_features) / hidden_features

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        if self.dwconv_g is not None:
            x_gate = self.dwconv_g(x_gate)
            x = self.dwconv_x(x)
        x = self.act(x_gate) * x
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossActGLU(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation_function=None,
        drop=0.0,
        mlp_use_dw_conv=True,
        mlp_dw_kernel_size=3,
        mlp_in_kernel_size=1,
        mlp_out_kernel_size=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.merge_fc_g = Conv2dBlock(2 * in_features, in_features, 1)

        self.fc1_g = Conv2dBlock(
            in_features,
            hidden_features,
            mlp_in_kernel_size,
            padding=mlp_in_kernel_size // 2,
        )
        self.fc1_y = Conv2dBlock(
            in_features,
            hidden_features,
            mlp_in_kernel_size,
            padding=mlp_in_kernel_size // 2,
        )
        self.dwconv_g = None
        self.dwconv_y = None
        if mlp_use_dw_conv:
            self.dwconv_g = DWConv(hidden_features, mlp_dw_kernel_size)
            self.dwconv_y = DWConv(hidden_features, mlp_dw_kernel_size)
        act = (
            get_activation("gelu")
            if activation_function is None
            else activation_function
        )
        self.act = act(inplace=True)
        self.fc2 = Conv2dBlock(
            hidden_features,
            out_features,
            mlp_out_kernel_size,
            padding=mlp_out_kernel_size // 2,
        )
        self.drop = nn.Dropout(drop)

        self.in_hid_factor = float(hidden_features) / in_features
        self.hid_out_factor = float(out_features) / hidden_features

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        xy = self.merge_fc_g(torch.cat([x, y], 1))
        xy_gate = self.fc1_g(xy)
        y = self.fc1_y(y)
        if self.dwconv_g is not None:
            xy_gate = self.dwconv_g(xy_gate)
            y = self.dwconv_y(y)
        x = self.act(xy_gate) * y
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerTransition(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size=3, stride=2, in_chans=64, embed_dim=64):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = Conv2dBlock(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        return x


class CGU(nn.Module):
    def __init__(
        self,
        dim,
        drop=0.0,
        drop_path=0.0,
        activation_function=None,
        norm_layer=partial(GroupNorm2d, num_groups=8),
        use_cross=False,
        mlp_ratio=4,
        mlp_use_dw_conv=True,
        mlp_dw_kernel_size=7,
        mlp_in_kernel_size=1,
        mlp_out_kernel_size=1,
        layer_scale_init_value=1e-2,
    ):
        super().__init__()
        self.use_cross = use_cross

        self.norm = norm_layer(num_channels=dim)
        hidden_dim = int(dim * mlp_ratio)
        self.conv_self = ActGLU(
            in_features=dim,
            hidden_features=hidden_dim,
            activation_function=activation_function,
            drop=drop,
            mlp_use_dw_conv=mlp_use_dw_conv,
            mlp_dw_kernel_size=mlp_dw_kernel_size,
            mlp_in_kernel_size=mlp_in_kernel_size,
            mlp_out_kernel_size=mlp_out_kernel_size,
        )
        if use_cross:
            self.conv_cross = CrossActGLU(
                in_features=dim,
                hidden_features=hidden_dim,
                activation_function=activation_function,
                drop=drop,
                mlp_use_dw_conv=mlp_use_dw_conv,
                mlp_dw_kernel_size=mlp_dw_kernel_size,
                mlp_in_kernel_size=mlp_in_kernel_size,
                mlp_out_kernel_size=mlp_out_kernel_size,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if layer_scale_init_value < 1e-4:
            self.layer_scale = None
        else:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y=None):
        if self.use_cross:
            x_short = x.clone()
            y_short = y.clone()
            x = self.norm(x)
            y = self.norm(y)

            x = self.conv_self(x)
            y = self.conv_self(y)

            x = self.conv_cross(x, y)
            if self.layer_scale is not None:
                x = x * self.layer_scale[: x.shape[1]].unsqueeze(-1).unsqueeze(-1)
            x = self.drop_path(x)
            x = x + x_short

            y = self.conv_cross(y, x)
            if self.layer_scale is not None:
                y = y * self.layer_scale[: y.shape[1]].unsqueeze(-1).unsqueeze(-1)
            y = self.drop_path(y)
            y = y + y_short
        else:
            x_short = x.clone()
            x = self.norm(x)
            x = self.conv_self(x)
            x = x * self.layer_scale[: x.shape[1]].unsqueeze(-1).unsqueeze(-1)
            x = self.drop_path(x)
            x = x + x_short
        return x, y


class CGUStage(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        stride=2,
        drop=0.0,
        drop_path=0.0,
        activation_function=None,
        norm_layer=partial(GroupNorm2d, num_groups=8),
        depth=2,
        use_cross=False,
        mlp_ratio=4,
        mlp_use_dw_conv=True,
        mlp_dw_kernel_size=7,
        mlp_in_kernel_size=1,
        mlp_out_kernel_size=1,
        layer_scale_init_value=1e-2,
    ):
        super(CGUStage, self).__init__()
        self.conv_transition = None
        self.use_cross = use_cross
        if stride > 1 or in_chs != out_chs:
            patch_size = 1
            if stride > 1:
                patch_size = 3
            self.conv_transition = LayerTransition(
                patch_size=patch_size, stride=stride, in_chans=in_chs, embed_dim=out_chs
            )

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                CGU(
                    dim=out_chs,
                    drop=drop,
                    drop_path=drop_path,
                    activation_function=activation_function,
                    norm_layer=norm_layer,
                    use_cross=use_cross,
                    mlp_ratio=mlp_ratio,
                    mlp_use_dw_conv=mlp_use_dw_conv,
                    mlp_dw_kernel_size=mlp_dw_kernel_size,
                    mlp_in_kernel_size=mlp_in_kernel_size,
                    mlp_out_kernel_size=mlp_out_kernel_size,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
        self.norm = norm_layer(num_channels=out_chs)

    def forward(self, x, y=None, skip_transition=False):
        if self.conv_transition is not None and not skip_transition:
            x = self.conv_transition(x)
            if self.use_cross:
                y = self.conv_transition(y)
        for blk in self.blocks:
            x, y = blk(x, y)
        x = self.norm(x)
        if self.use_cross:
            y = self.norm(y)
            return x, y
        return x
