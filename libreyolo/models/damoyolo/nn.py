"""DAMO-YOLO nn.Module classes, per-size architecture table, and factory.

Class hierarchy and attribute names mirror upstream
(github.com/tinyvision/DAMO-YOLO, Apache-2.0) so upstream `.pth` checkpoints
load directly via ``state_dict``.

Layout:
- nn.Module classes (backbones, neck, head, ops) — bulk of the file
- ``FamilyConfig`` dataclass + per-size ``SIZES`` table near the end
- ``build_damoyolo(size, num_classes)`` factory at the bottom
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activations / norms
# ---------------------------------------------------------------------------


def get_activation(name: str = "silu", inplace: bool = True) -> nn.Module:
    if name is None:
        return nn.Identity()
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "lrelu":
        return nn.LeakyReLU(0.1, inplace=inplace)
    if name == "hardsigmoid":
        return nn.Hardsigmoid(inplace=inplace)
    if name == "identity":
        return nn.Identity()
    raise AttributeError(f"Unsupported act type: {name}")


def get_norm(name: str, out_channels: int) -> nn.Module:
    if name == "bn":
        return nn.BatchNorm2d(out_channels)
    if name == "gn":
        return nn.GroupNorm(out_channels, out_channels)
    raise NotImplementedError(name)


# ---------------------------------------------------------------------------
# Core convolution blocks
# ---------------------------------------------------------------------------


class ConvBNAct(nn.Module):
    """Conv2d → BN → activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        act: str = "silu",
        norm: str = "bn",
        reparam: bool = False,
    ) -> None:
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.with_norm = norm is not None
        self.with_act = act is not None
        if self.with_norm:
            self.bn = get_norm(norm, out_channels)
        if self.with_act:
            self.act = get_activation(act, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x


class SPPBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Tuple[int, int, int] = (5, 9, 13),
        activation: str = "silu",
    ) -> None:
        super().__init__()
        hidden = in_channels // 2
        self.conv1 = ConvBNAct(in_channels, hidden, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]
        )
        conv2_in = hidden * (len(kernel_sizes) + 1)
        self.conv2 = ConvBNAct(conv2_in, out_channels, 1, stride=1, act=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        return self.conv2(x)


class Focus(nn.Module):
    """Reduces H/W by 2 by interleaving 4 spatial offsets into channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int = 1,
        stride: int = 1,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.conv = ConvBNAct(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tl = x[..., ::2, ::2]
        tr = x[..., ::2, 1::2]
        bl = x[..., 1::2, ::2]
        br = x[..., 1::2, 1::2]
        return self.conv(torch.cat((tl, bl, tr, br), dim=1))


def _conv_bn(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, groups: int = 1) -> nn.Sequential:
    seq = nn.Sequential()
    seq.add_module(
        "conv",
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
    )
    seq.add_module("bn", nn.BatchNorm2d(out_channels))
    return seq


class RepConv(nn.Module):
    """Re-parameterizable 3x3 conv (RepVGG block)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        deploy: bool = False,
        act: str = "relu",
        norm=None,
    ) -> None:
        super().__init__()
        assert kernel_size == 3 and padding == 1
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.nonlinearity = get_activation(act) if isinstance(act, str) else act

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )
        else:
            self.rbr_identity = None
            self.rbr_dense = _conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups)
            self.rbr_1x1 = _conv_bn(in_channels, out_channels, 1, stride, padding - kernel_size // 2, groups)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(inputs))
        id_out = 0 if self.rbr_identity is None else self.rbr_identity(inputs)
        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    # ---- reparam fusion -------------------------------------------------

    def get_equivalent_kernel_bias(self):
        k3, b3 = self._fuse_bn_tensor(self.rbr_dense)
        k1, b1 = self._fuse_bn_tensor(self.rbr_1x1)
        kid, bid = self._fuse_bn_tensor(self.rbr_identity)
        return k3 + self._pad_1x1_to_3x3(k1) + kid, b3 + b1 + bid

    @staticmethod
    def _pad_1x1_to_3x3(kernel1x1):
        if isinstance(kernel1x1, int) and kernel1x1 == 0:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kv = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kv[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kv).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self) -> None:
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for p in self.parameters():
            p.detach_()
        del self.rbr_dense
        del self.rbr_1x1
        if hasattr(self, "rbr_identity"):
            del self.rbr_identity
        if hasattr(self, "id_tensor"):
            del self.id_tensor
        self.deploy = True


# ---------------------------------------------------------------------------
# MobileNet-style blocks (used by TinyNAS_mob and depthwise GiraffeNeckV2)
# ---------------------------------------------------------------------------


def make_divisible(v: float, divisor: int = 8, min_value: int = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Hsigmoid(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SEModule(nn.Module):
    def __init__(self, channel: int, reduction: int = 4) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def _depthwise_conv(i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False) -> nn.Conv2d:
    return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)


class MobileV3Block(nn.Module):
    """Backbone-flavor inverted-residual block.

    Mirrors ``damo/base_models/backbones/tinynas_mob.py::MobileV3Block``:
    9-slot Sequential (1x1 expand, BN, act, 5x5 depthwise, BN, SE-or-Identity,
    act, 1x1 project, BN). Variable expansion ratio via ``block_pos``.
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        btn_c=None,
        kernel_size: int = 5,
        stride: int = 1,
        act: str = "silu",
        reparam: bool = False,
        block_type: str = "k1kx",
        depthwise: bool = False,
        use_se: bool = False,
        block_pos=None,
    ) -> None:
        super().__init__()
        self.stride = stride
        exp_ratio = 2.5 if block_pos is None else 3.5 + (block_pos - 1) * 0.5
        branch = make_divisible(int(math.ceil(out_c * exp_ratio)))
        # Always materialize a slot at index 5: SEModule when use_se, else
        # nn.Identity. Upstream serializes the same way so both checkpoint
        # layouts (with/without SE) round-trip strict.
        se_layer = SEModule(branch) if use_se else nn.Identity()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, branch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch),
            get_activation(act),
            _depthwise_conv(branch, branch, kernel_size=5, stride=stride, padding=2),
            nn.BatchNorm2d(branch),
            se_layer,
            get_activation(act),
            nn.Conv2d(branch, out_c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.use_shotcut = stride == 1 and in_c == out_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.use_shotcut else self.conv(x)


class _MobileV3BlockNeck(nn.Module):
    """Neck-flavor inverted-residual block (no SE, fixed exp_ratio=3.0).

    Mirrors ``damo/base_models/core/ops.py::MobileV3Block``: 8-slot Sequential
    (1x1 expand, BN, act, 5x5 depthwise, BN, act, 1x1 project, BN). Used
    by ``BasicBlock_3x3_Reverse(depthwise=True)`` in GiraffeNeckV2.
    """

    def __init__(self, in_c: int, out_c: int, kernel_size: int = 5, stride: int = 1, act: str = "silu") -> None:
        super().__init__()
        self.stride = stride
        branch = make_divisible(int(math.ceil(out_c * 3.0)))
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, branch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch),
            get_activation(act),
            _depthwise_conv(branch, branch, kernel_size=5, stride=stride, padding=2),
            nn.BatchNorm2d(branch),
            get_activation(act),
            nn.Conv2d(branch, out_c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.use_shotcut = stride == 1 and in_c == out_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.use_shotcut else self.conv(x)


class DepthwiseConv(nn.Module):
    """Mob-style depthwise conv: dw conv → dwnorm → act → pointwise conv → pwnorm → act."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding=None,
        dilation: int = 1,
        bias=False,
        norm_cfg: str = "bn",
        act: str = "relu",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=in_channels, bias=bias,
        )
        self.dwnorm = get_norm(norm_cfg, in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.pwnorm = get_norm(norm_cfg, out_channels)
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.dwnorm(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.pwnorm(x)
        return self.act(x)


# ---------------------------------------------------------------------------
# Neck building blocks
# ---------------------------------------------------------------------------


class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_hidden_ratio: float,
        ch_out: int,
        act: str = "relu",
        shortcut: bool = True,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        assert ch_in == ch_out
        self.depthwise = depthwise
        if not depthwise:
            ch_hidden = int(ch_in * ch_hidden_ratio)
            self.conv1 = ConvBNAct(ch_hidden, ch_out, 3, stride=1, act=act)
            self.conv2 = RepConv(ch_in, ch_hidden, 3, stride=1, act=act)
        else:
            # Nano neck block: 8-slot _MobileV3BlockNeck (no SE, exp_ratio 3.0).
            self.conv = _MobileV3BlockNeck(in_c=ch_in, out_c=ch_out, kernel_size=5, stride=1, act=act)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.depthwise:
            return self.conv(x)
        y = self.conv2(x)
        y = self.conv1(y)
        return x + y if self.shortcut else y


class SPP(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, k: int, pool_size, act: str = "swish") -> None:
        super().__init__()
        self.pool: List[nn.MaxPool2d] = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2, ceil_mode=False)
            self.add_module(f"pool{i}", pool)
            self.pool.append(pool)
        self.conv = ConvBNAct(ch_in, ch_out, k, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [x] + [p(x) for p in self.pool]
        y = torch.cat(outs, dim=1)
        return self.conv(y)


class CSPStage(nn.Module):
    def __init__(
        self,
        block_fn: str,
        ch_in: int,
        ch_hidden_ratio: float,
        ch_out: int,
        n: int,
        act: str = "swish",
        spp: bool = False,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = ConvBNAct(ch_in, ch_first, 1, act=act)
        self.conv2 = ConvBNAct(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()
        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == "BasicBlock_3x3_Reverse":
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse(
                        next_ch_in, ch_hidden_ratio, ch_mid, act=act, shortcut=True, depthwise=depthwise,
                    ),
                )
            else:
                raise NotImplementedError(block_fn)
            if i == (n - 1) // 2 and spp:
                self.convs.add_module("spp", SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNAct(ch_mid * n + ch_first, ch_out, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        mids = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mids.append(y2)
        y = torch.cat(mids, dim=1)
        return self.conv3(y)


# ---------------------------------------------------------------------------
# TinyNAS backbone
# ---------------------------------------------------------------------------


class ConvKXBN(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_c, out_c, kernel_size, stride, (kernel_size - 1) // 2, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn1(self.conv1(x))


class ConvKXBNRELU(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int, stride: int, act: str = "silu") -> None:
        super().__init__()
        self.conv = ConvKXBN(in_c, out_c, kernel_size, stride)
        self.activation_function = get_activation(act) if act is not None else torch.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_function(self.conv(x))


class ResConvBlock(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        btn_c: int,
        kernel_size: int,
        stride: int,
        act: str = "silu",
        reparam: bool = False,
        block_type: str = "k1kx",
    ) -> None:
        super().__init__()
        self.stride = stride
        if block_type == "k1kx":
            self.conv1 = ConvKXBN(in_c, btn_c, kernel_size=1, stride=1)
        else:
            self.conv1 = ConvKXBN(in_c, btn_c, kernel_size=kernel_size, stride=1)

        if not reparam:
            self.conv2 = ConvKXBN(btn_c, out_c, kernel_size, stride)
        else:
            self.conv2 = RepConv(btn_c, out_c, kernel_size, stride, act="identity")

        self.activation_function = get_activation(act)
        if in_c != out_c and stride != 2:
            self.residual_proj = ConvKXBN(in_c, out_c, 1, 1)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reslink = self.residual_proj(x) if self.residual_proj is not None else x
        x = self.conv1(x)
        x = self.activation_function(x)
        x = self.conv2(x)
        if self.stride != 2:
            x = x + reslink
        return self.activation_function(x)


class SuperResStem(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        btn_c: int,
        kernel_size: int,
        stride: int,
        num_blocks: int,
        with_spp: bool = False,
        act: str = "silu",
        reparam: bool = False,
        block_type: str = "k1kx",
    ) -> None:
        super().__init__()
        self.act = get_activation(act) if act is not None else torch.relu
        self.block_list = nn.ModuleList()
        for block_id in range(num_blocks):
            if block_id == 0:
                in_channels, out_channels, this_stride, this_kernel = in_c, out_c, stride, kernel_size
            else:
                in_channels, out_channels, this_stride, this_kernel = out_c, out_c, 1, kernel_size
            self.block_list.append(
                ResConvBlock(
                    in_channels,
                    out_channels,
                    btn_c,
                    this_kernel,
                    this_stride,
                    act=act,
                    reparam=reparam,
                    block_type=block_type,
                )
            )
            if block_id == 0 and with_spp:
                self.block_list.append(SPPBottleneck(out_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block_list:
            x = block(x)
        return x


class CSPStem(nn.Module):
    """List of ``ResConvBlock``s without SPP (CSP backbone uses an outer wrapper for that)."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        btn_c: int,
        stride: int,
        kernel_size: int,
        num_blocks: int,
        act: str = "silu",
        reparam: bool = False,
        block_type: str = "k1kx",
    ) -> None:
        super().__init__()
        self.in_channels = in_c
        self.out_channels = out_c
        self.stride = stride
        # When the CSP wrapper handles downsampling, the stem itself runs at
        # stride 1; subtract one block so total layer count matches upstream.
        self.num_blocks = num_blocks - 1 if stride == 2 else num_blocks
        out_c_half = out_c // 2

        self.block_list = nn.ModuleList()
        for block_id in range(self.num_blocks):
            this_in = (in_c // 2) if (stride == 1 and block_id == 0) else out_c_half
            self.block_list.append(
                ResConvBlock(
                    this_in, out_c_half, btn_c, kernel_size, stride=1,
                    act=act, reparam=reparam, block_type=block_type,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.block_list:
            x = b(x)
        return x


class CSPWrapper(nn.Module):
    """Wraps one or two CSP stems with downsample + cross-stage shortcut + fuse."""

    def __init__(self, convstem, act: str = "relu", with_spp: bool = False) -> None:
        super().__init__()
        self.with_spp = with_spp
        if isinstance(convstem, tuple):
            in_c = convstem[0].in_channels
            out_c = convstem[-1].out_channels
            hidden_dim = convstem[0].out_channels // 2
            blocks = []
            for stem in convstem:
                for layer in stem.block_list:
                    blocks.append(layer)
        else:
            in_c = convstem.in_channels
            out_c = convstem.out_channels
            hidden_dim = out_c // 2
            blocks = list(convstem.block_list)

        self.convstem = nn.ModuleList(blocks)
        self.downsampler = ConvKXBNRELU(in_c, hidden_dim * 2, 3, 2, act=act)
        if self.with_spp:
            self.spp = SPPBottleneck(hidden_dim * 2, hidden_dim * 2)
        if len(self.convstem) > 0:
            self.conv_start = ConvKXBNRELU(hidden_dim * 2, hidden_dim, 1, 1, act=act)
            self.conv_shortcut = ConvKXBNRELU(hidden_dim * 2, out_c // 2, 1, 1, act=act)
            self.conv_fuse = ConvKXBNRELU(out_c, out_c, 1, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsampler(x)
        if self.with_spp:
            x = self.spp(x)
        if len(self.convstem) > 0:
            shortcut = self.conv_shortcut(x)
            x = self.conv_start(x)
            for b in self.convstem:
                x = b(x)
            x = torch.cat((x, shortcut), dim=1)
            x = self.conv_fuse(x)
        return x


class TinyNASCSP(nn.Module):
    """TinyNAS_csp backbone — used by DAMO-YOLO-M."""

    def __init__(
        self,
        structure_info,
        out_indices=(2, 3, 4),
        with_spp: bool = False,
        use_focus: bool = False,
        act: str = "silu",
        reparam: bool = False,
    ) -> None:
        super().__init__()
        self.out_indices = tuple(out_indices)

        # Build the 6 "raw" stems first (Focus/Conv stem + 5 CSPStems).
        raw = []
        for idx, info in enumerate(structure_info):
            cls = info["class"]
            if cls == "ConvKXBNRELU":
                if use_focus and idx == 0:
                    raw.append(Focus(info["in"], info["out"], info["k"], act=act))
                else:
                    raw.append(ConvKXBNRELU(info["in"], info["out"], info["k"], info["s"], act=act))
            elif cls in ("SuperResConvK1KX", "SuperResConvKXKX"):
                block_type = "k1kx" if cls == "SuperResConvK1KX" else "kxkx"
                raw.append(
                    CSPStem(
                        info["in"], info["out"], info["btn"],
                        info["s"], info["k"], info["L"],
                        act=act, reparam=reparam, block_type=block_type,
                    )
                )
            else:
                raise NotImplementedError(cls)

        # Re-bundle into 5 csp_stage entries: stem + 4 CSPWrappers, with the
        # 4th wrapper combining stems 3 and 4 (upstream's pattern). The
        # wrapper's downsample/conv_{start,shortcut,fuse} layers always use
        # ReLU regardless of the stems' activation — upstream's CSPWrapper
        # constructor defaults ``act='relu'`` and TinyNAS doesn't override.
        self.csp_stage = nn.ModuleList(
            [
                raw[0],
                CSPWrapper(raw[1]),
                CSPWrapper(raw[2]),
                CSPWrapper((raw[3], raw[4])),
                CSPWrapper(raw[5], with_spp=with_spp),
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for idx, block in enumerate(self.csp_stage):
            x = block(x)
            if idx in self.out_indices:
                out.append(x)
        return out


class _SuperResStemMob(nn.Module):
    """Mob-style stage: list of MobileV3Blocks (no ResConvBlock, no RepConv)."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        btn_c: int,
        kernel_size: int,
        stride: int,
        num_blocks: int,
        with_spp: bool = False,
        act: str = "silu",
        depthwise: bool = False,
        use_se: bool = False,
        block_pos=None,
    ) -> None:
        super().__init__()
        self.block_list = nn.ModuleList()
        for block_id in range(num_blocks):
            in_channels = in_c if block_id == 0 else out_c
            this_stride = stride if block_id == 0 else 1
            self.block_list.append(
                MobileV3Block(
                    in_channels, out_c, btn_c, kernel_size, this_stride,
                    act=act, depthwise=depthwise, use_se=use_se, block_pos=block_pos,
                )
            )
            if block_id == 0 and with_spp:
                self.block_list.append(SPPBottleneck(out_c, out_c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.block_list:
            x = b(x)
        return x


class TinyNASMob(nn.Module):
    """TinyNAS_mob backbone — used by DAMO-YOLO Nano sizes."""

    def __init__(
        self,
        structure_info,
        out_indices=(2, 4, 5),
        with_spp: bool = False,
        use_focus: bool = False,
        act: str = "silu",
        reparam: bool = False,
        depthwise: bool = True,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.out_indices = tuple(out_indices)
        self.block_list = nn.ModuleList()
        for idx, info in enumerate(structure_info):
            cls = info["class"]
            if cls == "ConvKXBNRELU":
                if use_focus:
                    block = Focus(info["in"], info["out"], info["k"], act=act)
                else:
                    # Upstream tinynas_mob hardcodes in=3, stride=2 here.
                    block = ConvKXBNRELU(3, info["out"], info["k"], 2, act=act)
            elif cls in ("SuperResConvK1KX", "SuperResConvKXKX"):
                spp = with_spp if idx == len(structure_info) - 1 else False
                block = _SuperResStemMob(
                    info["in"], info["out"], info["btn"], info["k"], info["s"], info["L"],
                    with_spp=spp, act=act, depthwise=depthwise, use_se=use_se, block_pos=idx,
                )
            else:
                raise NotImplementedError(cls)
            self.block_list.append(block)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for idx, block in enumerate(self.block_list):
            x = block(x)
            if idx in self.out_indices:
                out.append(x)
        return out


class TinyNAS(nn.Module):
    def __init__(
        self,
        structure_info,
        out_indices=(2, 4, 5),
        with_spp: bool = False,
        use_focus: bool = False,
        act: str = "silu",
        reparam: bool = False,
    ) -> None:
        super().__init__()
        self.out_indices = tuple(out_indices)
        self.block_list = nn.ModuleList()

        for idx, info in enumerate(structure_info):
            cls = info["class"]
            if cls == "ConvKXBNRELU":
                if use_focus:
                    block = Focus(info["in"], info["out"], info["k"], act=act)
                else:
                    block = ConvKXBNRELU(info["in"], info["out"], info["k"], info["s"], act=act)
            elif cls in ("SuperResConvK1KX", "SuperResConvKXKX"):
                block_type = "k1kx" if cls == "SuperResConvK1KX" else "kxkx"
                spp = with_spp if idx == len(structure_info) - 1 else False
                block = SuperResStem(
                    info["in"],
                    info["out"],
                    info["btn"],
                    info["k"],
                    info["s"],
                    info["L"],
                    spp,
                    act=act,
                    reparam=reparam,
                    block_type=block_type,
                )
            else:
                raise NotImplementedError(cls)
            self.block_list.append(block)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for idx, block in enumerate(self.block_list):
            x = block(x)
            if idx in self.out_indices:
                out.append(x)
        return out


# ---------------------------------------------------------------------------
# GiraffeNeckV2 (RepGFPN)
# ---------------------------------------------------------------------------


class GiraffeNeckV2(nn.Module):
    def __init__(
        self,
        depth: float = 1.0,
        hidden_ratio: float = 1.0,
        in_channels=(256, 512, 1024),
        out_channels=(256, 512, 1024),
        act: str = "silu",
        spp: bool = False,
        block_name: str = "BasicBlock_3x3_Reverse",
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        Conv = DepthwiseConv if depthwise else ConvBNAct
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.bu_conv13 = Conv(in_channels[1], in_channels[1], 3, 2, act=act)
        self.merge_3 = CSPStage(
            block_name, in_channels[1] + in_channels[2], hidden_ratio, in_channels[2],
            round(3 * depth), act=act, spp=spp, depthwise=depthwise,
        )

        self.bu_conv24 = Conv(in_channels[0], in_channels[0], 3, 2, act=act)
        self.merge_4 = CSPStage(
            block_name, in_channels[0] + in_channels[1] + in_channels[2], hidden_ratio, in_channels[1],
            round(3 * depth), act=act, spp=spp, depthwise=depthwise,
        )

        self.merge_5 = CSPStage(
            block_name, in_channels[1] + in_channels[0], hidden_ratio, out_channels[0],
            round(3 * depth), act=act, spp=spp, depthwise=depthwise,
        )

        self.bu_conv57 = Conv(out_channels[0], out_channels[0], 3, 2, act=act)
        self.merge_7 = CSPStage(
            block_name, out_channels[0] + in_channels[1], hidden_ratio, out_channels[1],
            round(3 * depth), act=act, spp=spp, depthwise=depthwise,
        )

        self.bu_conv46 = Conv(in_channels[1], in_channels[1], 3, 2, act=act)
        self.bu_conv76 = Conv(out_channels[1], out_channels[1], 3, 2, act=act)
        self.merge_6 = CSPStage(
            block_name, in_channels[1] + out_channels[1] + in_channels[2], hidden_ratio, out_channels[2],
            round(3 * depth), act=act, spp=spp, depthwise=depthwise,
        )

    def forward(self, features) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x2, x1, x0 = features

        # node 3: down(x1) ⊕ x0
        x13 = self.bu_conv13(x1)
        x3 = self.merge_3(torch.cat([x0, x13], 1))

        # node 4: x1 ⊕ down(x2) ⊕ up(x3)
        x34 = self.upsample(x3)
        x24 = self.bu_conv24(x2)
        x4 = self.merge_4(torch.cat([x1, x24, x34], 1))

        # node 5: x2 ⊕ up(x4)
        x45 = self.upsample(x4)
        x5 = self.merge_5(torch.cat([x2, x45], 1))

        # node 7: x4 ⊕ down(x5)
        x57 = self.bu_conv57(x5)
        x7 = self.merge_7(torch.cat([x4, x57], 1))

        # node 6: x3 ⊕ down(x4) ⊕ down(x7)
        x46 = self.bu_conv46(x4)
        x76 = self.bu_conv76(x7)
        x6 = self.merge_6(torch.cat([x3, x46, x76], 1))

        return x5, x7, x6


# ---------------------------------------------------------------------------
# ZeroHead (GFL / GFL-V2 detection head, eval-only port)
# ---------------------------------------------------------------------------


class Scale(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class Integral(nn.Module):
    """Project a 4×(reg_max+1) distribution onto a scalar distance per side."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("project", torch.linspace(0, reg_max, reg_max + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, hw, _, _ = x.size()
        x = x.reshape(b * hw * 4, self.reg_max + 1)
        y = self.project.type_as(x).unsqueeze(1)
        return torch.matmul(x, y).reshape(b, hw, 4)


def distance2bbox(points: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    return torch.stack([x1, y1, x2, y2], -1)


class ZeroHead(nn.Module):
    """GFL detection head — inference + training."""

    def __init__(
        self,
        num_classes: int,
        in_channels,
        stacked_convs: int = 4,
        feat_channels: int = 256,
        reg_max: int = 12,
        strides=(8, 16, 32),
        norm: str = "gn",
        act: str = "relu",
        legacy: bool = True,
        last_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.last_kernel_size = last_kernel_size
        self.act = act
        self.strides = list(strides)
        self.reg_max = reg_max
        self.cls_out_channels = num_classes + 1 if legacy else num_classes
        if stacked_convs == 0:
            # No shared cls/reg stack: gfl_{cls,reg} read the FPN feature
            # directly, so per-stride feat_channels must equal in_channels.
            self.feat_channels = list(in_channels)
        elif isinstance(feat_channels, (list, tuple)):
            self.feat_channels = list(feat_channels)
        else:
            self.feat_channels = [feat_channels] * len(self.strides)

        self.integral = Integral(self.reg_max)
        self._init_layers()

        # Training-only objects (lazy import — avoids loading them when
        # only running inference).
        self._assigner = None
        self._loss_cls = None
        self._loss_dfl = None
        self._loss_bbox = None

    def _ensure_train_modules(self) -> None:
        if self._assigner is not None:
            return
        from .loss import (
            AlignOTAAssigner,
            DistributionFocalLoss,
            GIoULoss,
            QualityFocalLoss,
        )

        self._assigner = AlignOTAAssigner(center_radius=2.5, cls_weight=1.0, iou_weight=3.0)
        self._loss_cls = QualityFocalLoss(use_sigmoid=False, beta=2.0, loss_weight=1.0)
        self._loss_bbox = GIoULoss(loss_weight=2.0)
        self._loss_dfl = DistributionFocalLoss(loss_weight=0.25)

    def _build_not_shared_convs(self, in_channel: int, feat_channels: int):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = feat_channels if i > 0 else in_channel
            kernel_size = 3 if i > 0 else 1
            cls_convs.append(
                ConvBNAct(chn, feat_channels, kernel_size, stride=1, groups=1, norm="bn", act=self.act)
            )
            reg_convs.append(
                ConvBNAct(chn, feat_channels, kernel_size, stride=1, groups=1, norm="bn", act=self.act)
            )
        return cls_convs, reg_convs

    def _init_layers(self) -> None:
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(len(self.strides)):
            cc, rc = self._build_not_shared_convs(self.in_channels[i], self.feat_channels[i])
            self.cls_convs.append(cc)
            self.reg_convs.append(rc)

        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels[i],
                    self.cls_out_channels,
                    self.last_kernel_size,
                    padding=self.last_kernel_size // 2,
                )
                for i in range(len(self.strides))
            ]
        )
        self.gfl_reg = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels[i],
                    4 * (self.reg_max + 1),
                    self.last_kernel_size,
                    padding=self.last_kernel_size // 2,
                )
                for i in range(len(self.strides))
            ]
        )
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    # ---- forward (eval) -------------------------------------------------

    def get_single_level_center_priors(self, batch_size, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = (torch.arange(0, int(w), dtype=dtype, device=device)) * stride
        y_range = (torch.arange(0, int(h), dtype=dtype, device=device)) * stride
        x = x_range.repeat(h, 1).flatten()
        y = y_range.unsqueeze(-1).repeat(1, w).flatten()
        strides = x.new_full((x.shape[0],), stride)
        priors = torch.stack([x, y, strides, strides], dim=-1)
        return priors.unsqueeze(0).repeat(batch_size, 1, 1)

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg, scale):
        cls_feat = x
        reg_feat = x
        for cc, rc in zip(cls_convs, reg_convs):
            cls_feat = cc(cls_feat)
            reg_feat = rc(reg_feat)
        bbox_pred = scale(gfl_reg(reg_feat)).float()
        cls_score = gfl_cls(cls_feat).sigmoid()
        return cls_score, bbox_pred

    def forward(self, xin, targets=None):
        """Dispatch to train/eval. Eval returns ``(cls, boxes)`` (xyxy in
        model-input pixels). Train returns a dict of loss components.
        """
        if self.training:
            assert targets is not None, "ZeroHead.forward(targets=None) in train mode"
            return self._forward_train(xin, targets)
        return self._forward_eval(xin)

    def _forward_eval(self, xin):
        # Priors are recomputed each call. Cheap (few thousand floats per
        # scale) and avoids cross-call state that breaks TorchScript tracing
        # and ONNX exporter shape inference.
        priors = torch.cat(
            [
                self.get_single_level_center_priors(
                    xin[i].shape[0], xin[i].shape[-2:], stride, dtype=torch.float32, device=xin[0].device
                )
                for i, stride in enumerate(self.strides)
            ],
            dim=1,
        )

        cls_flat: List[torch.Tensor] = []
        reg_flat: List[torch.Tensor] = []
        for i in range(len(self.strides)):
            cls_score, bbox_pred = self.forward_single(
                xin[i], self.cls_convs[i], self.reg_convs[i], self.gfl_cls[i], self.gfl_reg[i], self.scales[i]
            )
            N, _, H, W = bbox_pred.size()
            bp = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
            bp = bp.flatten(start_dim=3).permute(0, 3, 1, 2)
            cs = cls_score.flatten(start_dim=2).permute(0, 2, 1)
            cls_flat.append(cs)
            reg_flat.append(bp)
        cls_out = torch.cat(cls_flat, dim=1)[:, :, : self.num_classes]
        reg_out = torch.cat(reg_flat, dim=1)
        dis = self.integral(reg_out) * priors[..., 2, None]
        boxes = distance2bbox(priors[..., :2], dis)
        return cls_out, boxes

    # ---- training -------------------------------------------------------

    def _forward_train(self, xin, targets):
        """Train forward. ``targets`` is a list (length B) of dicts with
        keys ``boxes`` (xyxy, model-input pixel coords) and ``labels``.
        """
        self._ensure_train_modules()
        device = xin[0].device

        priors_list = [
            self.get_single_level_center_priors(
                xin[i].shape[0], xin[i].shape[-2:], stride, dtype=torch.float32, device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        mlvl_priors = torch.cat(priors_list, dim=1)

        # Per-scale forward: keep raw bbox dist (pre-softmax) for DFL, plus
        # a softmaxed copy for box decoding.
        cls_list: List[torch.Tensor] = []
        bbox_dist_list: List[torch.Tensor] = []           # softmaxed (N, HW, 4, R+1)
        bbox_dist_raw_list: List[torch.Tensor] = []       # pre-softmax (N, HW, 4, R+1)
        for i in range(len(self.strides)):
            cls_score, bbox_pred = self.forward_single(
                xin[i], self.cls_convs[i], self.reg_convs[i], self.gfl_cls[i], self.gfl_reg[i], self.scales[i]
            )
            N, _, H, W = bbox_pred.size()
            raw = bbox_pred.reshape(N, 4, self.reg_max + 1, H, W)
            sm = F.softmax(raw, dim=2)
            cs = cls_score.flatten(start_dim=2).permute(0, 2, 1)              # (N, HW, C_out)
            cls_list.append(cs)
            bbox_dist_list.append(sm.flatten(start_dim=3).permute(0, 3, 1, 2))
            bbox_dist_raw_list.append(raw.flatten(start_dim=3).permute(0, 3, 1, 2))

        cls_scores = torch.cat(cls_list, dim=1)                # (N, A, C_out)
        bbox_dist = torch.cat(bbox_dist_list, dim=1)           # (N, A, 4, R+1)
        bbox_dist_raw = torch.cat(bbox_dist_raw_list, dim=1)   # (N, A, 4, R+1)

        decoded = distance2bbox(
            mlvl_priors[..., :2], self.integral(bbox_dist) * mlvl_priors[..., 2, None]
        )

        # ---- per-image target assignment --------------------------------
        labels_all = []
        scores_all = []
        weights_all = []
        bbox_targets_all = []
        dfl_targets_all = []
        bbox_weights_all = []
        num_pos_total = 0
        for b in range(cls_scores.size(0)):
            t = targets[b]
            res = self._assign_one(
                mlvl_priors[b], cls_scores[b], decoded[b].detach(), t["boxes"], t["labels"],
            )
            labels_all.append(res["labels"])
            scores_all.append(res["label_scores"])
            weights_all.append(res["label_weights"])
            bbox_targets_all.append(res["bbox_targets"])
            bbox_weights_all.append(res["bbox_weights"])
            dfl_targets_all.append(res["dfl_targets"])
            num_pos_total += res["num_pos"]

        labels = torch.cat(labels_all, dim=0)
        label_scores = torch.cat(scores_all, dim=0)
        bbox_targets = torch.cat(bbox_targets_all, dim=0)
        dfl_targets = torch.cat(dfl_targets_all, dim=0)

        cls_flat = cls_scores.reshape(-1, self.cls_out_channels)
        bbox_dist_raw_flat = bbox_dist_raw.reshape(-1, 4 * (self.reg_max + 1))
        decoded_flat = decoded.reshape(-1, 4)

        num_total_pos = max(float(num_pos_total), 1.0)
        loss_cls = self._loss_cls(cls_flat, (labels, label_scores), avg_factor=num_total_pos)

        pos_inds = torch.nonzero((labels >= 0) & (labels < self.num_classes), as_tuple=False).squeeze(1)
        if pos_inds.numel() > 0:
            weight_targets = cls_flat.detach()[pos_inds].max(dim=1)[0]
            norm = max(float(weight_targets.sum().item()), 1.0)
            loss_bbox = self._loss_bbox(
                decoded_flat[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=norm,
            )
            loss_dfl = self._loss_dfl(
                bbox_dist_raw_flat[pos_inds].reshape(-1, self.reg_max + 1),
                dfl_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * norm,
            )
        else:
            loss_bbox = bbox_dist.sum() * 0.0
            loss_dfl = bbox_dist.sum() * 0.0

        return {
            "total_loss": loss_cls + loss_bbox + loss_dfl,
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_dfl": loss_dfl,
        }

    def _assign_one(self, priors, cls_scores, decoded, gt_bboxes, gt_labels):
        """Run AlignOTA + produce regression targets for a single image."""
        from .loss import bbox_overlaps  # noqa: F401  (kept for parity with upstream import surface)

        num_priors = priors.size(0)
        labels = priors.new_full((num_priors,), self.num_classes, dtype=torch.long)
        label_weights = priors.new_zeros(num_priors)
        label_scores = priors.new_zeros(num_priors)
        bbox_targets = torch.zeros_like(priors)
        bbox_weights = torch.zeros_like(priors)
        dfl_targets = torch.zeros_like(priors)

        if gt_labels.numel() == 0:
            return dict(
                labels=labels, label_scores=label_scores, label_weights=label_weights,
                bbox_targets=bbox_targets, bbox_weights=bbox_weights, dfl_targets=dfl_targets, num_pos=0,
            )

        # Assigner takes full-class scores (not the legacy +1 column).
        cls_scores_for_assigner = cls_scores[:, : self.num_classes]
        result = self._assigner.assign(cls_scores_for_assigner.detach(), priors, decoded, gt_bboxes, gt_labels)

        pos_inds = (result.gt_inds > 0).nonzero(as_tuple=False).squeeze(-1)
        neg_inds = (result.gt_inds == 0).nonzero(as_tuple=False).squeeze(-1)
        pos_assigned_gt = result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt] if gt_bboxes.numel() > 0 else gt_bboxes.new_empty((0, 4))
        pos_ious = result.max_overlaps[pos_inds]

        if pos_inds.numel() > 0:
            labels[pos_inds] = gt_labels[pos_assigned_gt].long()
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0
            bbox_targets[pos_inds] = pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0
            # Distance targets in stride units, clamped to [0, reg_max-eps].
            stride_y = priors[pos_inds, None, 2]
            tgt_dist = self._bbox2distance(
                priors[pos_inds, :2] / stride_y,
                pos_gt_bboxes / stride_y,
                self.reg_max,
            )
            dfl_targets[pos_inds, :] = tgt_dist
        if neg_inds.numel() > 0:
            label_weights[neg_inds] = 1.0

        return dict(
            labels=labels, label_scores=label_scores, label_weights=label_weights,
            bbox_targets=bbox_targets, bbox_weights=bbox_weights, dfl_targets=dfl_targets,
            num_pos=int(pos_inds.numel()),
        )

    @staticmethod
    def _bbox2distance(points, bbox, max_dis: int, eps: float = 0.1):
        left = (points[:, 0] - bbox[:, 0]).clamp(0, max_dis - eps)
        top = (points[:, 1] - bbox[:, 1]).clamp(0, max_dis - eps)
        right = (bbox[:, 2] - points[:, 0]).clamp(0, max_dis - eps)
        bottom = (bbox[:, 3] - points[:, 1]).clamp(0, max_dis - eps)
        return torch.stack([left, top, right, bottom], -1)


# ---------------------------------------------------------------------------
# Detector — backbone → neck → head
# ---------------------------------------------------------------------------


class Detector(nn.Module):
    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        # Upstream `Detector.init_model()` walks every BatchNorm2d and sets
        # eps=1e-3, momentum=0.03 *before* loading weights. eps is not in the
        # state_dict, so omitting this silently degrades inference: the
        # stored running stats were collected with eps=1e-3, and PyTorch's
        # default eps=1e-5 yields different normalised activations. Apply
        # here so subsequent state_dict loads are consistent.
        self._init_bn_constants()

    def _init_bn_constants(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, x: torch.Tensor, targets=None):
        feats = self.backbone(x)
        fpn = self.neck(feats)
        if self.training:
            return self.head(fpn, targets=targets)
        return self.head(fpn)

    def switch_to_deploy(self) -> None:
        for m in self.modules():
            if isinstance(m, RepConv):
                m.switch_to_deploy()


# ===========================================================================
# Per-size architecture metadata (formerly structures.py)
# ===========================================================================

# ---- TinyNAS_res structures (used by N / T / S / L) ----------------------

# tinynas_L20_k1kx.txt → DAMO-YOLO-T
TINYNAS_L20_K1KX = [
    {"class": "ConvKXBNRELU", "in": 3, "k": 3, "out": 24, "s": 1},
    {"L": 2, "btn": 24, "class": "SuperResConvK1KX", "in": 24, "k": 3, "out": 64, "s": 2},
    {"L": 2, "btn": 64, "class": "SuperResConvK1KX", "in": 64, "k": 3, "out": 96, "s": 2},
    {"L": 2, "btn": 96, "class": "SuperResConvK1KX", "in": 96, "k": 3, "out": 192, "s": 2},
    {"L": 2, "btn": 152, "class": "SuperResConvK1KX", "in": 192, "k": 3, "out": 192, "s": 1},
    {"L": 1, "btn": 192, "class": "SuperResConvK1KX", "in": 192, "k": 3, "out": 384, "s": 2},
]


# tinynas_L25_k1kx.txt → DAMO-YOLO-S
TINYNAS_L25_K1KX = [
    {"class": "ConvKXBNRELU", "in": 3, "k": 3, "out": 32, "s": 1},
    {"L": 1, "btn": 24, "class": "SuperResConvK1KX", "in": 32, "k": 3, "out": 128, "s": 2},
    {"L": 5, "btn": 88, "class": "SuperResConvK1KX", "in": 128, "k": 3, "out": 128, "s": 2},
    {"L": 3, "btn": 128, "class": "SuperResConvK1KX", "in": 128, "k": 3, "out": 256, "s": 2},
    {"L": 2, "btn": 120, "class": "SuperResConvK1KX", "in": 256, "k": 3, "out": 256, "s": 1},
    {"L": 1, "btn": 144, "class": "SuperResConvK1KX", "in": 256, "k": 3, "out": 512, "s": 2},
]


# tinynas_L35_kxkx.txt → DAMO-YOLO-M (uses TinyNAS_csp backbone, kxkx blocks)
TINYNAS_L35_KXKX = [
    {"class": "ConvKXBNRELU", "in": 3, "k": 3, "out": 32, "s": 1},
    {"L": 2, "btn": 64, "class": "SuperResConvKXKX", "in": 32, "k": 3, "out": 128, "s": 2},
    {"L": 4, "btn": 64, "class": "SuperResConvKXKX", "in": 128, "k": 3, "out": 128, "s": 2},
    {"L": 4, "btn": 256, "class": "SuperResConvKXKX", "in": 128, "k": 3, "out": 256, "s": 2},
    {"L": 4, "btn": 256, "class": "SuperResConvKXKX", "in": 256, "k": 3, "out": 256, "s": 1},
    {"L": 3, "btn": 256, "class": "SuperResConvKXKX", "in": 256, "k": 3, "out": 512, "s": 2},
]


# tinynas_L45_kxkx.txt → DAMO-YOLO-L (TinyNAS_csp, kxkx). Pretrained
# weights are not currently downloadable (Aliyun bucket is 404 and
# ModelScope hosts only T/S/M). Structure is registered for
# from-scratch training; download routing falls through to the user.
TINYNAS_L45_KXKX = [
    {"class": "ConvKXBNRELU", "in": 3, "k": 3, "out": 32, "s": 1},
    {"L": 3, "btn": 96, "class": "SuperResConvKXKX", "in": 32, "k": 3, "out": 128, "s": 2},
    {"L": 5, "btn": 96, "class": "SuperResConvKXKX", "in": 128, "k": 3, "out": 128, "s": 2},
    {"L": 5, "btn": 384, "class": "SuperResConvKXKX", "in": 128, "k": 3, "out": 256, "s": 2},
    {"L": 5, "btn": 384, "class": "SuperResConvKXKX", "in": 256, "k": 3, "out": 256, "s": 1},
    {"L": 4, "btn": 384, "class": "SuperResConvKXKX", "in": 256, "k": 3, "out": 512, "s": 2},
]


@dataclass(frozen=True)
class FamilyConfig:
    """A complete DAMO-YOLO family member spec."""

    structure: List[Dict]
    backbone_class: str   # "tinynas_res", "tinynas_csp", or "tinynas_mob"
    backbone_with_spp: bool
    backbone_use_focus: bool
    backbone_act: str
    backbone_reparam: bool
    backbone_out_indices: Tuple[int, int, int]
    neck_in_channels: Tuple[int, int, int]
    neck_out_channels: Tuple[int, int, int]
    neck_depth: float
    neck_hidden_ratio: float
    neck_act: str
    neck_spp: bool
    head_in_channels: Tuple[int, int, int]
    head_stacked_convs: int
    head_reg_max: int
    head_act: str
    head_legacy: bool
    head_feat_channels: int = 256
    head_last_kernel_size: int = 3
    backbone_depthwise: bool = False
    backbone_use_se: bool = False
    neck_depthwise: bool = False


# ---- DAMO-YOLO-T (42.0 mAP, target for first-pass parity) ----------------

DAMOYOLO_T = FamilyConfig(
    structure=TINYNAS_L20_K1KX,
    backbone_class="tinynas_res",
    backbone_with_spp=True,
    backbone_use_focus=True,
    backbone_act="relu",
    backbone_reparam=True,
    backbone_out_indices=(2, 4, 5),
    neck_in_channels=(96, 192, 384),
    neck_out_channels=(64, 128, 256),
    neck_depth=1.0,
    neck_hidden_ratio=1.0,
    neck_act="relu",
    neck_spp=False,
    head_in_channels=(64, 128, 256),
    head_stacked_convs=0,
    head_reg_max=16,
    head_act="silu",
    # legacy=False matches the post-distill release weights (T 42.0,
    # S 46.0, M 50.2) recovered from the Internet Archive. The earlier
    # ModelScope-hosted pre-distill weights used legacy=True (cls head
    # emits num_classes + 1 channels with the trailing channel unused).
    head_legacy=False,
)


DAMOYOLO_S = FamilyConfig(
    structure=TINYNAS_L25_K1KX,
    backbone_class="tinynas_res",
    backbone_with_spp=True,
    backbone_use_focus=True,
    backbone_act="relu",
    backbone_reparam=True,
    backbone_out_indices=(2, 4, 5),
    neck_in_channels=(128, 256, 512),
    neck_out_channels=(128, 256, 512),
    neck_depth=1.0,
    neck_hidden_ratio=0.75,
    neck_act="relu",
    neck_spp=False,
    head_in_channels=(128, 256, 512),
    head_stacked_convs=0,
    head_reg_max=16,
    head_act="silu",
    head_legacy=False,
)


DAMOYOLO_M = FamilyConfig(
    structure=TINYNAS_L35_KXKX,
    backbone_class="tinynas_csp",
    backbone_with_spp=True,
    backbone_use_focus=True,
    backbone_act="silu",
    backbone_reparam=True,
    # CSP backbone outputs from csp_stage indices (5-element list,
    # not the 6-element raw block list).
    backbone_out_indices=(2, 3, 4),
    neck_in_channels=(128, 256, 512),
    neck_out_channels=(128, 256, 512),
    neck_depth=1.5,
    neck_hidden_ratio=1.0,
    neck_act="silu",
    neck_spp=False,
    head_in_channels=(128, 256, 512),
    head_stacked_convs=0,
    head_reg_max=16,
    head_act="silu",
    head_legacy=False,
)


# tinynas_nano_*.txt → DAMO-YOLO Nano variants (TinyNAS_mob backbone, depthwise everywhere)
TINYNAS_NANO_SMALL = [
    {"class": "ConvKXBNRELU", "in": 3, "k": 3, "out": 16, "s": 1},
    {"L": 1, "btn": 24, "class": "SuperResConvK1KX", "in": 16, "k": 3, "out": 24, "s": 2},
    {"L": 2, "btn": 64, "class": "SuperResConvK1KX", "in": 24, "k": 3, "out": 40, "s": 2},
    {"L": 2, "btn": 40, "class": "SuperResConvK1KX", "in": 40, "k": 3, "out": 64, "s": 2},
    {"L": 2, "btn": 152, "class": "SuperResConvK1KX", "in": 64, "k": 3, "out": 80, "s": 1},
    {"L": 2, "btn": 192, "class": "SuperResConvK1KX", "in": 80, "k": 3, "out": 160, "s": 2},
]

TINYNAS_NANO_MIDDLE = [
    {"class": "ConvKXBNRELU", "in": 3, "k": 3, "out": 16, "s": 1},
    {"L": 2, "btn": 24, "class": "SuperResConvK1KX", "in": 16, "k": 3, "out": 40, "s": 2},
    {"L": 2, "btn": 64, "class": "SuperResConvK1KX", "in": 40, "k": 3, "out": 64, "s": 2},
    {"L": 2, "btn": 40, "class": "SuperResConvK1KX", "in": 64, "k": 3, "out": 112, "s": 2},
    {"L": 2, "btn": 152, "class": "SuperResConvK1KX", "in": 112, "k": 3, "out": 128, "s": 1},
    {"L": 1, "btn": 192, "class": "SuperResConvK1KX", "in": 128, "k": 3, "out": 256, "s": 2},
]

TINYNAS_NANO_LARGE = [
    {"class": "ConvKXBNRELU", "in": 3, "k": 3, "out": 24, "s": 1},
    {"L": 1, "btn": 24, "class": "SuperResConvK1KX", "in": 24, "k": 3, "out": 48, "s": 2},
    {"L": 2, "btn": 64, "class": "SuperResConvK1KX", "in": 48, "k": 3, "out": 80, "s": 2},
    {"L": 2, "btn": 40, "class": "SuperResConvK1KX", "in": 80, "k": 3, "out": 160, "s": 2},
    {"L": 3, "btn": 152, "class": "SuperResConvK1KX", "in": 160, "k": 3, "out": 160, "s": 1},
    {"L": 2, "btn": 192, "class": "SuperResConvK1KX", "in": 160, "k": 3, "out": 320, "s": 2},
]


def _make_nano_config(structure, neck_io: Tuple[int, int, int]) -> "FamilyConfig":
    return FamilyConfig(
        structure=structure,
        backbone_class="tinynas_mob",
        backbone_with_spp=True,
        backbone_use_focus=False,
        backbone_act="silu",
        backbone_reparam=False,
        backbone_out_indices=(2, 4, 5),
        backbone_depthwise=True,
        backbone_use_se=False,
        neck_in_channels=neck_io,
        neck_out_channels=neck_io,
        neck_depth=0.5,
        neck_hidden_ratio=0.5,
        neck_act="silu",
        neck_spp=False,
        neck_depthwise=True,
        head_in_channels=neck_io,
        head_stacked_convs=0,
        head_reg_max=7,
        head_act="silu",
        head_legacy=False,
        head_last_kernel_size=1,
    )


DAMOYOLO_NS = _make_nano_config(TINYNAS_NANO_SMALL, (40, 80, 160))
DAMOYOLO_NM = _make_nano_config(TINYNAS_NANO_MIDDLE, (64, 128, 256))
DAMOYOLO_NL = _make_nano_config(TINYNAS_NANO_LARGE, (80, 160, 320))


DAMOYOLO_L = FamilyConfig(
    structure=TINYNAS_L45_KXKX,
    backbone_class="tinynas_csp",
    backbone_with_spp=True,
    backbone_use_focus=True,
    backbone_act="silu",
    backbone_reparam=True,
    backbone_out_indices=(2, 3, 4),
    neck_in_channels=(128, 256, 512),
    neck_out_channels=(128, 256, 512),
    neck_depth=2.0,        # upstream: depth 2.0 for L
    neck_hidden_ratio=1.0,
    neck_act="silu",
    neck_spp=False,
    head_in_channels=(128, 256, 512),
    head_stacked_convs=0,
    head_reg_max=16,
    head_act="silu",
    head_legacy=False,
)


SIZES: Dict[str, FamilyConfig] = {
    "ns": DAMOYOLO_NS,
    "nm": DAMOYOLO_NM,
    "nl": DAMOYOLO_NL,
    "t": DAMOYOLO_T,
    "s": DAMOYOLO_S,
    "m": DAMOYOLO_M,
    "l": DAMOYOLO_L,
}


# ===========================================================================
# Factory builders (formerly builder.py)
# ===========================================================================

_BACKBONES = {
    "tinynas_res": TinyNAS,
    "tinynas_csp": TinyNASCSP,
    "tinynas_mob": TinyNASMob,
}


def _build_backbone(cfg: FamilyConfig):
    cls = _BACKBONES[cfg.backbone_class]
    kwargs = dict(
        structure_info=cfg.structure,
        out_indices=cfg.backbone_out_indices,
        with_spp=cfg.backbone_with_spp,
        use_focus=cfg.backbone_use_focus,
        act=cfg.backbone_act,
        reparam=cfg.backbone_reparam,
    )
    if cfg.backbone_class == "tinynas_mob":
        kwargs["depthwise"] = cfg.backbone_depthwise
        kwargs["use_se"] = cfg.backbone_use_se
    return cls(**kwargs)


def _build_neck(cfg: FamilyConfig) -> GiraffeNeckV2:
    return GiraffeNeckV2(
        depth=cfg.neck_depth,
        hidden_ratio=cfg.neck_hidden_ratio,
        in_channels=cfg.neck_in_channels,
        out_channels=cfg.neck_out_channels,
        act=cfg.neck_act,
        spp=cfg.neck_spp,
        block_name="BasicBlock_3x3_Reverse",
        depthwise=cfg.neck_depthwise,
    )


def _build_head(cfg: FamilyConfig, num_classes: int) -> ZeroHead:
    return ZeroHead(
        num_classes=num_classes,
        in_channels=cfg.head_in_channels,
        stacked_convs=cfg.head_stacked_convs,
        feat_channels=cfg.head_feat_channels,
        reg_max=cfg.head_reg_max,
        strides=(8, 16, 32),
        act=cfg.head_act,
        legacy=cfg.head_legacy,
        last_kernel_size=cfg.head_last_kernel_size,
    )


def build_damoyolo(size: str = "t", num_classes: int = 80) -> nn.Module:
    """Build a DAMO-YOLO Detector for the given size."""
    if size not in SIZES:
        raise ValueError(f"Unknown DAMO-YOLO size {size!r}. Available: {sorted(SIZES)}")
    cfg = SIZES[size]
    backbone = _build_backbone(cfg)
    neck = _build_neck(cfg)
    head = _build_head(cfg, num_classes=num_classes)
    return Detector(backbone, neck, head)
