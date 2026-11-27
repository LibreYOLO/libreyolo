"""HybridEncoder for LoopDVR.

Ported from Intellindust-AI-Lab/LoopDVR (commit f2aba1f122029e4db757b69cd37fc5a4909db34e):
    loopdvr/models/encoder.py
Copyright (c) 2026 The LoopDVR Authors (modifications).
SPDX-License-Identifier: Apache-2.0

Modified from EdgeCrafter, D-FINE (Apache-2.0), and RT-DETR (Apache-2.0).
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


def _activation(name: str | None) -> nn.Module:
    if name is None:
        return nn.Identity()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported encoder activation {name!r}")


class ConvNormLayer(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int,
        stride: int,
        g: int = 1,
        padding: int | None = None,
        bias: bool = False,
        act: str | None = None,
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride, padding, groups=g, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = _activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))

    def fused_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        std = (self.norm.running_var + self.norm.eps).sqrt()
        scale = (self.norm.weight / std).reshape(-1, 1, 1, 1)
        bias = self.norm.bias - self.norm.running_mean * self.norm.weight / std
        return self.conv.weight * scale, bias


class ConvNormLayer_fuse(ConvNormLayer):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv_bn_fused"):
            return self.act(self.conv_bn_fused(x))
        return super().forward(x)

    @torch.no_grad()
    def convert_to_deploy(self) -> None:
        if hasattr(self, "conv_bn_fused"):
            return
        weight, bias = self.fused_parameters()
        self.conv_bn_fused = nn.Conv2d(
            self.conv.in_channels,
            self.conv.out_channels,
            self.conv.kernel_size,
            self.conv.stride,
            self.conv.padding,
            groups=self.conv.groups,
            bias=True,
            device=weight.device,
            dtype=weight.dtype,
        )
        self.conv_bn_fused.weight.copy_(weight)
        self.conv_bn_fused.bias.copy_(bias)
        del self.conv, self.norm


class SCDown(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(channels, channels, 1, 1)
        self.cv2 = ConvNormLayer_fuse(channels, channels, 3, 2, g=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    def __init__(self, channels: int, act: str):
        super().__init__()
        self.conv1 = ConvNormLayer(channels, channels, 3, 1)
        self.conv2 = ConvNormLayer(channels, channels, 1, 1)
        self.act = _activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return self.act(self.conv(x))
        return self.act(self.conv1(x) + self.conv2(x))

    @torch.no_grad()
    def convert_to_deploy(self) -> None:
        if hasattr(self, "conv"):
            return
        weight3, bias3 = self.conv1.fused_parameters()
        weight1, bias1 = self.conv2.fused_parameters()
        self.conv = nn.Conv2d(
            weight3.shape[1],
            weight3.shape[0],
            3,
            padding=1,
            device=weight3.device,
            dtype=weight3.dtype,
        )
        self.conv.weight.copy_(weight3 + F.pad(weight1, [1, 1, 1, 1]))
        self.conv.bias.copy_(bias3 + bias1)
        del self.conv1, self.conv2


class CSPLayer(nn.Module):
    def __init__(self, in_channels: int, channels: int, count: int, act: str):
        super().__init__()
        self.conv1 = ConvNormLayer_fuse(in_channels, 2 * channels, 1, 1, act=act)
        self.bottlenecks = nn.Sequential(*(VGGBlock(channels, act) for _ in range(count)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip, y = self.conv1(x).chunk(2, 1)
        return skip + self.bottlenecks(y)


class FeatureFusion(nn.Module):
    def __init__(self, channels: int, inner_channels: int, count: int, act: str):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(channels, 2 * channels, 1, 1, act=act)
        self.cv2 = nn.Sequential(CSPLayer(channels, inner_channels, count, act))
        self.cv3 = nn.Sequential(CSPLayer(inner_channels, inner_channels, count, act))
        self.cv4 = ConvNormLayer_fuse(
            2 * channels + 2 * inner_channels, channels, 1, 1, act=act
        )
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.channels, self.channels), 1))
        y.append(self.cv2(y[-1]))
        y.append(self.cv3(y[-1]))
        return self.cv4(torch.cat(y, dim=1))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, dim: int, nhead: int, dim_feedforward: int, dropout: float, act: str
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.activation = _activation(act)

    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        query = key = x + position
        attention, _ = self.self_attn(query, key, value=x)
        x = self.norm1(x + attention)
        return self.norm2(x + self.linear2(self.activation(self.linear1(x))))


class TransformerEncoder(nn.Module):
    def __init__(self, layer: nn.Module, count: int):
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(count))

    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, position)
        return x


class HybridEncoder(nn.Module):
    """Refine coarse features with attention and fuse them across pyramid levels."""

    def __init__(
        self,
        in_channels,
        feat_strides,
        hidden_dim: int,
        nhead: int,
        dim_feedforward: int,
        use_encoder_idx,
        num_encoder_layers: int,
        expansion: float,
        depth_mult: float,
        act: str,
        eval_spatial_size,
        dropout: float = 0.0,
        csp_type: str = "csp2",
        fuse_op: str = "sum",
        pe_temperature: int = 10000,
    ):
        super().__init__()
        if csp_type != "csp2" or fuse_op != "sum":
            raise ValueError("The released encoder uses CSP2 blocks and additive fusion")
        if (
            len(in_channels) != 3
            or len(feat_strides) != 3
            or any(c != hidden_dim for c in in_channels)
        ):
            raise ValueError("Expected three feature levels of width hidden_dim")
        self.use_encoder_idx = tuple(use_encoder_idx)
        self.hidden_dim = hidden_dim
        self.pe_temperature = pe_temperature
        layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout, act)
        self.encoder = nn.ModuleList(
            TransformerEncoder(layer, num_encoder_layers) for _ in self.use_encoder_idx
        )
        for idx in self.use_encoder_idx:
            h, w = (side // feat_strides[idx] for side in eval_spatial_size)
            position = self._position_embedding(w, h, hidden_dim, pe_temperature)
            self.register_buffer(f"pos_embed{idx}", position, persistent=False)

        def fusion_block() -> FeatureFusion:
            inner = round(expansion * hidden_dim // 2)
            return FeatureFusion(hidden_dim, inner, round(3 * depth_mult), act)

        self.lateral_convs = nn.ModuleList(
            ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1) for _ in range(2)
        )
        self.fpn_blocks = nn.ModuleList(fusion_block() for _ in range(2))
        self.downsample_convs = nn.ModuleList(
            nn.Sequential(SCDown(hidden_dim)) for _ in range(2)
        )
        self.pan_blocks = nn.ModuleList(fusion_block() for _ in range(2))

    @staticmethod
    def _position_embedding(
        w: int, h: int, dim: int, temperature: int
    ) -> torch.Tensor:
        grid_w, grid_h = torch.meshgrid(
            torch.arange(w, dtype=torch.float32),
            torch.arange(h, dtype=torch.float32),
            indexing="ij",
        )
        pos_dim = dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.cat(
            [out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1
        )[None]

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        features = list(features)
        for idx, transformer in zip(self.use_encoder_idx, self.encoder):
            batch, channels, height, width = features[idx].shape
            tokens = features[idx].flatten(2).permute(0, 2, 1)
            pos = getattr(self, f"pos_embed{idx}")
            if pos.shape[1] != height * width:
                pos = self._position_embedding(
                    width, height, self.hidden_dim, self.pe_temperature
                ).to(device=tokens.device, dtype=tokens.dtype)
            tokens = transformer(tokens, pos)
            features[idx] = (
                tokens.permute(0, 2, 1).reshape(batch, channels, height, width).contiguous()
            )

        inner = [features[-1]]
        for low, lateral, fuse in zip(
            reversed(features[:-1]), self.lateral_convs, self.fpn_blocks
        ):
            inner[0] = lateral(inner[0])
            inner.insert(
                0,
                fuse(F.interpolate(inner[0], scale_factor=2.0, mode="nearest") + low),
            )

        outputs = [inner[0]]
        for high, downsample, fuse in zip(
            inner[1:], self.downsample_convs, self.pan_blocks
        ):
            outputs.append(fuse(downsample(outputs[-1]) + high))
        return outputs
