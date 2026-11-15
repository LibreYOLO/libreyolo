"""Native MoGe-2 normal network.

The decoder and encoder wrapper are ported from Microsoft MoGe at commit
925b8ed835a7a9cdb7578ba15c658a0afc969030 (MIT). The DINOv2 backbone is
reused from LibreYOLO's existing Apache-2.0 Depth Anything vendor instead of
duplicating the same upstream implementation.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..depth_anything._vendor.dinov2 import DINOv2

PATCH_SIZE = 14


def normalized_view_plane_uv(
    width: int,
    height: int,
    *,
    aspect_ratio: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build MoGe's diagonal-normalized image-plane coordinate field."""
    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / (1.0 + aspect_ratio**2) ** 0.5
    span_y = 1.0 / (1.0 + aspect_ratio**2) ** 0.5
    u = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        dtype=dtype,
        device=device,
    )
    v = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        dtype=dtype,
        device=device,
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack((u, v), dim=-1)


class ResidualConvBlock(nn.Module):
    """Two-convolution residual block used throughout the MoGe decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        hidden_channels: int | None = None,
        kernel_size: int = 3,
        padding_mode: str = "replicate",
        activation: Literal["relu", "leaky_relu", "silu", "elu"] = "relu",
        in_norm: Literal[
            "group_norm", "layer_norm", "instance_norm", "none"
        ] = "layer_norm",
        hidden_norm: Literal[
            "group_norm", "layer_norm", "instance_norm", "none"
        ] = "group_norm",
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        activations = {
            "relu": nn.ReLU,
            "leaky_relu": functools.partial(nn.LeakyReLU, negative_slope=0.2),
            "silu": nn.SiLU,
            "elu": nn.ELU,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported MoGe activation: {activation!r}")
        activation_cls = activations[activation]

        def norm(kind: str, channels: int) -> nn.Module:
            if kind == "group_norm":
                return nn.GroupNorm(channels // 32, channels)
            if kind == "layer_norm":
                return nn.GroupNorm(1, channels)
            if kind == "instance_norm":
                return nn.InstanceNorm2d(channels)
            if kind == "none":
                return nn.Identity()
            raise ValueError(f"Unsupported MoGe normalization: {kind!r}")

        self.layers = nn.Sequential(
            norm(in_norm, in_channels),
            activation_cls(),
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
            ),
            norm(hidden_norm, hidden_channels),
            activation_cls(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
            ),
        )
        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + self.skip_connection(x)


class Resampler(nn.Sequential):
    """One level of the coarse-to-fine convolutional decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        type_: Literal[
            "pixel_shuffle",
            "nearest",
            "bilinear",
            "conv_transpose",
            "pixel_unshuffle",
            "avg_pool",
            "max_pool",
        ],
        scale_factor: int = 2,
    ):
        if type_ == "pixel_shuffle":
            super().__init__(
                nn.Conv2d(
                    in_channels,
                    out_channels * scale_factor**2,
                    kernel_size=3,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
            with torch.no_grad():
                for index in range(1, scale_factor**2):
                    self[0].weight[index :: scale_factor**2].copy_(
                        self[0].weight[0 :: scale_factor**2]
                    )
                    self[0].bias[index :: scale_factor**2].copy_(
                        self[0].bias[0 :: scale_factor**2]
                    )
        elif type_ in ("nearest", "bilinear"):
            super().__init__(
                nn.Upsample(
                    scale_factor=scale_factor,
                    mode=type_,
                    align_corners=False if type_ == "bilinear" else None,
                ),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
        elif type_ == "conv_transpose":
            super().__init__(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=scale_factor,
                    stride=scale_factor,
                ),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
            with torch.no_grad():
                self[0].weight.copy_(self[0].weight[:, :, :1, :1])
        elif type_ == "pixel_unshuffle":
            super().__init__(
                nn.PixelUnshuffle(scale_factor),
                nn.Conv2d(
                    in_channels * scale_factor**2,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
        elif type_ == "avg_pool":
            super().__init__(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor),
            )
        elif type_ == "max_pool":
            super().__init__(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor),
            )
        else:
            raise ValueError(f"Unsupported MoGe resampler: {type_!r}")


class ConvStack(nn.Module):
    """Multi-scale residual decoder shared by the neck and normal head."""

    def __init__(
        self,
        dim_in: list[int | None],
        dim_res_blocks: list[int],
        dim_out: list[int | None] | None,
        resamplers: str | list[str],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int | list[int] = 1,
        res_block_in_norm: str = "layer_norm",
        res_block_hidden_norm: str = "group_norm",
        activation: str = "relu",
    ):
        super().__init__()
        output_dims = (
            dim_out
            if isinstance(dim_out, Sequence)
            else list(itertools.repeat(dim_out, len(dim_res_blocks)))
        )
        resampler_types = (
            resamplers
            if isinstance(resamplers, Sequence) and not isinstance(resamplers, str)
            else list(itertools.repeat(resamplers, len(dim_res_blocks) - 1))
        )
        block_counts = (
            num_res_blocks
            if isinstance(num_res_blocks, list)
            else [num_res_blocks] * len(dim_res_blocks)
        )
        self.input_blocks = nn.ModuleList(
            [
                nn.Conv2d(dim_in_, dim_res_block, kernel_size=1)
                if dim_in_ is not None
                else nn.Identity()
                for dim_in_, dim_res_block in zip(dim_in, dim_res_blocks)
            ]
        )
        self.resamplers = nn.ModuleList(
            [
                Resampler(previous, successor, type_=kind)
                for previous, successor, kind in zip(
                    dim_res_blocks[:-1],
                    dim_res_blocks[1:],
                    resampler_types,
                )
            ]
        )
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        ResidualConvBlock(
                            dim_res_block,
                            dim_res_block,
                            dim_times_res_block_hidden * dim_res_block,
                            activation=activation,
                            in_norm=res_block_in_norm,
                            hidden_norm=res_block_hidden_norm,
                        )
                        for _ in range(block_counts[index])
                    ]
                )
                for index, dim_res_block in enumerate(dim_res_blocks)
            ]
        )
        self.output_blocks = nn.ModuleList(
            [
                nn.Conv2d(dim_res_block, dim_out_, kernel_size=1)
                if dim_out_ is not None
                else nn.Identity()
                for dim_out_, dim_res_block in zip(output_dims, dim_res_blocks)
            ]
        )

    def forward(self, in_features: list[torch.Tensor]) -> list[torch.Tensor]:
        out_features = []
        x = None
        for index in range(len(self.res_blocks)):
            feature = self.input_blocks[index](in_features[index])
            x = feature if index == 0 else x + feature
            x = self.res_blocks[index](x)
            out_features.append(self.output_blocks[index](x))
            if index < len(self.res_blocks) - 1:
                x = self.resamplers[index](x)
        return out_features


class DINOv2Encoder(nn.Module):
    """MoGe's DINOv2 feature wrapper; input is RGB in ``[0, 1]``."""

    def __init__(
        self,
        backbone: str,
        intermediate_layers: list[int],
        dim_out: int,
    ):
        super().__init__()
        backbone_configs = {
            "dinov2_vits14": ("vits", 384),
            "dinov2_vitb14": ("vitb", 768),
            "dinov2_vitl14": ("vitl", 1024),
            # Used by MoGe-3 ViT-g (libreyolo.models.moge3).
            "dinov2_vitg14": ("vitg", 1536),
        }
        if backbone not in backbone_configs:
            raise ValueError(
                f"Unsupported MoGe backbone {backbone!r}; expected one of "
                f"{sorted(backbone_configs)}."
            )
        encoder_name, feature_dim = backbone_configs[backbone]
        self.intermediate_layers = list(intermediate_layers)
        self.backbone = DINOv2(encoder_name)
        self.dim_features = feature_dim
        self.output_projections = nn.ModuleList(
            [
                nn.Conv2d(self.dim_features, dim_out, kernel_size=1)
                for _ in self.intermediate_layers
            ]
        )
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        self.onnx_compatible_mode = False

    def forward(
        self,
        image: torch.Tensor,
        token_rows: int,
        token_cols: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = F.interpolate(
            image,
            (token_rows * PATCH_SIZE, token_cols * PATCH_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=not self.onnx_compatible_mode,
        )
        image = (image - self.image_mean) / self.image_std
        features = self.backbone.get_intermediate_layers(
            image,
            n=self.intermediate_layers,
            return_class_token=True,
        )
        projected = torch.stack(
            [
                projection(
                    feature.permute(0, 2, 1)
                    .unflatten(2, (token_rows, token_cols))
                    .contiguous()
                )
                for projection, (feature, _) in zip(
                    self.output_projections,
                    features,
                )
            ],
            dim=1,
        ).sum(dim=1)
        return projected, features[-1][1]


_SIZE_CONFIGS = {
    "s": {
        "backbone": "dinov2_vits14",
        "intermediate_layers": [5, 11],
        "dim": 384,
        "neck_res_blocks": [0, 1, 1, 1, 0],
    },
    "b": {
        "backbone": "dinov2_vitb14",
        "intermediate_layers": [5, 11],
        "dim": 768,
        "neck_res_blocks": [0, 1, 1, 1, 0],
    },
    "l": {
        "backbone": "dinov2_vitl14",
        "intermediate_layers": [5, 11, 17, 23],
        "dim": 1024,
        "neck_res_blocks": [0, 2, 2, 2, 0],
    },
}


class MoGe2NormalNet(nn.Module):
    """MoGe-2 S/B/L normal-only graph with unused geometry heads omitted."""

    def __init__(self, size: str = "s"):
        super().__init__()
        if size not in _SIZE_CONFIGS:
            raise ValueError(
                f"MoGe-2 size must be one of {sorted(_SIZE_CONFIGS)}, got {size!r}."
            )
        config = _SIZE_CONFIGS[size]
        dim = config["dim"]
        decoder_dims = [dim, 256, 128, 64, 32]
        self.encoder = DINOv2Encoder(
            backbone=config["backbone"],
            intermediate_layers=config["intermediate_layers"],
            dim_out=dim,
        )
        self.neck = ConvStack(
            dim_in=[dim + 2, 2, 2, 2, 2],
            dim_out=None,
            dim_res_blocks=decoder_dims,
            num_res_blocks=config["neck_res_blocks"],
            res_block_in_norm="none",
            res_block_hidden_norm="none",
            resamplers=[
                "conv_transpose",
                "conv_transpose",
                "conv_transpose",
                "bilinear",
            ],
        )
        self.normal_head = ConvStack(
            dim_in=decoder_dims,
            dim_out=[None, None, None, None, 3],
            dim_res_blocks=decoder_dims,
            num_res_blocks=[0, 1, 1, 1, 0],
            res_block_in_norm="none",
            res_block_hidden_norm="none",
            resamplers=[
                "conv_transpose",
                "conv_transpose",
                "conv_transpose",
                "bilinear",
            ],
        )
        self.onnx_compatible_mode = False

    def set_onnx_compatible_mode(self, enabled: bool) -> None:
        self.onnx_compatible_mode = bool(enabled)
        self.encoder.onnx_compatible_mode = bool(enabled)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size, _, image_h, image_w = image.shape
        if image_h % PATCH_SIZE or image_w % PATCH_SIZE:
            raise ValueError(
                "MoGe-2 input height and width must be divisible by 14, got "
                f"{image_h}x{image_w}."
            )
        token_rows = image_h // PATCH_SIZE
        token_cols = image_w // PATCH_SIZE
        aspect_ratio = image_w / image_h

        feature, _ = self.encoder(image, token_rows, token_cols)
        features: list[torch.Tensor | None] = [feature, None, None, None, None]
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=token_cols * 2**level,
                height=token_rows * 2**level,
                aspect_ratio=aspect_ratio,
                dtype=image.dtype,
                device=image.device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            features[level] = (
                uv
                if features[level] is None
                else torch.cat((features[level], uv), dim=1)
            )

        decoded = self.neck(features)
        normal = self.normal_head(decoded)[-1]
        normal = F.interpolate(
            normal,
            (image_h, image_w),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        return F.normalize(normal, dim=1)


__all__ = ["MoGe2NormalNet", "normalized_view_plane_uv"]
