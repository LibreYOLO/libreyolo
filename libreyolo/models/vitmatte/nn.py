"""Native ViTMatte-S inference graph.

Portions follow Apache-2.0 sources carrying these notices:
Copyright 2023 HUST-VL and The HuggingFace Inc. team.
Copyright 2023 Meta AI and The HuggingFace Inc. team.

This is an inference-only PyTorch port of the Apache-2.0 Transformers
implementation introduced in huggingface/transformers commit
7d6354e04794f3246bf9a0faf4fead080edeebb6. Module names intentionally match
that reference so its published safetensors checkpoint loads strictly.

Only the ViTDet-S backbone and detail-capture path used by
``hustvl/vitmatte-small-composition-1k`` are included. Transformers config and
model wrappers, the HUST training stack, and its optional dependencies are not
part of this port.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


HIDDEN_SIZE = 384
NUM_HEADS = 6
NUM_LAYERS = 12
PATCH_SIZE = 16
WINDOW_SIZE = 14
WINDOW_BLOCK_INDICES = frozenset({0, 1, 3, 4, 6, 7, 9, 10})
RESIDUAL_BLOCK_INDICES = frozenset({2, 5, 8, 11})


class VitDetEmbeddings(nn.Module):
    """Four-channel patch projection plus interpolated absolute positions."""

    def __init__(self) -> None:
        super().__init__()
        # The ViT-S pretraining grid is 14x14 plus one unused class-token slot.
        self.position_embeddings = nn.Parameter(torch.zeros(1, 197, HIDDEN_SIZE))
        self.projection = nn.Conv2d(
            4,
            HIDDEN_SIZE,
            kernel_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        )

    @staticmethod
    def get_absolute_positions(
        absolute_positions: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        positions = absolute_positions[:, 1:]
        count = int(positions.shape[1])
        side = int(math.sqrt(count))
        if side * side != count:
            raise ValueError("Absolute position embeddings must form a square grid.")
        if torch.jit.is_tracing() or side != height or side != width:
            positions = F.interpolate(
                positions.reshape(1, side, side, -1).permute(0, 3, 1, 2),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
            return positions.permute(0, 2, 3, 1)
        return positions.reshape(1, height, width, -1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 4:
            raise ValueError(
                "ViTMatte expects pixel_values with shape (N, 4, H, W); "
                f"got {tuple(pixel_values.shape)}."
            )
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        embeddings = embeddings + self.get_absolute_positions(
            self.position_embeddings,
            int(embeddings.shape[1]),
            int(embeddings.shape[2]),
        )
        return embeddings.permute(0, 3, 1, 2)


def get_relative_positions(
    query_size: int,
    key_size: int,
    relative_positions: torch.Tensor,
) -> torch.Tensor:
    """Resize and index one axis of decomposed relative positions."""
    max_distance = int(2 * max(query_size, key_size) - 1)
    if relative_positions.shape[0] != max_distance:
        resized = F.interpolate(
            relative_positions.reshape(1, relative_positions.shape[0], -1).permute(
                0, 2, 1
            ),
            size=max_distance,
            mode="linear",
        )
        resized = resized.reshape(-1, max_distance).permute(1, 0)
    else:
        resized = relative_positions

    device = relative_positions.device
    query_coordinates = torch.arange(query_size, device=device)[:, None] * max(
        key_size / query_size, 1.0
    )
    key_coordinates = torch.arange(key_size, device=device)[None, :] * max(
        query_size / key_size, 1.0
    )
    coordinates = (query_coordinates - key_coordinates) + (key_size - 1) * max(
        query_size / key_size, 1.0
    )
    return resized[coordinates.long()]


def add_decomposed_relative_positions(
    attention: torch.Tensor,
    queries: torch.Tensor,
    relative_height: torch.Tensor,
    relative_width: torch.Tensor,
    query_size: tuple[int, int],
    key_size: tuple[int, int],
) -> torch.Tensor:
    """Add MViTv2-style decomposed height/width relative positions."""
    query_height, query_width = query_size
    key_height, key_width = key_size
    rel_h = get_relative_positions(query_height, key_height, relative_height)
    rel_w = get_relative_positions(query_width, key_width, relative_width)

    batch_heads, _, channels = queries.shape
    reshaped_queries = queries.reshape(batch_heads, query_height, query_width, channels)
    height_bias = torch.einsum("bhwc,hkc->bhwk", reshaped_queries, rel_h)
    width_bias = torch.einsum("bhwc,wkc->bhwk", reshaped_queries, rel_w)
    return (
        attention.view(
            batch_heads,
            query_height,
            query_width,
            key_height,
            key_width,
        )
        + height_bias[:, :, :, :, None]
        + width_bias[:, :, :, None, :]
    ).view(batch_heads, query_height * query_width, key_height * key_width)


class VitDetAttention(nn.Module):
    """Manual multi-head attention used by the pinned reference graph."""

    def __init__(self, input_size: tuple[int, int]) -> None:
        super().__init__()
        self.num_heads = NUM_HEADS
        self.scale = (HIDDEN_SIZE // NUM_HEADS) ** -0.5
        self.qkv = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE * 3, bias=True)
        self.proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        head_size = HIDDEN_SIZE // NUM_HEADS
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_size))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = hidden_state.shape
        qkv = (
            self.qkv(hidden_state)
            .reshape(batch, height * width, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        queries, keys, values = qkv.reshape(
            3, batch * self.num_heads, height * width, -1
        ).unbind(0)

        attention = (queries * self.scale) @ keys.transpose(-2, -1)
        attention = add_decomposed_relative_positions(
            attention,
            queries,
            self.rel_pos_h,
            self.rel_pos_w,
            (height, width),
            (height, width),
        )
        probabilities = attention.softmax(dim=-1)
        hidden_state = probabilities @ values
        hidden_state = hidden_state.view(
            batch, self.num_heads, height, width, channels // self.num_heads
        )
        hidden_state = hidden_state.permute(0, 2, 3, 1, 4).reshape(
            batch, height, width, channels
        )
        return self.proj(hidden_state)


class VitDetLayerNorm(nn.Module):
    """Channel-first LayerNorm used by ViTDet residual bottlenecks."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mean = value.mean(1, keepdim=True)
        variance = (value - mean).pow(2).mean(1, keepdim=True)
        normalized = (value - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * normalized + self.bias[:, None, None]


class VitDetResBottleneckBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bottleneck = HIDDEN_SIZE // 2
        self.conv1 = nn.Conv2d(HIDDEN_SIZE, bottleneck, 1, bias=False)
        self.norm1 = VitDetLayerNorm(bottleneck)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1, bias=False)
        self.norm2 = VitDetLayerNorm(bottleneck)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(bottleneck, HIDDEN_SIZE, 1, bias=False)
        self.norm3 = VitDetLayerNorm(HIDDEN_SIZE)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.act1(self.norm1(self.conv1(value)))
        value = self.act2(self.norm2(self.conv2(value)))
        value = self.norm3(self.conv3(value))
        return residual + value


class VitDetMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(HIDDEN_SIZE * 4, HIDDEN_SIZE)
        self.drop = nn.Dropout(0.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.drop(self.act(self.fc1(value)))
        return self.drop(self.fc2(value))


def window_partition(
    hidden_state: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    batch, height, width, channels = hidden_state.shape
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    hidden_state = F.pad(hidden_state, (0, 0, 0, pad_width, 0, pad_height))
    padded_height, padded_width = height + pad_height, width + pad_width
    hidden_state = hidden_state.view(
        batch,
        padded_height // window_size,
        window_size,
        padded_width // window_size,
        window_size,
        channels,
    )
    windows = (
        hidden_state.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, channels)
    )
    return windows, (padded_height, padded_width)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    padded_size: tuple[int, int],
    original_size: tuple[int, int],
) -> torch.Tensor:
    padded_height, padded_width = padded_size
    height, width = original_size
    batch = windows.shape[0] // (
        padded_height * padded_width // window_size // window_size
    )
    hidden_state = windows.view(
        batch,
        padded_height // window_size,
        padded_width // window_size,
        window_size,
        window_size,
        -1,
    )
    hidden_state = (
        hidden_state.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(batch, padded_height, padded_width, -1)
    )
    return hidden_state[:, :height, :width, :].contiguous()


class VitDetLayer(nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(HIDDEN_SIZE, eps=1e-6)
        self.window_size = WINDOW_SIZE if index in WINDOW_BLOCK_INDICES else 0
        attention_size = (
            (self.window_size, self.window_size)
            if self.window_size
            else (512 // PATCH_SIZE, 512 // PATCH_SIZE)
        )
        self.attention = VitDetAttention(attention_size)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(HIDDEN_SIZE, eps=1e-6)
        self.mlp = VitDetMlp()
        self.residual = (
            VitDetResBottleneckBlock() if index in RESIDUAL_BLOCK_INDICES else None
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = hidden_state.permute(0, 2, 3, 1)
        shortcut = hidden_state
        value = self.norm1(hidden_state)

        if self.window_size:
            height, width = int(value.shape[1]), int(value.shape[2])
            value, padded_size = window_partition(value, self.window_size)
        value = self.attention(value)
        if self.window_size:
            value = window_unpartition(
                value,
                self.window_size,
                padded_size,
                (height, width),
            )

        hidden_state = shortcut + self.drop_path(value)
        hidden_state = hidden_state + self.drop_path(self.mlp(self.norm2(hidden_state)))
        hidden_state = hidden_state.permute(0, 3, 1, 2)
        if self.residual is not None:
            hidden_state = self.residual(hidden_state)
        return hidden_state


class VitDetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList([VitDetLayer(index) for index in range(NUM_LAYERS)])

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            hidden_state = layer(hidden_state)
        return hidden_state


class VitDetBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = VitDetEmbeddings()
        self.encoder = VitDetEncoder()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embeddings(pixel_values))


class VitMatteBasicConv3x3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.relu = nn.ReLU()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.relu(self.batch_norm(self.conv(hidden_state)))


class VitMatteConvStream(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = (4, 48, 96, 192)
        self.convs = nn.ModuleList(
            [
                VitMatteBasicConv3x3(channels[index], channels[index + 1])
                for index in range(len(channels) - 1)
            ]
        )

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        features = [pixel_values]
        hidden_state = pixel_values
        for convolution in self.convs:
            hidden_state = convolution(hidden_state)
            features.append(hidden_state)
        return features


class VitMatteFusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = VitMatteBasicConv3x3(
            in_channels,
            out_channels,
            stride=1,
            padding=1,
        )

    def forward(
        self,
        features: torch.Tensor,
        detailed_feature_map: torch.Tensor,
    ) -> torch.Tensor:
        features = F.interpolate(
            features,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        return self.conv(torch.cat([detailed_feature_map, features], dim=1))


class VitMatteHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matting_convs = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.matting_convs(hidden_state)


class VitMatteDetailCaptureModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convstream = VitMatteConvStream()
        self.fusion_blocks = nn.ModuleList(
            [
                VitMatteFusionBlock(384 + 192, 256),
                VitMatteFusionBlock(256 + 96, 128),
                VitMatteFusionBlock(128 + 48, 64),
                VitMatteFusionBlock(64 + 4, 32),
            ]
        )
        self.matting_head = VitMatteHead()

    def forward(
        self,
        features: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        detail_features = self.convstream(pixel_values)
        for index, fusion in enumerate(self.fusion_blocks):
            features = fusion(features, detail_features[3 - index])
        # This is the graph's one and only sigmoid. Callers receive alpha
        # probabilities, not logits.
        return torch.sigmoid(self.matting_head(features))


def constrain_alpha_to_trimap(
    alpha: torch.Tensor,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """Set known trimap background/foreground exactly to zero/one."""
    trimap = pixel_values[:, 3:4]
    alpha = torch.where(trimap == 0.0, torch.zeros_like(alpha), alpha)
    return torch.where(trimap == 1.0, torch.ones_like(alpha), alpha)


class LibreViTMatteModel(nn.Module):
    """ViTMatte-S backbone and detail-capture decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = VitDetBackbone()
        self.decoder = VitMatteDetailCaptureModule()

    def forward_unconstrained(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values)
        return self.decoder(features, pixel_values)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        alpha = self.forward_unconstrained(pixel_values)
        return constrain_alpha_to_trimap(alpha, pixel_values)


__all__ = [
    "LibreViTMatteModel",
    "constrain_alpha_to_trimap",
]
