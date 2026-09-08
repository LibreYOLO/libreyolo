"""Native inference graph for the LeVJEPA video encoder.

This module is independently implemented from the published architecture and
checkpoint tensor contract. It does not contain code from LeVJEPA's
CC-BY-NC-4.0 ``module.py`` or Hugging Face remote-code implementation.

The factorized three-dimensional rotary operation follows LibreYOLO's
Apache-2.0-derived V-JEPA 2 implementation. See ``NOTICE`` in this directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F


LEVJEPA_CONFIGS: dict[str, dict[str, int | float | bool | str]] = {
    "l": {
        "img_size": 224,
        "patch_size": 16,
        "num_frames": 16,
        "tubelet_size": 1,
        "in_chans": 3,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "attn_mode": "block_causal",
    }
}


@dataclass(frozen=True)
class LeVJEPAConfig:
    """Inference-time architecture parameters for one LeVJEPA encoder."""

    img_size: int = 224
    patch_size: int = 16
    num_frames: int = 16
    tubelet_size: int = 1
    in_chans: int = 3
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    attn_mode: str = "block_causal"
    layer_norm_eps: float = 1e-6

    SUPPORTED_ATTENTION: ClassVar[tuple[str, ...]] = ("block_causal", "full")

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.img_size % self.patch_size:
            raise ValueError("img_size must be divisible by patch_size")
        if self.num_frames % self.tubelet_size:
            raise ValueError("num_frames must be divisible by tubelet_size")
        if self.attn_mode not in self.SUPPORTED_ATTENTION:
            raise ValueError(
                f"attn_mode must be one of {self.SUPPORTED_ATTENTION}, "
                f"got {self.attn_mode!r}"
            )

    @classmethod
    def for_size(cls, size: str, **overrides) -> "LeVJEPAConfig":
        if size not in LEVJEPA_CONFIGS:
            raise ValueError(
                f"unknown LeVJEPA size {size!r}; expected one of "
                f"{sorted(LEVJEPA_CONFIGS)}"
            )
        values = dict(LEVJEPA_CONFIGS[size])
        values.update(overrides)
        return cls(**values)

    @property
    def grid_size(self) -> int:
        return self.img_size // self.patch_size

    @property
    def temporal_grid_size(self) -> int:
        return self.num_frames // self.tubelet_size

    @property
    def patch_tokens(self) -> int:
        return self.temporal_grid_size * self.grid_size * self.grid_size


class LeVJEPAPatchEmbed(nn.Module):
    """Non-overlapping per-frame spatiotemporal patch embedding."""

    def __init__(self, config: LeVJEPAConfig):
        super().__init__()
        self.proj = nn.Conv3d(
            config.in_chans,
            config.embed_dim,
            kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
            stride=(config.tubelet_size, config.patch_size, config.patch_size),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.proj(video).flatten(2).transpose(1, 2)


def rotate_queries_or_keys(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Apply one axis of the factorized rotary embedding."""

    width = x.shape[-1]
    omega = torch.arange(width // 2, dtype=x.dtype, device=x.device)
    omega = 1.0 / (10000 ** (omega / (width / 2.0)))
    frequency = pos.to(dtype=x.dtype).unsqueeze(-1) * omega
    sin = frequency.sin().repeat(1, 1, 1, 2)
    cos = frequency.cos().repeat(1, 1, 1, 2)

    pairs = x.unflatten(-1, (-1, 2))
    first, second = pairs.unbind(dim=-1)
    rotated = torch.stack((-second, first), dim=-1).flatten(-2)
    return x * cos + rotated * sin


class LeVJEPAAttention(nn.Module):
    """Fused-QKV attention with factorized 3D RoPE."""

    def __init__(self, config: LeVJEPAConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(
            config.embed_dim,
            3 * config.embed_dim,
            bias=config.qkv_bias,
        )
        self.proj = nn.Linear(config.embed_dim, config.embed_dim, bias=True)

        axis_dim = 2 * ((self.head_dim // 3) // 2)
        self.temporal_dim = axis_dim
        self.height_dim = axis_dim
        self.width_dim = axis_dim

    def _apply_rotary(
        self,
        tensor: torch.Tensor,
        positions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # CLS is a readout token without a spatiotemporal coordinate.
        cls, patches = tensor[..., :1, :], tensor[..., 1:, :]
        temporal, height, width = positions
        offset = 0
        parts = []
        for axis_width, position in (
            (self.temporal_dim, temporal),
            (self.height_dim, height),
            (self.width_dim, width),
        ):
            part = patches[..., offset : offset + axis_width]
            parts.append(rotate_queries_or_keys(part, position))
            offset += axis_width
        if offset < self.head_dim:
            parts.append(patches[..., offset:])
        return torch.cat((cls, torch.cat(parts, dim=-1)), dim=-2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, tokens, channels = hidden_states.shape
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query = self._apply_rotary(query, positions)
        key = self._apply_rotary(key, positions)

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )
        output = output.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(output)


class LeVJEPAMLP(nn.Module):
    def __init__(self, config: LeVJEPAConfig):
        super().__init__()
        hidden = int(config.embed_dim * config.mlp_ratio)
        self.fc1 = nn.Linear(config.embed_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, config.embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states)))


class LeVJEPABlock(nn.Module):
    def __init__(self, config: LeVJEPAConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.attn = LeVJEPAAttention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.mlp = LeVJEPAMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), positions, attention_mask
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


def _patch_positions(config: LeVJEPAConfig) -> tuple[torch.Tensor, ...]:
    temporal = torch.arange(config.temporal_grid_size)
    height = torch.arange(config.grid_size)
    width = torch.arange(config.grid_size)
    grid = torch.meshgrid(temporal, height, width, indexing="ij")
    return tuple(axis.reshape(1, 1, -1) for axis in grid)


def _block_causal_mask(config: LeVJEPAConfig) -> torch.Tensor | None:
    if config.attn_mode == "full":
        return None
    tokens_per_frame = config.grid_size * config.grid_size
    frame_ids = torch.arange(config.temporal_grid_size).repeat_interleave(
        tokens_per_frame
    )
    patch_mask = frame_ids[:, None] >= frame_ids[None, :]
    mask = torch.zeros(
        config.patch_tokens + 1,
        config.patch_tokens + 1,
        dtype=torch.bool,
    )
    mask[0, :] = True
    mask[1:, 1:] = patch_mask
    return mask


class LeVJEPAEncoder(nn.Module):
    """LeVJEPA encoder returning ``CLS + time-major patch tokens``."""

    def __init__(self, config: LeVJEPAConfig):
        super().__init__()
        self.config = config
        self.patch_embed = LeVJEPAPatchEmbed(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.blocks = nn.ModuleList([LeVJEPABlock(config) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        positions = _patch_positions(config)
        self.register_buffer("rope_temporal", positions[0], persistent=False)
        self.register_buffer("rope_height", positions[1], persistent=False)
        self.register_buffer("rope_width", positions[2], persistent=False)
        self.register_buffer(
            "block_causal_mask", _block_causal_mask(config), persistent=False
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(
                f"LeVJEPA expects (B, F, C, H, W), got {tuple(pixel_values.shape)}"
            )
        expected = (
            self.config.num_frames,
            self.config.in_chans,
            self.config.img_size,
            self.config.img_size,
        )
        if tuple(pixel_values.shape[1:]) != expected:
            raise ValueError(
                f"LeVJEPA expects clip shape (B, {expected[0]}, {expected[1]}, "
                f"{expected[2]}, {expected[3]}), got {tuple(pixel_values.shape)}"
            )

        target_dtype = self.patch_embed.proj.weight.dtype
        video = pixel_values.to(dtype=target_dtype).permute(0, 2, 1, 3, 4)
        patches = self.patch_embed(video)
        cls = self.cls_token.expand(patches.shape[0], -1, -1)
        hidden_states = torch.cat((cls, patches), dim=1)
        positions = (self.rope_temporal, self.rope_height, self.rope_width)
        for block in self.blocks:
            hidden_states = block(hidden_states, positions, self.block_causal_mask)
        return self.norm(hidden_states)


class LeVJEPAModel(nn.Module):
    """Checkpoint-compatible root module containing the encoder."""

    def __init__(self, config: LeVJEPAConfig):
        super().__init__()
        self.config = config
        self.encoder = LeVJEPAEncoder(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values)
