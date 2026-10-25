# SPDX-License-Identifier: Apache-2.0
# Ported from https://github.com/Tencent/WeVisOne-VL
# (wevisone_vl/mask_decoder.py, wevisone_vl/config.py, wevisone_vl/decoder.py)
# at commit 9a6bcd85404505df0d05cfb5d97512ccacf35dee.
# Copyright (C) 2026 Tencent. All rights reserved.
# Modified for LibreYOLO: inference-only (dropout and training-time
# initializers removed), the deformable cross-attention reuses LibreYOLO's
# Apache-2.0 Deformable-DETR port (libreyolo/models/deformable_detr), and the
# stem is fixed to the released output stride of 2. Module and attribute names
# are unchanged so the released mask_decoder.pt state dict loads strictly.
"""WeVisOne-VL pixel decoder: ``<OBJ>`` hidden states + boxes -> alpha mattes."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..deformable_detr.ms_deform_attn import MSDeformAttn

ARCHITECTURE_VERSION = "wevisone_vl_v1"


@dataclass
class Qwen3VLSegConfig:
    """Fields that instantiate the released decoder (``mask_decoder.pt`` cfg)."""

    architecture_version: str = ARCHITECTURE_VERSION
    vision_feature_layers: Tuple[int, ...] = (5, 11, 17, 23)
    hidden_dim: int = 192
    mask_feature_dim: int = 64
    num_fourier_freqs: int = 32
    msda_encoder_layers: int = 3
    msda_num_levels: int = 4
    msda_num_points: int = 4
    transformer_layers: int = 6
    transformer_heads: int = 8
    transformer_ffn_dim: int = 768
    transformer_dropout: float = 0.0
    group_norm_groups: int = 32
    stem_mid_dim: int = 128
    alpha_cascade_dim: int = 48
    alpha_cascade_blocks: int = 2
    cnn_image_size: int = 1024

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]]) -> "Qwen3VLSegConfig":
        if not values:
            return cls()
        known = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in values.items() if k in known})
        cfg.vision_feature_layers = tuple(int(v) for v in cfg.vision_feature_layers)
        return cfg


def _group_norm(channels: int, groups: int) -> nn.GroupNorm:
    groups = min(int(groups), int(channels))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _mask_feature_dim(cfg: Qwen3VLSegConfig) -> int:
    value = int(cfg.mask_feature_dim or 0)
    return value if value > 0 else int(cfg.hidden_dim)


class FourierPE1D(nn.Module):
    def __init__(self, num_freqs: int):
        super().__init__()
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(num_freqs, dtype=torch.float32),
            persistent=False,
        )

    @property
    def out_dim(self) -> int:
        return 2 * int(self.frequencies.numel())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * values[..., None].float() * self.frequencies
        return torch.cat([phase.sin(), phase.cos()], dim=-1).to(values.dtype)


class BoxFourierEncoder(nn.Module):
    """Fixed-frequency x1 / y1 / log(w) / log(h) box encoding."""

    def __init__(self, hidden_dim: int, num_freqs: int):
        super().__init__()
        self.pe = FourierPE1D(num_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(4 * self.pe.out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
        encoded_w = 0.2 * (x2 - x1).clamp_min(1e-6).log() + 0.5
        encoded_h = 0.2 * (y2 - y1).clamp_min(1e-6).log() + 0.5
        encoded = torch.cat(
            [self.pe(x1), self.pe(y1), self.pe(encoded_w), self.pe(encoded_h)], dim=-1
        )
        return self.mlp(encoded)


class MaskQueryBuilder(nn.Module):
    """Combine the LLM ``<OBJ>`` content with a Fourier box position."""

    def __init__(self, llm_dim: int, hidden_dim: int, num_freqs: int):
        super().__init__()
        self.content_proj = nn.Linear(llm_dim, hidden_dim)
        self.box_position = BoxFourierEncoder(hidden_dim, num_freqs)
        self.content_norm = nn.LayerNorm(hidden_dim)
        self.position_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, t_seg: torch.Tensor, boxes_xyxy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        content = self.content_norm(self.content_proj(t_seg))
        position = self.position_norm(self.box_position(boxes_xyxy))
        return content, position


class PositionEmbeddingSine2D(nn.Module):
    """Normalized 2-D sine/cosine position encoding over a token grid."""

    def __init__(self, hidden_dim: int, temperature: float = 10000.0):
        super().__init__()
        self.num_pos_feats = hidden_dim // 2
        self.temperature = float(temperature)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feature.shape
        y = torch.arange(h, device=feature.device, dtype=torch.float32) + 0.5
        x = torch.arange(w, device=feature.device, dtype=torch.float32) + 0.5
        y = y / max(float(h), 1.0) * (2.0 * math.pi)
        x = x / max(float(w), 1.0) * (2.0 * math.pi)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dim_t = torch.arange(
            self.num_pos_feats, device=feature.device, dtype=torch.float32
        )
        dim_t = self.temperature ** (
            2.0 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )
        pos_x = xx[..., None] / dim_t
        pos_y = yy[..., None] / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1)
        pos = torch.cat([pos_y.flatten(-2), pos_x.flatten(-2)], dim=-1)
        return pos.permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1).to(feature.dtype)


def _spatial_metadata(
    features: Sequence[torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    shapes = torch.tensor(
        [[int(x.shape[-2]), int(x.shape[-1])] for x in features],
        device=device,
        dtype=torch.int64,
    )
    sizes = shapes[:, 0] * shapes[:, 1]
    starts = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)[:-1]])
    return shapes, starts


def _boxes_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.clamp(0.0, 1.0).unbind(-1)
    return torch.stack(
        [
            (x1 + x2) * 0.5,
            (y1 + y2) * 0.5,
            (x2 - x1).clamp_min(1e-4),
            (y2 - y1).clamp_min(1e-4),
        ],
        dim=-1,
    )


class SpatialConvEncoderLayer(nn.Module):
    """Depthwise-conv token mixer plus FFN, run per level on its own grid."""

    def __init__(self, cfg: Qwen3VLSegConfig):
        super().__init__()
        d = int(cfg.hidden_dim)
        self.depthwise = nn.Conv2d(d, d, 3, padding=1, groups=d, bias=False)
        self.pointwise = nn.Conv2d(d, d, 1, bias=False)
        self.norm1 = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, cfg.transformer_ffn_dim)
        self.linear2 = nn.Linear(cfg.transformer_ffn_dim, d)
        self.norm2 = nn.LayerNorm(d)

    def forward(
        self, src: torch.Tensor, pos: torch.Tensor, spatial_shapes: torch.Tensor
    ) -> torch.Tensor:
        shapes = [(int(h), int(w)) for h, w in spatial_shapes.tolist()]
        mixed = []
        for tokens, (height, width) in zip(
            (src + pos).split([h * w for h, w in shapes], dim=1), shapes
        ):
            feature = tokens.transpose(1, 2).reshape(
                tokens.shape[0], tokens.shape[-1], height, width
            )
            feature = self.pointwise(F.gelu(self.depthwise(feature)))
            mixed.append(feature.flatten(2).transpose(1, 2))
        src = self.norm1(src + torch.cat(mixed, dim=1))
        return self.norm2(src + self.linear2(F.gelu(self.linear1(src))))


class DINOEncoder(nn.Module):
    def __init__(self, cfg: Qwen3VLSegConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [SpatialConvEncoderLayer(cfg) for _ in range(cfg.msda_encoder_layers)]
        )

    def forward(
        self, src: torch.Tensor, pos: torch.Tensor, spatial_shapes: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, pos, spatial_shapes)
        return src


class DINODecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3VLSegConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.self_attn = nn.MultiheadAttention(
            d, cfg.transformer_heads, batch_first=True
        )
        # 192 / 8 = 24 channels per head is fixed by the checkpoint.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="A power-of-two dimension")
            self.cross_attn = MSDeformAttn(
                d_model=d,
                n_levels=cfg.msda_num_levels,
                n_heads=cfg.transformer_heads,
                n_points=cfg.msda_num_points,
            )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, cfg.transformer_ffn_dim)
        self.linear2 = nn.Linear(cfg.transformer_ffn_dim, d)

    def forward(
        self,
        target: torch.Tensor,
        query_pos: torch.Tensor,
        references: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        qk = target + query_pos
        target = self.norm1(
            target + self.self_attn(qk, qk, target, need_weights=False)[0]
        )
        cross = self.cross_attn(
            target + query_pos, references, memory, spatial_shapes, level_start_index
        )
        target = self.norm2(target + cross)
        return self.norm3(target + self.linear2(F.gelu(self.linear1(target))))


class DINOMaskDecoder(nn.Module):
    def __init__(self, cfg: Qwen3VLSegConfig):
        super().__init__()
        self.num_levels = cfg.msda_num_levels
        self.layers = nn.ModuleList(
            [DINODecoderLayer(cfg) for _ in range(cfg.transformer_layers)]
        )

    def forward(
        self,
        target: torch.Tensor,
        query_pos: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        references = _boxes_xyxy_to_cxcywh(boxes_xyxy)
        references = references[None, :, None].expand(
            target.shape[0], -1, self.num_levels, -1
        )
        for layer in self.layers:
            target = layer(
                target, query_pos, references, memory, spatial_shapes, level_start_index
            )
        return target


class SameScaleMSDAEncoder(nn.Module):
    """Project the raw ViT levels and run the convolutional encoder."""

    def __init__(self, visual_in_dims: Sequence[int], cfg: Qwen3VLSegConfig):
        super().__init__()
        if len(visual_in_dims) != cfg.msda_num_levels:
            raise ValueError(
                f"decoder expects {cfg.msda_num_levels} ViT levels, "
                f"got {len(visual_in_dims)}"
            )
        d = cfg.hidden_dim
        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(int(c), d, kernel_size=1, bias=False),
                    _group_norm(d, cfg.group_norm_groups),
                )
                for c in visual_in_dims
            ]
        )
        self.position = PositionEmbeddingSine2D(d)
        self.level_embed = nn.Parameter(torch.zeros(len(visual_in_dims), d))
        self.encoder = DINOEncoder(cfg)

    def forward(
        self, features: Sequence[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = [proj(feature) for proj, feature in zip(self.input_proj, features)]
        spatial_shapes, starts = _spatial_metadata(projected, projected[0].device)
        src = torch.cat([f.flatten(2).transpose(1, 2) for f in projected], dim=1)
        pos = torch.cat(
            [
                (self.position(f) + self.level_embed[level][None, :, None, None])
                .flatten(2)
                .transpose(1, 2)
                for level, f in enumerate(projected)
            ],
            dim=1,
        )
        memory = self.encoder(src, pos, spatial_shapes)
        shapes = [(int(h), int(w)) for h, w in spatial_shapes.tolist()]
        chunks = memory.split([h * w for h, w in shapes], dim=1)
        maps = [
            chunk.transpose(1, 2).reshape(chunk.shape[0], chunk.shape[-1], h, w)
            for chunk, (h, w) in zip(chunks, shapes)
        ]
        return maps, memory, spatial_shapes, starts


class SameScaleFPN(nn.Module):
    """Top-down fusion of ViT levels that share one token grid."""

    def __init__(self, hidden_dim: int, num_levels: int, gn_groups: int):
        super().__init__()
        self.lateral = nn.ModuleList(
            [
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
                for _ in range(num_levels)
            ]
        )
        self.output = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                    _group_norm(hidden_dim, gn_groups),
                    nn.GELU(),
                )
                for _ in range(num_levels - 1)
            ]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        fused = self.lateral[-1](features[-1])
        for output_idx, level in enumerate(range(len(features) - 2, -1, -1)):
            fused = self.output[output_idx](
                fused + self.lateral[level](features[level])
            )
        return fused


class ShallowImageStem(nn.Module):
    """RGB stem at output stride 2 (the released configuration)."""

    def __init__(self, hidden_dim: int, mid_dim: int, gn_groups: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, mid_dim, 3, stride=2, padding=1, bias=False),
            _group_norm(mid_dim, gn_groups),
            nn.GELU(),
            nn.Conv2d(mid_dim, hidden_dim, 3, stride=1, padding=1, bias=False),
            _group_norm(hidden_dim, gn_groups),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class PixelShuffleStage(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(int(in_dim), int(out_dim) * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.shuffle(self.conv(feature)))


class ResidualPixelShuffleStage(nn.Module):
    """Bilinear 2x upsample plus a learned sub-pixel residual."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(int(channels), int(channels) * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(
            feature, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        return base + self.shuffle(self.conv(feature))


class AdditivePixelFusion(nn.Module):
    """Fuse upsampled ViT semantics with an RGB stem at stride 2."""

    def __init__(self, cfg: Qwen3VLSegConfig):
        super().__init__()
        hidden_dim = int(cfg.hidden_dim)
        mask_dim = _mask_feature_dim(cfg)
        groups = int(cfg.group_norm_groups)
        self.fpn = SameScaleFPN(hidden_dim, cfg.msda_num_levels, groups)
        self.upsample = nn.Sequential(
            PixelShuffleStage(hidden_dim, hidden_dim),
            PixelShuffleStage(hidden_dim, mask_dim),
        )
        self.s2_upsample = ResidualPixelShuffleStage(mask_dim)
        self.cnn = ShallowImageStem(mask_dim, cfg.stem_mid_dim, groups)
        self.fpn_proj = nn.Conv2d(mask_dim, mask_dim, 1, bias=False)
        self.cnn_proj = nn.Conv2d(mask_dim, mask_dim, 1, bias=False)
        self.smooth = nn.Sequential(
            _group_norm(mask_dim, groups),
            nn.GELU(),
            nn.Conv2d(mask_dim, mask_dim, 3, padding=1, bias=False),
        )

    def forward(
        self, image: torch.Tensor, encoded_levels: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        fpn = self.upsample(self.fpn(encoded_levels))
        cnn = self.cnn(image)
        dst_h, dst_w = (max(1, int(v) // 2) for v in cnn.shape[-2:])
        src_h, src_w = (int(v) for v in fpn.shape[-2:])
        if (src_h, src_w) != (dst_h, dst_w):
            pool_h, pool_w = min(src_h, dst_h), min(src_w, dst_w)
            if (pool_h, pool_w) != (src_h, src_w):
                fpn = F.adaptive_avg_pool2d(fpn, (pool_h, pool_w))
            if (pool_h, pool_w) != (dst_h, dst_w):
                fpn = F.interpolate(
                    fpn, size=(dst_h, dst_w), mode="bilinear", align_corners=False
                )
        fused = self.s2_upsample(self.smooth(self.fpn_proj(fpn)))
        return fused + self.cnn_proj(cnn)


class DynamicMaskPredictor(nn.Module):
    def __init__(self, hidden_dim: int, mask_dim: int):
        super().__init__()
        self.kernel = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mask_dim + 1),
        )

    def forward(self, queries: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        params = self.kernel(queries)
        kernels, bias = params[:, :-1], params[:, -1]
        return torch.einsum("qc,chw->qhw", kernels, feature[0]) + bias[:, None, None]


class AlphaResidualBlock(nn.Module):
    def __init__(self, channels: int, gn_groups: int):
        super().__init__()
        self.net = nn.Sequential(
            _group_norm(channels, gn_groups),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _group_norm(channels, gn_groups),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.net(feature)


class FullCanvasAlphaCascadeHead(nn.Module):
    """Refine the stride-2 prediction at strides 4, 2 and 1 over the full image."""

    def __init__(self, cfg: Qwen3VLSegConfig):
        super().__init__()
        mask_dim = _mask_feature_dim(cfg)
        hidden = int(cfg.hidden_dim)
        channels = int(cfg.alpha_cascade_dim)
        groups = min(int(cfg.group_norm_groups), channels)
        blocks = int(cfg.alpha_cascade_blocks)

        def project(in_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                _group_norm(channels, groups),
                nn.GELU(),
            )

        def residual_stack() -> nn.Sequential:
            return nn.Sequential(
                *[AlphaResidualBlock(channels, groups) for _ in range(blocks)]
            )

        self.semantic = project(mask_dim)
        self.rgb_s4 = project(3)
        self.rgb_s2 = project(3)
        self.coarse_s4 = project(1)
        self.coarse_s2 = project(1)
        self.previous_s2 = project(1)
        self.previous_s1 = project(1)
        self.fuse_s4 = nn.Conv2d(3 * channels, channels, 1, bias=False)
        self.blocks_s4 = residual_stack()
        self.up_s4_to_s2 = nn.ConvTranspose2d(
            channels, channels, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.fuse_s2 = nn.Conv2d(5 * channels, channels, 1, bias=False)
        self.blocks_s2 = residual_stack()
        self.up_s2_to_s1 = nn.ConvTranspose2d(
            channels, channels, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.rgb_s1 = project(3)
        self.fuse_s1 = nn.Conv2d(4 * channels, channels, 1, bias=False)
        self.blocks_s1 = residual_stack()
        self.query_s4 = nn.Linear(hidden, 2 * channels)
        self.query_s2 = nn.Linear(hidden, 2 * channels)
        self.query_s1 = nn.Linear(hidden, 2 * channels)
        self.output_s4 = nn.Conv2d(channels, 1, 1)
        self.output_s2 = nn.Conv2d(channels, 1, 1)
        self.output_s1 = nn.Conv2d(channels, 1, 1)

    @staticmethod
    def _resize(value: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if tuple(value.shape[-2:]) == tuple(size):
            return value
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def _condition(
        feature: torch.Tensor, queries: torch.Tensor, affine: nn.Linear
    ) -> torch.Tensor:
        scale, bias = affine(queries).chunk(2, dim=-1)
        return (
            feature * (1.0 + 0.1 * scale.tanh()[:, :, None, None])
            + 0.1 * bias[:, :, None, None]
        )

    def forward(
        self,
        image: torch.Tensor,
        pixel_feature: torch.Tensor,
        queries: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> torch.Tensor:
        n = int(queries.shape[0])
        s2_hw = tuple(int(x) for x in coarse_logits.shape[-2:])
        s1_hw = tuple(int(x) for x in image.shape[-2:])
        if s1_hw != (2 * s2_hw[0], 2 * s2_hw[1]):
            raise ValueError(
                f"alpha cascade needs a stride-2 canvas: image={s1_hw}, coarse={s2_hw}"
            )
        s4_hw = (max(1, s2_hw[0] // 2), max(1, s2_hw[1] // 2))

        semantic_s2 = self.semantic(pixel_feature)
        semantic_s4 = self._resize(semantic_s2, s4_hw)
        coarse_s4 = self._resize(coarse_logits[:, None], s4_hw)[:, 0]
        feature_s4 = self.fuse_s4(
            torch.cat(
                [
                    semantic_s4.expand(n, -1, -1, -1),
                    self.rgb_s4(self._resize(image, s4_hw)).expand(n, -1, -1, -1),
                    self.coarse_s4(coarse_s4[:, None]),
                ],
                dim=1,
            )
        )
        feature_s4 = self.blocks_s4(self._condition(feature_s4, queries, self.query_s4))
        logits_s4 = coarse_s4 + self.output_s4(feature_s4)[:, 0]

        previous_s2 = self._resize(logits_s4[:, None], s2_hw)
        feature_s2 = self.fuse_s2(
            torch.cat(
                [
                    self.up_s4_to_s2(feature_s4),
                    semantic_s2.expand(n, -1, -1, -1),
                    self.rgb_s2(self._resize(image, s2_hw)).expand(n, -1, -1, -1),
                    self.coarse_s2(coarse_logits[:, None]),
                    self.previous_s2(previous_s2.sigmoid()),
                ],
                dim=1,
            )
        )
        feature_s2 = self.blocks_s2(self._condition(feature_s2, queries, self.query_s2))
        logits_s2 = coarse_logits + self.output_s2(feature_s2)[:, 0]

        previous_s1 = self._resize(logits_s2[:, None], s1_hw)
        feature_s1 = self.fuse_s1(
            torch.cat(
                [
                    self.up_s2_to_s1(feature_s2),
                    self._resize(semantic_s2, s1_hw).expand(n, -1, -1, -1),
                    self.rgb_s1(image).expand(n, -1, -1, -1),
                    self.previous_s1(previous_s1.sigmoid()),
                ],
                dim=1,
            )
        )
        feature_s1 = self.blocks_s1(self._condition(feature_s1, queries, self.query_s1))
        return previous_s1[:, 0] + self.output_s1(feature_s1)[:, 0]


class WeVisOneMaskDecoder(nn.Module):
    """Released WeVisOne-VL v1 decoder (about 9.6M parameters)."""

    def __init__(self, visual_in_dims: List[int], llm_dim: int, cfg: Qwen3VLSegConfig):
        super().__init__()
        self.visual_encoder = SameScaleMSDAEncoder(visual_in_dims, cfg)
        self.query_builder = MaskQueryBuilder(
            llm_dim, cfg.hidden_dim, cfg.num_fourier_freqs
        )
        self.transformer = DINOMaskDecoder(cfg)
        self.pixel_fusion = AdditivePixelFusion(cfg)
        self.mask_predictor = DynamicMaskPredictor(
            cfg.hidden_dim, _mask_feature_dim(cfg)
        )
        self.highres_refine = FullCanvasAlphaCascadeHead(cfg)

    def forward(
        self,
        image: torch.Tensor,
        visual_features_2d: List[torch.Tensor],
        t_seg: torch.Tensor,
        boxes01: torch.Tensor,
    ) -> torch.Tensor:
        """Return full-resolution mask logits ``(Q, S, S)`` for ``Q`` queries.

        ``image`` is the RGB canvas ``(1, 3, S, S)`` in ``[0, 1]``; the ViT maps
        are ``(1, C, h, w)`` pre-merge grids; ``t_seg`` holds the ``<OBJ>``
        hidden states and ``boxes01`` the matching normalized ``xyxy`` boxes.
        The released weights are FP32 and run in FP32 regardless of the
        backbone dtype.
        """
        if t_seg.shape[0] != boxes01.shape[0]:
            raise ValueError(
                f"#OBJ={t_seg.shape[0]} does not match #boxes={boxes01.shape[0]}"
            )
        with torch.autocast(device_type=image.device.type, enabled=False):
            image = image.float()
            boxes01 = boxes01.float()
            encoded, memory, shapes, starts = self.visual_encoder(
                [f.float() for f in visual_features_2d]
            )
            content, position = self.query_builder(t_seg.float(), boxes01)
            queries = self.transformer(
                content.unsqueeze(0),
                position.unsqueeze(0),
                boxes01,
                memory,
                shapes,
                starts,
            ).squeeze(0)
            pixel_feature = self.pixel_fusion(image, encoded)
            coarse_logits = self.mask_predictor(queries, pixel_feature)
            return self.highres_refine(image, pixel_feature, queries, coarse_logits)


def build_mask_decoder(
    visual_dim: int, llm_dim: int, cfg: Optional[Qwen3VLSegConfig] = None
) -> WeVisOneMaskDecoder:
    """Build the decoder for a backbone's ViT width and LLM hidden size."""
    cfg = cfg or Qwen3VLSegConfig()
    if cfg.architecture_version != ARCHITECTURE_VERSION:
        raise ValueError(
            f"unsupported decoder architecture: {cfg.architecture_version!r}"
        )
    visual_in_dims = [int(visual_dim)] * len(cfg.vision_feature_layers)
    return WeVisOneMaskDecoder(visual_in_dims, int(llm_dim), cfg)
