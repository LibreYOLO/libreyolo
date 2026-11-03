# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0. Ported from
# huggingface/transformers v5.22.1 (commit
# 446a570e62b04a89ea001604ada7366baac2ea67):
# src/transformers/models/pp_doclayout_v3/modeling_pp_doclayout_v3.py and
# src/transformers/models/hgnet_v2/modeling_hgnet_v2.py.
#
# LibreYOLO changes: inference-only pure-PyTorch modules (no transformers
# runtime), module names mirror the official safetensors keys so conversion is
# a prefix strip, eager attention only, the shared heads are tied by reference,
# per-layer auxiliary heads are evaluated only for the last decoder layer
# (their values feed nothing else), and a tuple export wrapper.
"""PP-DocLayoutV3 network: HGNetV2-L, mask-enhanced RT-DETR, reading-order head."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# HGNetV2-L stage layout (the only backbone PP-DocLayoutV3 ships with).
_HGNETV2_L = {
    "stem_channels": (3, 32, 48),
    "stem_strides": (2, 1, 1, 2, 1),
    "stage_in_channels": (48, 128, 512, 1024),
    "stage_mid_channels": (48, 96, 192, 384),
    "stage_out_channels": (128, 512, 1024, 2048),
    "stage_num_blocks": (1, 1, 3, 1),
    "stage_num_layers": (6, 6, 6, 6),
    "stage_downsample": (False, True, True, True),
    "stage_light_block": (False, False, True, True),
    "stage_kernel_size": (3, 3, 5, 5),
}

# Architecture hyperparameters of the released checkpoint (config.json).
DEFAULT_CONFIG = {
    "d_model": 256,
    "encoder_ffn_dim": 1024,
    "encoder_attention_heads": 8,
    "decoder_ffn_dim": 1024,
    "decoder_attention_heads": 8,
    "decoder_layers": 6,
    "decoder_n_points": 4,
    "num_queries": 300,
    "num_prototypes": 32,
    "mask_feature_channels": (64, 64),
    "x4_feat_dim": 128,
    "global_pointer_head_size": 64,
    "feat_strides": (8, 16, 32),
    "layer_norm_eps": 1e-5,
    "batch_norm_eps": 1e-5,
    "positional_encoding_temperature": 10000,
}


# ---------------------------------------------------------------------------
# HGNetV2-L backbone (frozen batch norm, as in the released model)
# ---------------------------------------------------------------------------


class FrozenBatchNorm2d(nn.Module):
    """Batch norm with fixed statistics and affine parameters."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = weight * (running_var + 1e-5).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


class HGNetV2ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.normalization = FrozenBatchNorm2d(out_channels)
        self.activation = nn.ReLU() if activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.normalization(self.convolution(x)))


class HGNetV2ConvLayerLight(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv1 = HGNetV2ConvLayer(
            in_channels, out_channels, kernel_size=1, activation=False
        )
        self.conv2 = HGNetV2ConvLayer(
            out_channels, out_channels, kernel_size=kernel_size, groups=out_channels
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(self.conv1(x))


class HGNetV2Embeddings(nn.Module):
    def __init__(self, stem_channels: Sequence[int], stem_strides: Sequence[int]):
        super().__init__()
        c0, c1, c2 = stem_channels
        self.stem1 = HGNetV2ConvLayer(c0, c1, 3, stride=stem_strides[0])
        self.stem2a = HGNetV2ConvLayer(c1, c1 // 2, 2, stride=stem_strides[1])
        self.stem2b = HGNetV2ConvLayer(c1 // 2, c1, 2, stride=stem_strides[2])
        self.stem3 = HGNetV2ConvLayer(c1 * 2, c1, 3, stride=stem_strides[3])
        self.stem4 = HGNetV2ConvLayer(c1, c2, 1, stride=stem_strides[4])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x: Tensor) -> Tensor:
        embedding = self.stem1(x)
        embedding = F.pad(embedding, (0, 1, 0, 1))
        emb_stem_2a = self.stem2a(embedding)
        emb_stem_2a = F.pad(emb_stem_2a, (0, 1, 0, 1))
        emb_stem_2a = self.stem2b(emb_stem_2a)
        pooled = self.pool(embedding)
        embedding = torch.cat([pooled, emb_stem_2a], dim=1)
        embedding = self.stem3(embedding)
        return self.stem4(embedding)


class HGNetV2BasicLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        layer_num: int,
        kernel_size: int,
        residual: bool,
        light_block: bool,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            block_in = in_channels if i == 0 else middle_channels
            if light_block:
                block = HGNetV2ConvLayerLight(block_in, middle_channels, kernel_size)
            else:
                block = HGNetV2ConvLayer(block_in, middle_channels, kernel_size)
            self.layers.append(block)
        total_channels = in_channels + layer_num * middle_channels
        self.aggregation = nn.Sequential(
            HGNetV2ConvLayer(total_channels, out_channels // 2, 1),
            HGNetV2ConvLayer(out_channels // 2, out_channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        outputs = [x]
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        x = self.aggregation(torch.cat(outputs, dim=1))
        if self.residual:
            x = x + identity
        return x


class HGNetV2Stage(nn.Module):
    def __init__(self, cfg: dict, stage_index: int) -> None:
        super().__init__()
        in_channels = cfg["stage_in_channels"][stage_index]
        mid_channels = cfg["stage_mid_channels"][stage_index]
        out_channels = cfg["stage_out_channels"][stage_index]
        if cfg["stage_downsample"][stage_index]:
            self.downsample = HGNetV2ConvLayer(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                groups=in_channels,
                activation=False,
            )
        else:
            self.downsample = nn.Identity()
        self.blocks = nn.ModuleList(
            HGNetV2BasicLayer(
                in_channels if i == 0 else out_channels,
                mid_channels,
                out_channels,
                cfg["stage_num_layers"][stage_index],
                kernel_size=cfg["stage_kernel_size"][stage_index],
                residual=i != 0,
                light_block=cfg["stage_light_block"][stage_index],
            )
            for i in range(cfg["stage_num_blocks"][stage_index])
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class HGNetV2Encoder(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            HGNetV2Stage(cfg, i) for i in range(len(cfg["stage_in_channels"]))
        )


class HGNetV2Backbone(nn.Module):
    """HGNetV2 returning the output of all four stages (strides 4 to 32)."""

    def __init__(self, cfg: dict = _HGNETV2_L) -> None:
        super().__init__()
        self.embedder = HGNetV2Embeddings(cfg["stem_channels"], cfg["stem_strides"])
        self.encoder = HGNetV2Encoder(cfg)
        self.channels = list(cfg["stage_out_channels"])

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.embedder(x)
        features = []
        for stage in self.encoder.stages:
            x = stage(x)
            features.append(x)
        return features


class ConvEncoder(nn.Module):
    """Holds the backbone under ``model`` to mirror the checkpoint key path."""

    def __init__(self) -> None:
        super().__init__()
        self.model = HGNetV2Backbone()

    def forward(self, x: Tensor) -> List[Tensor]:
        return self.model(x)


# ---------------------------------------------------------------------------
# Hybrid encoder (AIFI + CSP-RepVGG FPN/PAN + mask feature branch)
# ---------------------------------------------------------------------------


class MLPPredictionHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + hidden, hidden + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ConvLayer(nn.Module):
    """Conv + BatchNorm + SiLU (mask branch)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.normalization(self.convolution(x)))


class ConvNormLayer(nn.Module):
    """Conv + BatchNorm + optional SiLU (FPN/PAN)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int | None = None,
        activation: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels, eps)
        self.activation = nn.SiLU() if activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormLayer(channels, channels, 3, 1, padding=1)
        self.conv2 = ConvNormLayer(channels, channels, 1, 1, padding=0)
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.conv1(x) + self.conv2(x))


class CSPRepLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_blocks: int = 3) -> None:
        super().__init__()
        self.conv1 = ConvNormLayer(hidden_dim * 2, hidden_dim, 1, 1, activation=True)
        self.conv2 = ConvNormLayer(hidden_dim * 2, hidden_dim, 1, 1, activation=True)
        self.bottlenecks = nn.Sequential(
            *[RepVggBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.conv3 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.bottlenecks(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(x1 + x2)


def eager_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scaling: float,
    mask: Tensor | None = None,
) -> Tensor:
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if mask is not None:
        weights = weights + mask
    weights = F.softmax(weights, dim=-1)
    return torch.matmul(weights, value).transpose(1, 2).contiguous()


class SelfAttention(nn.Module):
    """Multi-head self-attention; position embeddings go into queries and keys."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, hidden_states: Tensor, position_embeddings: Tensor | None
    ) -> Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        qk_input = (
            hidden_states + position_embeddings
            if position_embeddings is not None
            else hidden_states
        )
        query = self.q_proj(qk_input).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(qk_input).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        output = eager_attention(query, key, value, self.scaling)
        output = output.reshape(*input_shape, -1).contiguous()
        return self.out_proj(output)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.self_attn = SelfAttention(d, cfg["encoder_attention_heads"])
        self.self_attn_layer_norm = nn.LayerNorm(d, eps=cfg["layer_norm_eps"])
        self.fc1 = nn.Linear(d, cfg["encoder_ffn_dim"])
        self.fc2 = nn.Linear(cfg["encoder_ffn_dim"], d)
        self.final_layer_norm = nn.LayerNorm(d, eps=cfg["layer_norm_eps"])

    def forward(self, hidden_states: Tensor, pos_embed: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, pos_embed)
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)
        residual = hidden_states
        hidden_states = self.fc2(F.gelu(self.fc1(hidden_states)))
        return self.final_layer_norm(residual + hidden_states)


def build_2d_sinusoidal_position_embedding(
    height: int,
    width: int,
    embed_dim: int,
    temperature: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """``[sin_h | cos_h | sin_w | cos_w]`` grid embedding, row-major, ``(H*W, D)``.

    The frequencies are computed in float64 except on MPS, which has no
    float64 support; there they use float32.
    """
    pos_dim = embed_dim // 4
    compute_dtype = torch.float32 if device.type == "mps" else torch.float64
    omega = torch.arange(pos_dim, dtype=compute_dtype, device=device) / pos_dim
    omega = 1.0 / temperature**omega
    grid_h = torch.arange(height, dtype=compute_dtype, device=device)
    grid_w = torch.arange(width, dtype=compute_dtype, device=device)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
    emb_h = grid_h.flatten().outer(omega)
    emb_w = grid_w.flatten().outer(omega)
    pos = torch.cat([emb_h.sin(), emb_h.cos(), emb_w.sin(), emb_w.cos()], dim=1)
    return pos.to(dtype)


class AIFILayer(nn.Module):
    """Attention-based intra-scale feature interaction on the stride-32 map."""

    def __init__(self, cfg: dict, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_dim = cfg["d_model"]
        self.temperature = cfg["positional_encoding_temperature"]
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(num_layers))

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        height, width = int(x.shape[2]), int(x.shape[3])
        hidden_states = x.flatten(2).transpose(1, 2)
        pos_embed = build_2d_sinusoidal_position_embedding(
            height, width, self.hidden_dim, self.temperature, x.device, x.dtype
        ).unsqueeze(0)
        for layer in self.layers:
            hidden_states = layer(hidden_states, pos_embed)
        return (
            hidden_states.transpose(1, 2)
            .reshape(batch_size, self.hidden_dim, height, width)
            .contiguous()
        )


class ScaleHead(nn.Module):
    def __init__(
        self, in_channels: int, feature_channels: int, fpn_stride: int, base_stride: int
    ):
        super().__init__()
        head_length = max(1, int(math.log2(fpn_stride) - math.log2(base_stride)))
        self.layers = nn.ModuleList()
        for k in range(head_length):
            self.layers.append(
                ConvLayer(
                    in_channels if k == 0 else feature_channels, feature_channels, 3
                )
            )
            if fpn_stride != base_stride:
                self.layers.append(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MaskFeatFPN(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        fpn_strides: Sequence[int],
        feature_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        order = sorted(range(len(fpn_strides)), key=lambda i: fpn_strides[i])
        self.reorder_index = order
        strides = [fpn_strides[i] for i in order]
        channels = [in_channels[i] for i in order]
        self.scale_heads = nn.ModuleList(
            ScaleHead(channels[i], feature_channels, strides[i], strides[0])
            for i in range(len(strides))
        )
        self.output_conv = ConvLayer(feature_channels, out_channels, 3)

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        x = [inputs[i] for i in self.reorder_index]
        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.scale_heads)):
            output = output + F.interpolate(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.output_conv(output)


class EncoderMaskOutput(nn.Module):
    def __init__(self, in_channels: int, num_prototypes: int) -> None:
        super().__init__()
        self.base_conv = ConvLayer(in_channels, in_channels, 3)
        self.conv = nn.Conv2d(in_channels, num_prototypes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.base_conv(x))


class HybridEncoder(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d = cfg["d_model"]
        strides = list(cfg["feat_strides"])
        num_stages = len(strides) - 1
        # ``encoder`` holds the AIFI stack (applied to the last level only).
        self.encoder = nn.ModuleList([AIFILayer(cfg)])
        self.lateral_convs = nn.ModuleList(
            ConvNormLayer(d, d, 1, 1, activation=True) for _ in range(num_stages)
        )
        self.fpn_blocks = nn.ModuleList(CSPRepLayer(d) for _ in range(num_stages))
        self.downsample_convs = nn.ModuleList(
            ConvNormLayer(d, d, 3, 2, activation=True) for _ in range(num_stages)
        )
        self.pan_blocks = nn.ModuleList(CSPRepLayer(d) for _ in range(num_stages))
        mask_channels = cfg["mask_feature_channels"]
        self.mask_feature_head = MaskFeatFPN(
            [d] * len(strides),
            strides,
            feature_channels=mask_channels[0],
            out_channels=mask_channels[1],
        )
        self.encoder_mask_lateral = ConvLayer(cfg["x4_feat_dim"], mask_channels[1], 3)
        self.encoder_mask_output = EncoderMaskOutput(
            mask_channels[1], cfg["num_prototypes"]
        )

    def forward(
        self, feats: List[Tensor], x4_feat: Tensor
    ) -> Tuple[List[Tensor], Tensor]:
        feats = list(feats)
        feats[-1] = self.encoder[0](feats[-1])

        num_stages = len(self.lateral_convs)
        fpn_maps = [feats[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(
            zip(self.lateral_convs, self.fpn_blocks)
        ):
            backbone_map = feats[num_stages - idx - 1]
            top = lateral_conv(fpn_maps[-1])
            fpn_maps[-1] = top
            top = F.interpolate(top, scale_factor=2.0, mode="nearest")
            fpn_maps.append(fpn_block(torch.concat([top, backbone_map], dim=1)))
        fpn_maps.reverse()

        pan_maps = [fpn_maps[0]]
        for idx, (downsample_conv, pan_block) in enumerate(
            zip(self.downsample_convs, self.pan_blocks)
        ):
            down = downsample_conv(pan_maps[-1])
            pan_maps.append(pan_block(torch.concat([down, fpn_maps[idx + 1]], dim=1)))

        mask_feat = self.mask_feature_head(pan_maps)
        mask_feat = F.interpolate(
            mask_feat, scale_factor=2, mode="bilinear", align_corners=False
        )
        mask_feat = mask_feat + self.encoder_mask_lateral(x4_feat)
        mask_feat = self.encoder_mask_output(mask_feat)
        return pan_maps, mask_feat


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def multi_scale_deformable_attention(
    value: Tensor,
    spatial_shapes: Sequence[Tuple[int, int]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Pure-PyTorch deformable attention (``grid_sample``), any device."""
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampled = []
    for level, (height, width) in enumerate(spatial_shapes):
        value_l = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(
            F.grid_sample(
                value_l,
                grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampled, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class MultiscaleDeformableAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_levels: int, num_points: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = num_heads
        self.n_levels = num_levels
        self.n_points = num_points
        self.sampling_offsets = nn.Linear(
            d_model, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(d_model, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tensor,
        encoder_hidden_states: Tensor,
        reference_points: Tensor,
        spatial_shapes: Sequence[Tuple[int, int]],
    ) -> Tensor:
        hidden_states = hidden_states + position_embeddings
        batch_size, num_queries, _ = hidden_states.shape
        _, sequence_length, _ = encoder_hidden_states.shape
        value = self.value_proj(encoder_hidden_states).view(
            batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads
        )
        offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        weights = F.softmax(weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # Box reference points: (B, Q, 1, 4) cxcywh in [0, 1].
        locations = (
            reference_points[:, :, None, :, None, :2]
            + offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        )
        output = multi_scale_deformable_attention(
            value, spatial_shapes, locations, weights
        )
        return self.output_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d = cfg["d_model"]
        eps = cfg["layer_norm_eps"]
        self.self_attn = SelfAttention(d, cfg["decoder_attention_heads"])
        self.self_attn_layer_norm = nn.LayerNorm(d, eps=eps)
        self.encoder_attn = MultiscaleDeformableAttention(
            d,
            cfg["decoder_attention_heads"],
            len(cfg["feat_strides"]),
            cfg["decoder_n_points"],
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(d, eps=eps)
        self.fc1 = nn.Linear(d, cfg["decoder_ffn_dim"])
        self.fc2 = nn.Linear(cfg["decoder_ffn_dim"], d)
        self.final_layer_norm = nn.LayerNorm(d, eps=eps)

    def forward(
        self,
        hidden_states: Tensor,
        query_pos: Tensor,
        encoder_hidden_states: Tensor,
        reference_points: Tensor,
        spatial_shapes: Sequence[Tuple[int, int]],
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, query_pos)
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)
        residual = hidden_states
        hidden_states = self.encoder_attn(
            hidden_states,
            query_pos,
            encoder_hidden_states,
            reference_points,
            spatial_shapes,
        )
        hidden_states = self.encoder_attn_layer_norm(residual + hidden_states)
        residual = hidden_states
        hidden_states = self.fc2(F.relu(self.fc1(hidden_states)))
        return self.final_layer_norm(residual + hidden_states)


class Decoder(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(cfg) for _ in range(cfg["decoder_layers"])
        )
        self.query_pos_head = MLPPredictionHead(
            4, 2 * cfg["d_model"], cfg["d_model"], 2
        )


class GlobalPointer(nn.Module):
    """Pairwise reading-order scorer: ``logits[i, j]`` for query i before j."""

    def __init__(self, d_model: int, head_size: int) -> None:
        super().__init__()
        self.head_size = head_size
        self.dense = nn.Linear(d_model, head_size * 2)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, length, _ = x.shape
        proj = self.dense(x).reshape(batch_size, length, 2, self.head_size)
        queries, keys = torch.unbind(proj, dim=2)
        logits = (queries @ keys.transpose(-2, -1)) / (self.head_size**0.5)
        mask = torch.tril(torch.ones(length, length, device=logits.device)).bool()
        return logits.masked_fill(mask.unsqueeze(0), -1e4)


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def mask_to_box_coordinate(mask: Tensor, dtype: torch.dtype) -> Tensor:
    """Tight normalized cxcywh box around each boolean mask (zeros if empty)."""
    mask = mask.bool()
    height, width = mask.shape[-2:]
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=mask.device),
        torch.arange(width, device=mask.device),
        indexing="ij",
    )
    x_coords = x_coords.to(dtype)
    y_coords = y_coords.to(dtype)
    finfo_max = torch.full((), torch.finfo(dtype).max, dtype=dtype, device=mask.device)
    x_masked = x_coords * mask
    x_max = x_masked.flatten(start_dim=-2).max(dim=-1).values + 1
    x_min = (
        torch.where(mask, x_masked, finfo_max).flatten(start_dim=-2).min(dim=-1).values
    )
    y_masked = y_coords * mask
    y_max = y_masked.flatten(start_dim=-2).max(dim=-1).values + 1
    y_min = (
        torch.where(mask, y_masked, finfo_max).flatten(start_dim=-2).min(dim=-1).values
    )
    box = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    box = box * torch.any(mask, dim=(-2, -1)).unsqueeze(-1)
    norm = torch.tensor([width, height, width, height], device=mask.device, dtype=dtype)
    x0, y0, x1, y1 = (box / norm).unbind(dim=-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def generate_anchors(
    spatial_shapes: Sequence[Tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype,
    grid_size: float = 0.05,
) -> Tuple[Tensor, Tensor]:
    anchors = []
    for level, (height, width) in enumerate(spatial_shapes):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(end=height, device=device).to(dtype),
            torch.arange(end=width, device=device).to(dtype),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], -1).unsqueeze(0) + 0.5
        grid_xy[..., 0] /= width
        grid_xy[..., 1] /= height
        wh = torch.ones_like(grid_xy) * grid_size * (2.0**level)
        anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, height * width, 4))
    eps = 1e-2
    anchors = torch.concat(anchors, 1)
    valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
    anchors = torch.log(anchors / (1 - anchors))
    anchors = torch.where(
        valid_mask,
        anchors,
        torch.full((), torch.finfo(dtype).max, dtype=dtype, device=device),
    )
    return anchors, valid_mask


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class PPDocLayoutV3Model(nn.Module):
    """PP-DocLayoutV3 inference graph.

    ``forward`` returns ``pred_logits`` ``(B, Q, nc)``, ``pred_boxes``
    ``(B, Q, 4)`` normalized cxcywh, ``order_logits`` ``(B, Q, Q)`` and, with
    ``return_masks=True``, ``pred_masks`` ``(B, Q, H/4, W/4)`` mask logits.
    """

    def __init__(self, nc: int = 25, **overrides) -> None:
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **overrides}
        self.cfg = cfg
        self.nc = nc
        d = cfg["d_model"]
        eps = cfg["layer_norm_eps"]
        self.num_queries = cfg["num_queries"]

        self.backbone = ConvEncoder()
        channels = self.backbone.model.channels
        self.encoder_input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, d, kernel_size=1, bias=False), nn.BatchNorm2d(d))
            for c in channels[1:]
        )
        self.encoder = HybridEncoder(cfg)
        # Training-only denoising embedding, kept so the checkpoint loads strictly.
        self.denoising_class_embed = nn.Embedding(nc, d)
        self.enc_output = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d, eps=eps))
        self.enc_score_head = nn.Linear(d, nc)
        self.enc_bbox_head = MLPPredictionHead(d, d, 4, num_layers=3)
        self.decoder_input_proj = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(d, d, kernel_size=1, bias=False),
                nn.BatchNorm2d(d, cfg["batch_norm_eps"]),
            )
            for _ in cfg["feat_strides"]
        )
        self.decoder = Decoder(cfg)
        self.decoder_order_head = nn.ModuleList(
            nn.Linear(d, d) for _ in range(cfg["decoder_layers"])
        )
        self.decoder_global_pointer = GlobalPointer(d, cfg["global_pointer_head_size"])
        self.decoder_norm = nn.LayerNorm(d, eps=eps)
        self.mask_query_head = MLPPredictionHead(
            d, d, cfg["num_prototypes"], num_layers=3
        )

    # The decoder reuses the encoder heads (tied in the released checkpoint).
    @property
    def class_embed(self) -> nn.Linear:
        return self.enc_score_head

    @property
    def bbox_embed(self) -> MLPPredictionHead:
        return self.enc_bbox_head

    def forward(self, x: Tensor, return_masks: bool = False) -> Dict[str, Tensor]:
        features = self.backbone(x)
        x4_feat = features[0]
        proj_feats = [proj(f) for proj, f in zip(self.encoder_input_proj, features[1:])]
        pan_maps, mask_feat = self.encoder(proj_feats, x4_feat)

        sources = [proj(f) for proj, f in zip(self.decoder_input_proj, pan_maps)]
        spatial_shapes = [(int(s.shape[2]), int(s.shape[3])) for s in sources]
        source_flatten = torch.cat([s.flatten(2).transpose(1, 2) for s in sources], 1)
        dtype = source_flatten.dtype

        # The anchors only matter for their validity mask here: the encoder
        # box proposals they offset are replaced by the mask-derived boxes.
        _, valid_mask = generate_anchors(spatial_shapes, source_flatten.device, dtype)
        memory = valid_mask.to(dtype) * source_flatten
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)

        _, topk_ind = torch.topk(
            enc_outputs_class.max(-1).values, self.num_queries, dim=1
        )

        batch_ind = torch.arange(memory.shape[0], device=memory.device).unsqueeze(1)
        out_query = self.decoder_norm(output_memory[batch_ind, topk_ind])
        mask_query_embed = self.mask_query_head(out_query)
        batch_size, mask_dim, _ = mask_query_embed.shape
        _, _, mask_h, mask_w = mask_feat.shape

        target = output_memory.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        ).detach()

        # Mask-enhanced initialisation: the reference box of each query is the
        # tight box around its encoder mask.
        enc_out_masks = torch.bmm(
            mask_query_embed, mask_feat.flatten(start_dim=2)
        ).reshape(batch_size, mask_dim, mask_h, mask_w)
        reference_points = mask_to_box_coordinate(enc_out_masks > 0, dtype=dtype)
        reference_points = F.sigmoid(inverse_sigmoid(reference_points).detach())

        hidden_states = target
        for decoder_layer in self.decoder.layers:
            query_pos = self.decoder.query_pos_head(reference_points)
            hidden_states = decoder_layer(
                hidden_states,
                query_pos,
                source_flatten,
                reference_points.unsqueeze(2),
                spatial_shapes,
            )
            predicted = self.bbox_embed(hidden_states)
            new_reference_points = F.sigmoid(
                predicted + inverse_sigmoid(reference_points)
            )
            reference_points = new_reference_points.detach()

        last_query = self.decoder_norm(hidden_states)
        order_input = self.decoder_order_head[len(self.decoder.layers) - 1](
            last_query[:, -self.num_queries :]
        )
        outputs = {
            "pred_logits": self.class_embed(last_query),
            "pred_boxes": new_reference_points,
            "order_logits": self.decoder_global_pointer(order_input),
        }
        if return_masks:
            mask_embed = self.mask_query_head(last_query)
            outputs["pred_masks"] = torch.bmm(
                mask_embed, mask_feat.flatten(start_dim=2)
            ).reshape(batch_size, mask_embed.shape[1], mask_h, mask_w)
        return outputs


class PPDocLayoutV3ExportWrapper(nn.Module):
    """Flatten outputs to ``(pred_logits, pred_boxes, order_votes)``.

    The ``(Q, Q)`` pairwise order logits are reduced in-graph to one vote per
    query, which is all the postprocess needs to rank regions.
    """

    def __init__(self, model: PPDocLayoutV3Model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        from ...postprocess.ppdoclayoutv3 import reading_order_votes

        out = self.model(x)
        return (
            out["pred_logits"],
            out["pred_boxes"],
            reading_order_votes(out["order_logits"]),
        )
