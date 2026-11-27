"""Recurrent detection, instance segmentation, and pose decoders for LoopDVR.

Ported from Intellindust-AI-Lab/LoopDVR (commit f2aba1f122029e4db757b69cd37fc5a4909db34e):
    loopdvr/models/decoder_common.py
    loopdvr/models/detection.py
    loopdvr/models/segmentation.py
    loopdvr/models/pose.py
Copyright (c) 2026 The LoopDVR Authors (modifications).
SPDX-License-Identifier: Apache-2.0

Modified from EdgeCrafter, D-FINE (Apache-2.0), RT-DETR (Apache-2.0),
RF-DETR (Apache-2.0), DETRPose (Apache-2.0), Conditional DETR (Apache-2.0),
DETR (Apache-2.0), and Deformable DETR (Apache-2.0).
"""

from __future__ import annotations

import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F

from .loop_adapter import LoopAdapter

# ---------------------------------------------------------------------------
# Shared decoder helpers
# ---------------------------------------------------------------------------


def activation_fn(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported decoder activation: {name}")


def group_route(group_sizes) -> tuple[int, ...]:
    """Map each decoder refinement layer to the index of its sharing group."""
    if not group_sizes or any(type(n) is not int or n < 1 for n in group_sizes):
        raise ValueError("Decoder group sizes must be positive integers")
    return tuple(
        group_idx for group_idx, size in enumerate(group_sizes) for _ in range(size)
    )


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act: str = "relu",
    ):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in itertools.pairwise(dims)
        )
        self.act = activation_fn(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = self.act(x)
        return x


class Gate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(2 * dim, 2 * dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        weights = self.gate(torch.cat([first, second], dim=-1)).sigmoid()
        w1, w2 = weights.chunk(2, dim=-1)
        return self.norm(w1 * first + w2 * second)


def inverse_sigmoid(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.clamp(0, 1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def regression_bins(
    reg_max: int, up: torch.Tensor, reg_scale: torch.Tensor
) -> torch.Tensor:
    """Non-uniform bin centers from the reference FP32 validation path."""
    half = up[0].abs() * reg_scale.abs()
    step = (half + 1) ** (2 / (reg_max - 2))
    left = [-(step**i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right = [step**i - 1 for i in range(1, reg_max // 2)]
    return torch.cat([-2 * half, *left, torch.zeros_like(half), *right, 2 * half])


def expectation(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits.reshape(-1, bins.numel()), dim=1)
    return F.linear(probs, bins).reshape(*logits.shape[:-1], -1)


def flatten_features(
    features: list[torch.Tensor],
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    shapes = [(int(feature.shape[-2]), int(feature.shape[-1])) for feature in features]
    tokens = [feature.flatten(2).transpose(1, 2) for feature in features]
    return torch.cat(tokens, 1), shapes


# ---------------------------------------------------------------------------
# Segmentation head (RF-DETR / EdgeCrafter style)
# ---------------------------------------------------------------------------


class DepthwiseConvBlock(nn.Module):
    """Mix spatial features at the mask resolution."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.dwconv(x).permute(0, 2, 3, 1)
        update = self.act(self.pwconv1(self.norm(spatial)))
        return update.permute(0, 3, 1, 2) + x


class MLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.layers = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(self.norm_in(x))


class SegmentationHead(nn.Module):
    """Predict one mask per final decoder query from the first feature map."""

    def __init__(self, dim: int, image_size, downsample_ratio: int):
        super().__init__()
        self.mask_size = tuple(side // downsample_ratio for side in image_size)
        self.blocks = nn.ModuleList([DepthwiseConvBlock(dim)])
        self.spatial_features_proj = nn.Conv2d(dim, dim, 1)
        self.query_features_block = MLPBlock(dim)
        self.query_features_proj = nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, spatial: torch.Tensor, queries: torch.Tensor
    ) -> torch.Tensor:
        spatial = F.interpolate(
            spatial, size=self.mask_size, mode="bilinear", align_corners=False
        )
        spatial = self.spatial_features_proj(self.blocks[0](spatial))
        queries = self.query_features_proj(self.query_features_block(queries))
        return torch.einsum("bchw,bnc->bnhw", spatial, queries) + self.bias


# ---------------------------------------------------------------------------
# Detection & instance segmentation decoder
# ---------------------------------------------------------------------------


class BoxAttention(nn.Module):
    """Sample image features around each query's current reference box."""

    def __init__(self, dim: int, heads: int, points):
        super().__init__()
        self.heads = heads
        self.points = tuple(points)
        total = heads * sum(points)
        self.sampling_offsets = nn.Linear(dim, total * 2)
        self.attention_weights = nn.Linear(dim, total)
        scales = [1 / n for n in points for _ in range(n)]
        self.register_buffer("num_points_scale", torch.tensor(scales))

    def forward(
        self,
        query: torch.Tensor,
        reference: torch.Tensor,
        values: tuple[torch.Tensor, ...],
        shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        batch, count = query.shape[:2]
        n = sum(self.points)
        offsets = self.sampling_offsets(query).reshape(batch, count, self.heads, n, 2)
        weights = (
            self.attention_weights(query)
            .reshape(batch, count, self.heads, n)
            .softmax(-1)
        )
        offset = (
            offsets
            * self.num_points_scale.to(query.dtype)[:, None]
            * reference[:, :, None, None, 2:]
            * 0.5
        )
        locations = reference[:, :, None, None, :2] + offset
        grids = (
            (2 * locations - 1)
            .permute(0, 2, 1, 3, 4)
            .flatten(0, 1)
            .split(self.points, dim=-2)
        )
        samples = []
        for value, (height, width), grid in zip(values, shapes, grids):
            feature = value.reshape(batch * self.heads, -1, height, width)
            samples.append(
                F.grid_sample(
                    feature,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
            )
        weights = weights.permute(0, 2, 1, 3).reshape(batch * self.heads, 1, count, n)
        result = (torch.cat(samples, dim=-1) * weights).sum(-1)
        return result.reshape(batch, -1, count).permute(0, 2, 1)


class DetectionLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: int, points, activation: str):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = BoxAttention(dim, heads, points)
        self.gateway = Gate(dim)
        self.linear1 = nn.Linear(dim, ffn)
        self.activation = activation_fn(activation)
        self.linear2 = nn.Linear(ffn, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        target: torch.Tensor,
        pos: torch.Tensor,
        refs: torch.Tensor,
        values: tuple[torch.Tensor, ...],
        shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        query = target + pos
        target = self.norm1(target + self.self_attn(query, query, target)[0])
        target = self.gateway(
            target, self.cross_attn(target + pos, refs, values, shapes)
        )
        update = self.linear2(self.activation(self.linear1(target)))
        return self.norm2((target + update).clamp(-65504, 65504))


class BoxQuality(nn.Module):
    def __init__(self, reg_max: int, activation: str):
        super().__init__()
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * 5, 64, 1, 2, activation)

    def forward(self, scores: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(
            corners.reshape(*corners.shape[:2], 4, self.reg_max + 1), -1
        )
        top = probs.topk(4, dim=-1).values
        stats = torch.cat([top, top.mean(dim=-1, keepdim=True)], dim=-1)
        return scores + self.reg_conf(stats.flatten(-2))


def refine_boxes(
    box: torch.Tensor, delta: torch.Tensor, reg_scale: torch.Tensor
) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    scale = reg_scale.abs()
    x1 = cx - (0.5 * scale + delta[..., 0]) * (w / scale)
    y1 = cy - (0.5 * scale + delta[..., 1]) * (h / scale)
    x2 = cx + (0.5 * scale + delta[..., 2]) * (w / scale)
    y2 = cy + (0.5 * scale + delta[..., 3]) * (h / scale)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


class BoxGroups(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn: int,
        adapter_dim: int,
        heads: int,
        points,
        activation: str,
        group_sizes,
        reg_max: int,
        reg_scale: torch.Tensor,
        up: torch.Tensor,
        mask_head: SegmentationHead | None,
    ):
        super().__init__()
        self.layer_to_group = group_route(group_sizes)
        self.layers = nn.ModuleList(
            DetectionLayer(dim, heads, ffn, points, activation) for _ in group_sizes
        )
        self.loop_adapters = nn.ModuleList(
            LoopAdapter(dim, adapter_dim) for _ in self.layer_to_group
        )
        self.lqe_layers = nn.ModuleList(
            BoxQuality(reg_max, activation) for _ in group_sizes
        )
        self.up, self.reg_scale = up, reg_scale
        self.segmentation_head = mask_head


class DetectionDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dim_feedforward: int,
        adapter_hidden_dim: int,
        num_queries: int,
        eval_spatial_size,
        num_classes: int = 80,
        num_points=(3, 6, 3),
        group_sizes=(2, 2),
        feat_strides=(8, 16, 32),
        nhead: int = 8,
        reg_max: int = 32,
        reg_scale: int = 4,
        activation: str = "silu",
        mask_downsample_ratio: int | None = None,
    ):
        super().__init__()
        if len(num_points) != len(feat_strides) or hidden_dim % nhead != 0:
            raise ValueError(
                "Incompatible decoder feature levels or attention dimensions"
            )
        self.num_queries = num_queries
        self.nhead = nhead
        self.reg_max = reg_max
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(
            torch.tensor([reg_scale], dtype=torch.float32), requires_grad=False
        )
        mask_head = (
            SegmentationHead(hidden_dim, eval_spatial_size, mask_downsample_ratio)
            if mask_downsample_ratio is not None
            else None
        )
        self.decoder = BoxGroups(
            hidden_dim,
            dim_feedforward,
            adapter_hidden_dim,
            nhead,
            num_points,
            activation,
            group_sizes,
            reg_max,
            self.reg_scale,
            self.up,
            mask_head,
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, activation)
        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, activation)
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, activation)
        self.dec_score_head = nn.ModuleList(
            nn.Linear(hidden_dim, num_classes) for _ in group_sizes
        )
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (reg_max + 1), 3, activation)]
        )

        anchors = []
        for level, stride in enumerate(feat_strides):
            h, w = [side // stride for side in eval_spatial_size]
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            xy = (torch.stack([x, y], dim=-1) + 0.5) / torch.tensor(
                [w, h], dtype=torch.float32
            )
            wh = torch.ones_like(xy) * 0.05 * (2.0**level)
            anchors.append(torch.cat([xy, wh], dim=-1).reshape(1, h * w, 4))
        anchors = torch.cat(anchors, 1)
        valid = ((anchors > 0.01) & (anchors < 0.99)).all(-1, keepdim=True)
        self.register_buffer(
            "anchors", torch.where(valid, torch.log(anchors / (1 - anchors)), torch.inf)
        )
        self.register_buffer("valid_mask", valid)

    def forward(
        self, feats: list[torch.Tensor], spatial_feat: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | None]:
        memory, shapes = flatten_features(feats)
        proposals = self.valid_mask.to(memory.dtype) * memory
        scores = self.enc_score_head(proposals).max(-1).values
        index = scores.topk(self.num_queries, dim=-1).indices.unsqueeze(-1)
        output = proposals.gather(1, index.expand(-1, -1, memory.shape[-1]))
        anchors = self.anchors.expand(memory.shape[0], -1, -1).gather(
            1, index.expand(-1, -1, 4)
        )
        refs = (self.enc_bbox_head(output) + anchors).sigmoid()
        pos = self.query_pos_head(refs).clamp(-10, 10)
        values = memory.reshape(*memory.shape[:2], self.nhead, -1).permute(0, 2, 3, 1)
        values = values.split([h * w for h, w in shapes], dim=-1)
        bins = regression_bins(self.reg_max, self.up, self.reg_scale)
        previous, corners = 0, 0
        for layer_idx, group_idx in enumerate(self.decoder.layer_to_group):
            output = self.decoder.layers[group_idx](output, pos, refs, values, shapes)
            output = self.decoder.loop_adapters[layer_idx](output)
            if layer_idx == 0:
                base = (
                    self.pre_bbox_head(output) + inverse_sigmoid(refs, 1e-5)
                ).sigmoid()
            corners = self.dec_bbox_head[0](output + previous) + corners
            refs = refine_boxes(base, expectation(corners, bins), self.reg_scale)
            previous = output
        final_group = self.decoder.layer_to_group[-1]
        scores = self.decoder.lqe_layers[final_group](
            self.dec_score_head[final_group](output), corners
        )
        out: dict[str, torch.Tensor | None] = {
            "pred_logits": scores,
            "pred_boxes": refs,
        }
        if self.decoder.segmentation_head is not None:
            out["pred_masks"] = self.decoder.segmentation_head(spatial_feat, output)
        return out


# ---------------------------------------------------------------------------
# Pose decoder
# ---------------------------------------------------------------------------


class PoseAttention(nn.Module):
    """Deformable attention sampled around the instance and each joint."""

    def __init__(self, dim: int, heads: int, levels: int, points: int):
        super().__init__()
        self.heads = heads
        self.levels = levels
        self.points = points
        total = heads * levels * points
        self.sampling_offsets = nn.Linear(dim, total * 2)
        self.attention_weights = nn.Linear(dim, total)

    def forward(
        self,
        query: torch.Tensor,
        reference: torch.Tensor,
        values: tuple[torch.Tensor, ...],
        shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        batch, count = query.shape[:2]
        offsets = self.sampling_offsets(query).reshape(
            batch, count, self.heads, self.levels, self.points, 2
        )
        weights = self.attention_weights(query).reshape(
            batch, count, self.heads, self.levels * self.points
        )
        weights = (
            weights.softmax(-1)
            .transpose(1, 2)
            .reshape(batch * self.heads, 1, count, -1)
        )
        sizes = (
            torch.tensor(shapes, device=query.device)
            .flip([1])
            .reshape(1, 1, 1, self.levels, 1, 2)
        )
        locations = reference.flatten(1, 2)[:, :, None, None, None, :] + offsets / sizes
        grids = (2 * locations - 1).transpose(1, 2).flatten(0, 1)
        sampled = []
        for index, shape in enumerate(shapes):
            sampled.append(
                F.grid_sample(
                    values[index].unflatten(2, shape),
                    grids[:, :, index],
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
            )
        output = (torch.cat(sampled, dim=-1) * weights).sum(-1)
        return output.view(batch, -1, count).transpose(1, 2)


class PoseLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn: int,
        heads: int,
        levels: int,
        points: int,
        activation: str,
    ):
        super().__init__()
        self.within_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.within_norm = nn.LayerNorm(dim)
        self.across_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.across_norm = nn.LayerNorm(dim)
        self.cross_attn = PoseAttention(dim, heads, levels, points)
        self.gateway = Gate(dim)
        self.linear1 = nn.Linear(dim, ffn)
        self.linear2 = nn.Linear(ffn, dim)
        self.activation = activation_fn(activation)
        self.norm2 = nn.LayerNorm(dim)

    @staticmethod
    def with_position(tgt: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        return torch.cat((tgt[:, :, :1], tgt[:, :, 1:] + position), dim=2)

    def forward(
        self,
        tgt: torch.Tensor,
        pos: torch.Tensor,
        references: torch.Tensor,
        values: tuple[torch.Tensor, ...],
        shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        batch, queries, joints, dim = tgt.shape
        q = self.with_position(tgt, pos).flatten(0, 1)
        update = self.within_attn(q, q, tgt.flatten(0, 1))[0].reshape_as(tgt)
        tgt = self.within_norm(tgt + update)
        tgt = tgt.transpose(1, 2).flatten(0, 1)
        update = self.across_attn(tgt, tgt, tgt)[0]
        tgt = (
            self.across_norm(tgt + update)
            .reshape(batch, joints, queries, dim)
            .transpose(1, 2)
        )
        q = self.with_position(tgt, pos).flatten(1, 2)
        update = self.cross_attn(q, references, values, shapes).reshape_as(tgt)
        tgt = self.gateway(tgt, update)
        return self.norm2(
            (tgt + self.linear2(self.activation(self.linear1(tgt)))).clamp(
                -65504, 65504
            )
        )


class PoseQuality(nn.Module):
    def __init__(self, joints: int):
        super().__init__()
        self.reg_conf = MLP(joints * 5, 256, 1, 2)

    def forward(
        self, logits: torch.Tensor, coordinates: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        samples = F.grid_sample(
            features,
            2 * coordinates - 1,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        top = samples.topk(4, dim=-1).values
        stats = torch.cat([top, top.mean(-1, keepdim=True)], dim=-1)
        return logits + self.reg_conf(stats.flatten(-2))


class PoseGroups(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn: int,
        adapter_dim: int,
        heads: int,
        levels: int,
        points: int,
        activation: str,
        group_sizes,
    ):
        super().__init__()
        self.layer_to_group = group_route(group_sizes)
        self.layers = nn.ModuleList(
            PoseLayer(dim, ffn, heads, levels, points, activation) for _ in group_sizes
        )
        self.loop_adapters = nn.ModuleList(
            LoopAdapter(dim, adapter_dim) for _ in self.layer_to_group
        )
        self.half_pose_ref_point_head = MLP(dim, dim, dim, 2)
        frequencies = torch.arange(dim // 2, dtype=torch.float32)
        self.register_buffer("dim_t", 10000 ** (2 * (frequencies // 2) / (dim // 2)))

    def query_position(self, coordinates: torch.Tensor) -> torch.Tensor:
        def embedding(axis: torch.Tensor) -> torch.Tensor:
            phase = axis * (2 * math.pi) / self.dim_t
            return torch.stack(
                (phase[..., 0::2].sin(), phase[..., 1::2].cos()), dim=4
            ).flatten(3)

        y, x = embedding(coordinates[..., 1:2]), embedding(coordinates[..., :1])
        return self.half_pose_ref_point_head(torch.cat((y, x), dim=3))


class PoseDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dim_feedforward: int,
        adapter_hidden_dim: int,
        num_queries: int,
        eval_spatial_size,
        num_classes: int = 2,
        num_points: int = 4,
        group_sizes=(2, 2),
        feat_strides=(8, 16, 32),
        nhead: int = 8,
        reg_max: int = 32,
        reg_scale: int = 4,
        activation: str = "relu",
        num_body_points: int = 17,
    ):
        super().__init__()
        if hidden_dim % nhead or hidden_dim % 4:
            raise ValueError(
                "Pose dimension must be divisible by the head count and by four"
            )
        self.num_queries = num_queries
        self.num_body_points = num_body_points
        self.nhead = nhead
        self.reg_max = reg_max
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(
            torch.tensor([reg_scale], dtype=torch.float32), requires_grad=False
        )
        self.decoder = PoseGroups(
            hidden_dim,
            dim_feedforward,
            adapter_hidden_dim,
            nhead,
            len(feat_strides),
            num_points,
            activation,
            group_sizes,
        )
        self.keypoint_embedding = nn.Embedding(num_body_points, hidden_dim)
        self.instance_embedding = nn.Embedding(1, hidden_dim)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.enc_output_norm = nn.LayerNorm(hidden_dim)
        self.enc_out_class_embed = nn.Linear(hidden_dim, num_classes)
        self.enc_pose_embed = MLP(hidden_dim, 2 * hidden_dim, 2 * num_body_points, 4)
        self.class_embed = nn.ModuleList(
            nn.Linear(hidden_dim, num_classes) for _ in group_sizes
        )
        self.pose_embed = nn.ModuleList(
            MLP(hidden_dim, hidden_dim, 2 * (reg_max + 1), 3) for _ in group_sizes
        )
        self.lqe_embed = nn.ModuleList(
            PoseQuality(num_body_points) for _ in group_sizes
        )
        self.pre_pose_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        anchors = []
        for stride in feat_strides:
            h, w = [size // stride for size in eval_spatial_size]
            y, x = torch.meshgrid(
                torch.linspace(0, h - 1, h),
                torch.linspace(0, w - 1, w),
                indexing="ij",
            )
            xy = (torch.stack([x, y], dim=-1) + 0.5) / torch.tensor(
                [w, h], dtype=torch.float32
            )
            anchors.append(xy.reshape(1, h * w, 2))
        coords = torch.cat(anchors, 1)
        self.register_buffer(
            "valid_mask", ~((coords > 0.01) & (coords < 0.99)).all(-1, keepdim=True)
        )
        self.register_buffer("anchors", torch.log(coords / (1 - coords)))

    def forward(self, feats: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        memory, shapes = flatten_features(feats)
        proposal = self.enc_output_norm(
            self.enc_output(memory.masked_fill(self.valid_mask, 0.0))
        )
        index = (
            self.enc_out_class_embed(proposal)
            .max(-1)
            .values.topk(self.num_queries, dim=1)
            .indices.unsqueeze(-1)
        )
        top = proposal.gather(1, index.expand(-1, -1, proposal.shape[-1]))
        batch, count = top.shape[:2]
        anchors = self.anchors.expand(batch, -1, -1).gather(
            1, index.expand(-1, -1, 2)
        )
        offsets = self.enc_pose_embed(top).reshape(batch, count, self.num_body_points, 2)
        coords = (offsets + anchors.unsqueeze(-2)).sigmoid()
        references = torch.cat([coords.mean(2, keepdim=True), coords], dim=2)
        target = self.tgt_embed.weight[None, :, None, :].expand(batch, -1, -1, -1)
        joints = self.keypoint_embedding.weight[None, None, :, :] + target
        person = self.instance_embedding.weight[None, None, :, :].expand(
            batch, count, -1, -1
        )
        output = torch.cat([person, joints], dim=2)
        values = (
            memory.unflatten(2, (self.nhead, -1)).permute(0, 2, 3, 1).flatten(0, 1)
        )
        values = values.split([h * w for h, w in shapes], dim=-1)
        bins = regression_bins(self.reg_max, self.up, self.reg_scale)
        previous, corners = 0, 0
        for layer_idx, group_idx in enumerate(self.decoder.layer_to_group):
            coords = references[:, :, 1:]
            pos = self.decoder.query_position(coords)
            output = self.decoder.layers[group_idx](
                output, pos, references, values, shapes
            )
            output = self.decoder.loop_adapters[layer_idx](output)
            joints = output[:, :, 1:]
            if layer_idx == 0:
                base = (
                    self.pre_pose_embed(joints) + inverse_sigmoid(coords, 1e-3)
                ).sigmoid()
            corners = self.pose_embed[group_idx](joints + previous) + corners
            coords = base + expectation(corners, bins) / self.reg_scale.abs()
            references = torch.cat([coords.mean(2, keepdim=True), coords], dim=2)
            previous = joints
        final_group = self.decoder.layer_to_group[-1]
        scores = self.lqe_embed[final_group](
            self.class_embed[final_group](output[:, :, 0]), coords, feats[0]
        )
        return {"pred_logits": scores, "pred_keypoints": coords.flatten(-2)}
