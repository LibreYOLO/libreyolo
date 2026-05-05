"""RF-DETR backbone, projector, position encoding, and DINOv2 glue.

Ported from RF-DETR (https://github.com/roboflow/rf-detr).
Copyright (c) 2025 Roboflow, Inc. All Rights Reserved.
Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR).
Copyright (c) 2024 Baidu. All Rights Reserved.
Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR).
Copyright (c) 2021 Microsoft. All Rights Reserved.
Modified from DETR (https://github.com/facebookresearch/detr).
Copyright (c) Facebook, Inc. and its affiliates.
Modified from ViTDet (https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet).
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import math
import types
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import AutoBackbone

from .dinov2 import (
    WindowedDinov2WithRegistersBackbone,
    WindowedDinov2WithRegistersConfig,
)
from .tensors import NestedTensor

logger = logging.getLogger(__name__)

__all__ = [
    "BackboneBase",
    "Backbone",
    "DinoV2",
    "Joiner",
    "MultiScaleProjector",
    "PositionEmbeddingLearned",
    "PositionEmbeddingSine",
    "build_backbone",
    "build_position_encoding",
]


# ---------------------------------------------------------------------------
# Position encodings (originally DETR / Facebook).
# ---------------------------------------------------------------------------


class PositionEmbeddingSine(nn.Module):
    """Standard sinusoidal positional encoding generalized to images."""

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self._export = False

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

    def forward(self, tensor_list: NestedTensor, align_dim_orders=True):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        if align_dim_orders:
            pos = torch.cat((pos_y, pos_x), dim=3).permute(1, 2, 0, 3)
            # return: (H, W, bs, C)
        else:
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            # return: (bs, C, H, W)
        return pos

    def forward_export(self, mask: torch.Tensor, align_dim_orders=True):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        if align_dim_orders:
            pos = torch.cat((pos_y, pos_x), dim=3).permute(1, 2, 0, 3)
        else:
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()
        self._export = False

    def export(self):
        raise NotImplementedError

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[:2]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .unsqueeze(2)
            .repeat(1, 1, x.shape[2], 1)
        )
        return pos


def build_position_encoding(hidden_dim, position_embedding):
    num_steps = hidden_dim // 2
    if position_embedding in ("v2", "sine"):
        position_embedding = PositionEmbeddingSine(num_steps, normalize=True)
    elif position_embedding in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(num_steps)
    else:
        raise ValueError(f"not supported {position_embedding}")
    return position_embedding


# ---------------------------------------------------------------------------
# Projector primitives (originally LW-DETR / ViTDet).
# ---------------------------------------------------------------------------


class LayerNorm(nn.Module):
    """Channels-first LayerNorm (popularized by ConvNeXt).

    Performs point-wise mean and variance normalization over the channel
    dimension for inputs of shape ``(batch_size, channels, height, width)``.
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # NHWC layer_norm avoids fp16 overflow that the channels-first equivalent triggers.
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def get_norm(norm: Optional[Union[str, Callable[[int], nn.Module]]], out_channels: int) -> Optional[nn.Module]:
    """Resolve a norm spec ("LN" / callable / None) to an ``nn.Module`` (or ``None``)."""
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


def get_activation(name, inplace=False):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name in ["LeakyReLU", "leakyrelu", "lrelu"]:
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name is None:
        module = nn.Identity()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class ConvX(nn.Module):
    """Conv-bn module."""

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel=3,
        stride=1,
        groups=1,
        dilation=1,
        act="relu",
        layer_norm=False,
        rms_norm=False,
    ):
        super(ConvX, self).__init__()
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel)
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        if rms_norm:
            self.bn = nn.RMSNorm(out_planes)
        else:
            self.bn = get_norm("LN", out_planes) if layer_norm else nn.BatchNorm2d(out_planes)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x.contiguous())))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, act="silu", layer_norm=False, rms_norm=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvX(c1, c_, k[0], 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX(c_, c2, k[1], 1, groups=g, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act="silu", layer_norm=False, rms_norm=False):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvX(c1, 2 * self.c, 1, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX(
            (2 + n) * self.c, c2, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm
        )
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MultiScaleProjector(nn.Module):
    """Build pyramid features on top of an input feature map (LW-DETR projector)."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        scale_factors: Sequence[float],
        num_blocks: int = 3,
        layer_norm: bool = False,
        rms_norm: bool = False,
        survival_prob: float = 1.0,
        force_drop_last_n_features: int = 0,
    ) -> None:
        super(MultiScaleProjector, self).__init__()

        self.scale_factors = scale_factors
        self.survival_prob = survival_prob
        self.force_drop_last_n_features = force_drop_last_n_features

        stages_sampling = []
        stages = []
        self.use_extra_pool = False
        for scale in scale_factors:
            stages_sampling.append([])
            for in_dim in in_channels:
                layers = []

                if scale == 4.0:
                    layers.extend(
                        [
                            nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                            get_norm("LN", in_dim // 2),
                            nn.GELU(),
                            nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                        ]
                    )
                elif scale == 2.0:
                    layers.extend(
                        [
                            nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                        ]
                    )
                elif scale == 1.0:
                    pass
                elif scale == 0.5:
                    layers.extend(
                        [
                            ConvX(in_dim, in_dim, 3, 2, layer_norm=layer_norm),
                        ]
                    )
                elif scale == 0.25:
                    self.use_extra_pool = True
                    continue
                else:
                    raise NotImplementedError("Unsupported scale_factor:{}".format(scale))
                layers = nn.Sequential(*layers)
                stages_sampling[-1].append(layers)
            stages_sampling[-1] = nn.ModuleList(stages_sampling[-1])

            in_dim = int(sum(in_channel // max(1, scale) for in_channel in in_channels))
            layers = [
                C2f(in_dim, out_channels, num_blocks, layer_norm=layer_norm),
                get_norm("LN", out_channels),
            ]
            layers = nn.Sequential(*layers)
            stages.append(layers)

        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        num_features = len(x)
        if self.survival_prob < 1.0 and self.training:
            final_drop_prob = 1 - self.survival_prob
            drop_p = np.random.uniform()
            for i in range(1, num_features):
                critical_drop_prob = i * (final_drop_prob / (num_features - 1))
                if drop_p < critical_drop_prob:
                    x[i][:] = 0
        elif self.force_drop_last_n_features > 0:
            for i in range(self.force_drop_last_n_features):
                x[-(i + 1)] = torch.zeros_like(x[-(i + 1)])

        results = []
        for i, stage in enumerate(self.stages):
            feat_fuse = []
            for j, stage_sampling in enumerate(self.stages_sampling[i]):
                feat_fuse.append(stage_sampling(x[j]))
            if len(feat_fuse) > 1:
                feat_fuse = torch.cat(feat_fuse, dim=1)
            else:
                feat_fuse = feat_fuse[0]
            results.append(stage(feat_fuse))
        if self.use_extra_pool:
            results.append(F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0))
        return results


class SimpleProjector(nn.Module):
    def __init__(self, in_dim, out_dim, factor_kernel=False):
        super(SimpleProjector, self).__init__()
        if not factor_kernel:
            self.convx1 = ConvX(in_dim, in_dim * 2, layer_norm=True, act="silu")
            self.convx2 = ConvX(in_dim * 2, out_dim, layer_norm=True, act="silu")
        else:
            self.convx1 = ConvX(in_dim, out_dim, kernel=(3, 1), layer_norm=True, act="silu")
            self.convx2 = ConvX(out_dim, out_dim, kernel=(1, 3), layer_norm=True, act="silu")
        self.ln = get_norm("LN", out_dim)

    def forward(self, x):
        out = self.ln(self.convx2(self.convx1(x[0])))
        return [out]


# ---------------------------------------------------------------------------
# DINOv2 backbone wrapper (HuggingFace AutoBackbone with optional windowed attn).
# ---------------------------------------------------------------------------


size_to_width = {
    "tiny": 192,
    "small": 384,
    "base": 768,
    "large": 1024,
}


def get_config(size, use_registers):
    widths = {"small": 384, "base": 768, "large": 1024}
    heads = {"small": 6, "base": 12, "large": 16}
    layers = {"small": 12, "base": 12, "large": 24}
    return {
        "apply_layernorm": True,
        "attention_probs_dropout_prob": 0.0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": widths[size],
        "image_size": 518,
        "initializer_range": 0.02,
        "interpolate_antialias": True,
        "interpolate_offset": 0.0,
        "layer_norm_eps": 1e-6,
        "layerscale_value": 1.0,
        "mlp_ratio": 4,
        "num_attention_heads": heads[size],
        "num_channels": 3,
        "num_hidden_layers": layers[size],
        "num_register_tokens": 4 if use_registers else 0,
        "patch_size": 14,
        "qkv_bias": True,
        "reshape_hidden_states": True,
        "use_swiglu_ffn": False,
    }


class DinoV2(nn.Module):
    def __init__(
        self,
        shape=(640, 640),
        out_feature_indexes=[2, 4, 5, 9],
        size="base",
        use_registers=True,
        use_windowed_attn=True,
        gradient_checkpointing=False,
        load_dinov2_weights=True,
        patch_size=14,
        num_windows=4,
        positional_encoding_size=37,
        drop_path_rate=0.0,
    ):
        super().__init__()

        name = f"facebook/dinov2-with-registers-{size}" if use_registers else f"facebook/dinov2-{size}"

        self.shape = shape
        self.patch_size = patch_size
        self.num_windows = num_windows

        if not use_windowed_attn:
            assert not gradient_checkpointing, "Gradient checkpointing is not supported for non-windowed attention"
            assert load_dinov2_weights, "Using non-windowed attention requires loading dinov2 weights from hub"
            if drop_path_rate > 0.0:
                logger.warning(
                    "drop_path_rate > 0.0 is not supported for non-windowed DinoV2 backbones."
                    " drop_path will be ignored."
                )
            self.encoder = AutoBackbone.from_pretrained(
                name,
                out_features=[f"stage{i}" for i in out_feature_indexes],
                return_dict=False,
            )
        else:
            window_block_indexes = set(range(out_feature_indexes[-1] + 1))
            window_block_indexes.difference_update(out_feature_indexes)
            window_block_indexes = list(window_block_indexes)

            dino_config = get_config(size, use_registers)

            dino_config["return_dict"] = False
            dino_config["out_features"] = [f"stage{i}" for i in out_feature_indexes]
            dino_config["drop_path_rate"] = drop_path_rate

            implied_resolution = positional_encoding_size * patch_size

            if implied_resolution != dino_config["image_size"]:
                if load_dinov2_weights:
                    logger.warning(
                        "Using a different number of positional encodings than DINOv2, which means"
                        " we're not loading DINOv2 backbone weights. This is not a problem if"
                        " finetuning a pretrained RF-DETR model."
                    )
                dino_config["image_size"] = implied_resolution
                load_dinov2_weights = False

            if patch_size != 14:
                if load_dinov2_weights:
                    logger.warning(
                        f"Using patch size {patch_size} instead of 14, which means we're not loading"
                        " DINOv2 backbone weights. This is not a problem if finetuning a pretrained"
                        " RF-DETR model."
                    )
                dino_config["patch_size"] = patch_size
                load_dinov2_weights = False

            windowed_dino_config = WindowedDinov2WithRegistersConfig(
                **dino_config,
                num_windows=num_windows,
                window_block_indexes=window_block_indexes,
                gradient_checkpointing=gradient_checkpointing,
            )
            self.encoder = (
                WindowedDinov2WithRegistersBackbone.from_pretrained(
                    name,
                    config=windowed_dino_config,
                )
                if load_dinov2_weights
                else WindowedDinov2WithRegistersBackbone(windowed_dino_config)
            )

        self._out_feature_channels = [size_to_width[size]] * len(out_feature_indexes)
        self._export = False

    def export(self):
        if self._export:
            return
        self._export = True
        shape = self.shape

        def make_new_interpolated_pos_encoding(position_embeddings, patch_size, height, width):
            num_positions = position_embeddings.shape[1] - 1
            dim = position_embeddings.shape[-1]
            height = height // patch_size
            width = width // patch_size

            class_pos_embed = position_embeddings[:, 0]
            patch_pos_embed = position_embeddings[:, 1:]

            patch_pos_embed = patch_pos_embed.reshape(
                1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
            )
            patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

            patch_pos_embed = F.interpolate(
                patch_pos_embed,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
                antialias=patch_pos_embed.device.type != "mps",
            )

            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        with torch.no_grad():
            new_positions = make_new_interpolated_pos_encoding(
                self.encoder.embeddings.position_embeddings,
                self.encoder.config.patch_size,
                shape[0],
                shape[1],
            )
        old_interpolate_pos_encoding = self.encoder.embeddings.interpolate_pos_encoding

        def new_interpolate_pos_encoding(self_mod, embeddings, height, width):
            num_patches = embeddings.shape[1] - 1
            num_positions = self_mod.position_embeddings.shape[1] - 1
            if num_patches == num_positions and height == width:
                return self_mod.position_embeddings
            return old_interpolate_pos_encoding(embeddings, height, width)

        self.encoder.embeddings.position_embeddings = nn.Parameter(new_positions)
        self.encoder.embeddings.interpolate_pos_encoding = types.MethodType(
            new_interpolate_pos_encoding, self.encoder.embeddings
        )

    def forward(self, x):
        block_size = self.patch_size * self.num_windows
        assert x.shape[2] % block_size == 0 and x.shape[3] % block_size == 0, (
            f"Backbone requires input shape to be divisible by {block_size}, but got {x.shape}"
        )
        x = self.encoder(x)
        return list(x[0])


# ---------------------------------------------------------------------------
# Backbone abstract base + RF-DETR Backbone + Joiner + builder.
# ---------------------------------------------------------------------------


class BackboneBase(nn.Module):
    def __init__(self):
        super().__init__()

    def get_named_param_lr_pairs(self, args, prefix: str):
        raise NotImplementedError


class Backbone(BackboneBase):
    """RF-DETR backbone: DINOv2 encoder + MultiScaleProjector."""

    def __init__(
        self,
        name: str,
        pretrained_encoder: str = None,
        window_block_indexes: list = None,
        drop_path=0.0,
        out_channels=256,
        out_feature_indexes: list = None,
        projector_scale: list = None,
        use_cls_token: bool = False,
        freeze_encoder: bool = False,
        layer_norm: bool = False,
        target_shape: tuple[int, int] = (640, 640),
        rms_norm: bool = False,
        backbone_lora: bool = False,
        gradient_checkpointing: bool = False,
        load_dinov2_weights: bool = True,
        patch_size: int = 14,
        num_windows: int = 4,
        positional_encoding_size: int = 0,
    ):
        super().__init__()
        # Encoder names look like "dinov2_base" or "dinov2_registers_windowed_base":
        # the first token is always "dinov2", optional "registers" and "windowed"
        # toggle the variant, and the last token is the size.
        name_parts = name.split("_")
        assert name_parts[0] == "dinov2"
        use_registers = False
        if "registers" in name_parts:
            use_registers = True
            name_parts.remove("registers")
        use_windowed_attn = False
        if "windowed" in name_parts:
            use_windowed_attn = True
            name_parts.remove("windowed")
        assert len(name_parts) == 2, (
            "name should be dinov2, then either registers, windowed, both, or none, then the size"
        )
        self.encoder = DinoV2(
            size=name_parts[-1],
            out_feature_indexes=out_feature_indexes,
            shape=target_shape,
            use_registers=use_registers,
            use_windowed_attn=use_windowed_attn,
            gradient_checkpointing=gradient_checkpointing,
            load_dinov2_weights=load_dinov2_weights,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
            drop_path_rate=drop_path,
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        assert sorted(self.projector_scale) == self.projector_scale, (
            "only support projector scale P3/P4/P5/P6 in ascending order."
        )
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )

        self._export = False

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

        if not hasattr(self.encoder, "merge_and_unload"):
            return

        try:
            from peft import PeftModel
        except ModuleNotFoundError:
            logger.warning("peft is not installed; skipping LoRA weight merging during export.")
            return
        except ImportError as exc:
            logger.warning("Failed to import PeftModel from peft during export: %s", exc)
            raise

        if isinstance(self.encoder, PeftModel):
            logger.info("Merging and unloading LoRA weights")
            self.encoder = self.encoder.merge_and_unload()

    def forward(self, tensor_list: NestedTensor):
        feats = self.encoder(tensor_list.tensors)
        feats = self.projector(feats)
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out

    def forward_export(self, tensors: torch.Tensor):
        feats = self.encoder(tensors)
        feats = self.projector(feats)
        out_feats = []
        out_masks = []
        for feat in feats:
            b, _, h, w = feat.shape
            out_masks.append(torch.zeros((b, h, w), dtype=torch.bool, device=feat.device))
            out_feats.append(feat)
        return out_feats, out_masks

    def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
        num_layers = args.out_feature_indexes[-1] + 1
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs = {}
        for n, p in self.named_parameters():
            n = prefix + "." + n
            if backbone_key in n and p.requires_grad:
                lr = (
                    args.lr_encoder
                    * get_dinov2_lr_decay_rate(
                        n,
                        lr_decay_rate=args.lr_vit_layer_decay,
                        num_layers=num_layers,
                    )
                    * args.lr_component_decay**2
                )
                wd = args.weight_decay * get_dinov2_weight_decay_rate(n)
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
        return named_param_lr_pairs


def get_dinov2_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
    """Calculate lr decay rate for different ViT blocks."""
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if "embeddings" in name:
            layer_id = 0
        elif ".layer." in name and ".residual." not in name:
            layer_id = int(name[name.find(".layer.") :].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_dinov2_weight_decay_rate(name, weight_decay_rate=1.0):
    if (
        ("gamma" in name)
        or ("pos_embed" in name)
        or ("rel_pos" in name)
        or ("bias" in name)
        or ("norm" in name)
        or ("embeddings" in name)
    ):
        weight_decay_rate = 0.0
    return weight_decay_rate


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self._export = False

    def forward(self, tensor_list: NestedTensor):
        x = self[0](tensor_list)
        pos = []
        for x_ in x:
            pos.append(self[1](x_, align_dim_orders=False).to(x_.tensors.dtype))
        return x, pos

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for _, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()

    def forward_export(self, inputs: torch.Tensor):
        feats, masks = self[0](inputs)
        poss = []
        for feat, mask in zip(feats, masks):
            poss.append(self[1](mask, align_dim_orders=False).to(feat.dtype))
        return feats, None, poss


def build_backbone(
    encoder,
    vit_encoder_num_layers,
    pretrained_encoder,
    window_block_indexes,
    drop_path,
    out_channels,
    out_feature_indexes,
    projector_scale,
    use_cls_token,
    hidden_dim,
    position_embedding,
    freeze_encoder,
    layer_norm,
    target_shape,
    rms_norm,
    backbone_lora,
    force_no_pretrain,
    gradient_checkpointing,
    load_dinov2_weights,
    patch_size,
    num_windows,
    positional_encoding_size,
):
    del vit_encoder_num_layers, force_no_pretrain
    position_embedding = build_position_encoding(hidden_dim, position_embedding)
    backbone = Backbone(
        encoder,
        pretrained_encoder,
        window_block_indexes=window_block_indexes,
        drop_path=drop_path,
        out_channels=out_channels,
        out_feature_indexes=out_feature_indexes,
        projector_scale=projector_scale,
        use_cls_token=use_cls_token,
        layer_norm=layer_norm,
        freeze_encoder=freeze_encoder,
        target_shape=target_shape,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        gradient_checkpointing=gradient_checkpointing,
        load_dinov2_weights=load_dinov2_weights,
        patch_size=patch_size,
        num_windows=num_windows,
        positional_encoding_size=positional_encoding_size,
    )
    return Joiner(backbone, position_embedding)
