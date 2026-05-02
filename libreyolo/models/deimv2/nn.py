"""Top-level DEIMv2 model wiring."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from libreyolo.models.deim.backbone import HGNetv2

from .engine.backbone.dinov3_adapter import DINOv3STAs
from .engine.deim.deim_decoder import DEIMTransformer
from .engine.deim.hybrid_encoder import HybridEncoder
from .engine.deim.lite_encoder import LiteEncoder


SIZE_ALIASES = {"a": "atto", "f": "femto", "p": "pico"}
DINO_SIZES = {"s", "m", "l", "x"}


SIZE_CONFIGS: dict[str, dict[str, Any]] = {
    "atto": {
        "input_size": 320,
        "backbone": {
            "type": "hgnetv2",
            "name": "Atto",
            "return_idx": [2],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
            "pretrained": False,
        },
        "encoder": {
            "type": "lite",
            "in_channels": [256],
            "feat_strides": [16],
            "hidden_dim": 64,
            "expansion": 0.34,
            "depth_mult": 0.5,
            "act": "silu",
        },
        "decoder": {
            "eval_spatial_size": [320, 320],
            "feat_channels": [64, 64],
            "feat_strides": [16, 32],
            "hidden_dim": 64,
            "num_levels": 2,
            "num_points": [4, 2],
            "num_layers": 3,
            "eval_idx": -1,
            "num_queries": 100,
            "dim_feedforward": 160,
            "activation": "silu",
            "mlp_act": "silu",
            "share_bbox_head": True,
            "use_gateway": False,
        },
    },
    "femto": {
        "input_size": 416,
        "backbone": {
            "type": "hgnetv2",
            "name": "Femto",
            "return_idx": [2],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
            "pretrained": False,
        },
        "encoder": {
            "type": "lite",
            "in_channels": [512],
            "feat_strides": [16],
            "hidden_dim": 96,
            "expansion": 0.34,
            "depth_mult": 0.5,
            "act": "silu",
        },
        "decoder": {
            "eval_spatial_size": [416, 416],
            "feat_channels": [96, 96],
            "feat_strides": [16, 32],
            "hidden_dim": 96,
            "num_levels": 2,
            "num_points": [4, 2],
            "num_layers": 3,
            "eval_idx": -1,
            "num_queries": 150,
            "dim_feedforward": 256,
            "activation": "silu",
            "mlp_act": "silu",
            "share_bbox_head": True,
            "use_gateway": False,
        },
    },
    "pico": {
        "input_size": 640,
        "backbone": {
            "type": "hgnetv2",
            "name": "Pico",
            "return_idx": [2],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
            "pretrained": False,
        },
        "encoder": {
            "type": "lite",
            "in_channels": [512],
            "feat_strides": [16],
            "hidden_dim": 112,
            "expansion": 0.34,
            "depth_mult": 0.5,
            "act": "silu",
        },
        "decoder": {
            "eval_spatial_size": [640, 640],
            "feat_channels": [112, 112],
            "feat_strides": [16, 32],
            "hidden_dim": 112,
            "num_levels": 2,
            "num_points": [4, 2],
            "num_layers": 3,
            "eval_idx": -1,
            "num_queries": 200,
            "dim_feedforward": 320,
            "activation": "silu",
            "mlp_act": "silu",
            "share_bbox_head": True,
            "use_gateway": False,
        },
    },
    "n": {
        "input_size": 640,
        "backbone": {
            "type": "hgnetv2",
            "name": "B0",
            "return_idx": [2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
            "pretrained": False,
        },
        "encoder": {
            "type": "hybrid",
            "in_channels": [512, 1024],
            "feat_strides": [16, 32],
            "hidden_dim": 128,
            "dim_feedforward": 512,
            "expansion": 0.34,
            "depth_mult": 0.5,
            "use_encoder_idx": [1],
            "version": "dfine",
            "act": "silu",
            "csp_type": "csp2",
            "fuse_op": "sum",
        },
        "decoder": {
            "eval_spatial_size": [640, 640],
            "feat_channels": [128, 128],
            "feat_strides": [16, 32],
            "hidden_dim": 128,
            "num_levels": 2,
            "num_points": [6, 6],
            "num_layers": 3,
            "eval_idx": -1,
            "num_queries": 300,
            "dim_feedforward": 512,
            "activation": "silu",
            "mlp_act": "silu",
            "reg_max": 32,
            "reg_scale": 4,
        },
    },
    "s": {
        "input_size": 640,
        "backbone": {
            "type": "dinov3",
            "name": "vit_tiny",
            "embed_dim": 192,
            "interaction_indexes": [3, 7, 11],
            "num_heads": 3,
        },
        "encoder": {
            "type": "hybrid",
            "in_channels": [192, 192, 192],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 192,
            "dim_feedforward": 512,
            "expansion": 0.34,
            "depth_mult": 0.67,
            "use_encoder_idx": [2],
            "version": "deim",
            "act": "silu",
            "csp_type": "csp2",
            "fuse_op": "sum",
        },
        "decoder": {
            "eval_spatial_size": [640, 640],
            "feat_channels": [192, 192, 192],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 192,
            "num_levels": 3,
            "num_points": [3, 6, 3],
            "num_layers": 4,
            "eval_idx": -1,
            "num_queries": 300,
            "dim_feedforward": 512,
            "activation": "silu",
            "mlp_act": "silu",
            "reg_max": 32,
            "reg_scale": 4,
        },
    },
    "m": {
        "input_size": 640,
        "backbone": {
            "type": "dinov3",
            "name": "vit_tinyplus",
            "embed_dim": 256,
            "interaction_indexes": [3, 7, 11],
            "num_heads": 4,
        },
        "encoder": {
            "type": "hybrid",
            "in_channels": [256, 256, 256],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "dim_feedforward": 512,
            "expansion": 0.67,
            "depth_mult": 1.0,
            "use_encoder_idx": [2],
            "version": "deim",
            "act": "silu",
            "csp_type": "csp2",
            "fuse_op": "sum",
        },
        "decoder": {
            "eval_spatial_size": [640, 640],
            "feat_channels": [256, 256, 256],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "num_levels": 3,
            "num_points": [3, 6, 3],
            "num_layers": 4,
            "eval_idx": -1,
            "num_queries": 300,
            "dim_feedforward": 512,
            "activation": "silu",
            "mlp_act": "silu",
            "reg_max": 32,
            "reg_scale": 4,
        },
    },
    "l": {
        "input_size": 640,
        "backbone": {
            "type": "dinov3",
            "name": "dinov3_vits16",
            "embed_dim": 224,
            "hidden_dim": 224,
            "conv_inplane": 32,
            "interaction_indexes": [5, 8, 11],
            "num_heads": None,
        },
        "encoder": {
            "type": "hybrid",
            "in_channels": [224, 224, 224],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 224,
            "dim_feedforward": 896,
            "expansion": 1.0,
            "depth_mult": 1.0,
            "use_encoder_idx": [2],
            "version": "deim",
            "act": "silu",
            "csp_type": "csp2",
            "fuse_op": "sum",
        },
        "decoder": {
            "eval_spatial_size": [640, 640],
            "feat_channels": [224, 224, 224],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 224,
            "num_levels": 3,
            "num_points": [3, 6, 3],
            "num_layers": 4,
            "eval_idx": -1,
            "num_queries": 300,
            "dim_feedforward": 1792,
            "activation": "silu",
            "mlp_act": "silu",
            "reg_max": 32,
            "reg_scale": 4,
        },
    },
    "x": {
        "input_size": 640,
        "backbone": {
            "type": "dinov3",
            "name": "dinov3_vits16plus",
            "embed_dim": 256,
            "hidden_dim": 256,
            "conv_inplane": 64,
            "interaction_indexes": [5, 8, 11],
            "num_heads": None,
        },
        "encoder": {
            "type": "hybrid",
            "in_channels": [256, 256, 256],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "dim_feedforward": 1024,
            "expansion": 1.25,
            "depth_mult": 1.37,
            "use_encoder_idx": [2],
            "version": "deim",
            "act": "silu",
            "csp_type": "csp2",
            "fuse_op": "sum",
        },
        "decoder": {
            "eval_spatial_size": [640, 640],
            "feat_channels": [256, 256, 256],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "num_levels": 3,
            "num_points": [3, 6, 3],
            "num_layers": 6,
            "eval_idx": -1,
            "num_queries": 300,
            "dim_feedforward": 2048,
            "activation": "silu",
            "mlp_act": "silu",
            "reg_max": 32,
            "reg_scale": 4,
        },
    },
}


def normalize_size(size: str) -> str:
    return SIZE_ALIASES.get(size.lower(), size.lower())


class LibreDEIMv2Model(nn.Module):
    """Backbone + encoder + DEIMv2 transformer decoder."""

    def __init__(self, config: str, nb_classes: int = 80):
        super().__init__()
        config = normalize_size(config)
        if config not in SIZE_CONFIGS:
            raise ValueError(f"Unknown DEIMv2 size: {config!r}")

        cfg = copy.deepcopy(SIZE_CONFIGS[config])
        self.config = config
        self.uses_imagenet_norm = cfg["backbone"]["type"] == "dinov3"

        backbone_cfg = cfg["backbone"]
        backbone_type = backbone_cfg.pop("type")
        if backbone_type == "hgnetv2":
            self.backbone = HGNetv2(**backbone_cfg)
        elif backbone_type == "dinov3":
            self.backbone = DINOv3STAs(**backbone_cfg)
        else:
            raise ValueError(f"Unsupported DEIMv2 backbone type: {backbone_type}")

        encoder_cfg = cfg["encoder"]
        encoder_type = encoder_cfg.pop("type")
        if encoder_type == "lite":
            self.encoder = LiteEncoder(**encoder_cfg)
        elif encoder_type == "hybrid":
            self.encoder = HybridEncoder(**encoder_cfg)
        else:
            raise ValueError(f"Unsupported DEIMv2 encoder type: {encoder_type}")

        decoder_cfg = cfg["decoder"]
        self.decoder = DEIMTransformer(num_classes=nb_classes, **decoder_cfg)

    def forward(self, x: torch.Tensor, targets: list[dict] | None = None):
        feats = self.backbone(x)
        feats = self.encoder(feats)
        return self.decoder(feats, targets=targets)

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy") and m is not self:
                m.convert_to_deploy()
        return self


class DEIMv2ExportWrapper(nn.Module):
    """Tracing-friendly tuple-output wrapper."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.deploy()

    def forward(self, x):
        out = self.model(x)
        return out["pred_logits"], out["pred_boxes"]
