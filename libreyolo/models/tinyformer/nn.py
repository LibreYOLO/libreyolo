"""Top-level TinyFormer model wiring (PBM release configs).

Size table flattened from ``configs/tinyformer/tinyformer_dinov3_*_coco_pbm.yml``
in the official release (mmpmmpmmpjosh/TinyFormer). Every released size uses
the 4-scale SSA backbone + PBM neck; the decoder is DEIMv2's DEIMTransformer
with 3 levels at strides (8, 16, 32).
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from ..deimv2.engine.deim.deim_decoder import DEIMTransformer
from .backbone import DINOv3SSAs_4Scale
from .encoder import HybridEncoder_4Scale

# Every TinyFormer size runs on a DINO-lineage tower and expects
# ImageNet-normalised input (unlike DEIMv2, whose tiny sizes are HGNetv2).
DINO_SIZES = {"s", "m", "l", "x", "xl"}


SIZE_CONFIGS: dict[str, dict[str, Any]] = {
    "s": {
        "input_size": 640,
        "backbone": {
            "name": "vit_tiny",
            "embed_dim": 192,
            "interaction_indexes": [3, 7, 11],
            "num_heads": 3,
            "conv_inplane": 16,
            "hidden_dim": 192,
        },
        "encoder": {
            "in_channels": [192, 192, 192, 192],
            "feat_strides": [4, 8, 16, 32],
            "out_indices": [1, 2, 3],
            "hidden_dim": 192,
            "dim_feedforward": 512,
            "expansion": 0.34,
            "depth_mult": 0.67,
            "use_encoder_idx": [3],
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
            "name": "vit_tinyplus",
            "embed_dim": 256,
            "interaction_indexes": [3, 7, 11],
            "num_heads": 4,
            "conv_inplane": 16,
            "hidden_dim": 256,
        },
        "encoder": {
            "in_channels": [256, 256, 256, 256],
            "feat_strides": [4, 8, 16, 32],
            "out_indices": [1, 2, 3],
            "hidden_dim": 256,
            "dim_feedforward": 512,
            "expansion": 0.67,
            "depth_mult": 1.0,
            "use_encoder_idx": [3],
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
            "name": "dinov3_vits16",
            "interaction_indexes": [5, 8, 11],
            "conv_inplane": 32,
            "hidden_dim": 224,
        },
        "encoder": {
            "in_channels": [224, 224, 224, 224],
            "feat_strides": [4, 8, 16, 32],
            "out_indices": [1, 2, 3],
            "hidden_dim": 224,
            "dim_feedforward": 896,
            "expansion": 1.0,
            "depth_mult": 1.0,
            "use_encoder_idx": [3],
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
            "name": "dinov3_vits16plus",
            "interaction_indexes": [5, 8, 11],
            "conv_inplane": 64,
            "hidden_dim": 256,
        },
        "encoder": {
            "in_channels": [256, 256, 256, 256],
            "feat_strides": [4, 8, 16, 32],
            "out_indices": [1, 2, 3],
            "hidden_dim": 256,
            "dim_feedforward": 1024,
            "expansion": 1.25,
            "depth_mult": 1.37,
            "use_encoder_idx": [3],
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
    "xl": {
        "input_size": 640,
        "backbone": {
            "name": "dinov3_vitb16",
            "interaction_indexes": [5, 8, 11],
            "conv_inplane": 128,
            "hidden_dim": 384,
        },
        "encoder": {
            "in_channels": [384, 384, 384, 384],
            "feat_strides": [4, 8, 16, 32],
            "out_indices": [1, 2, 3],
            "hidden_dim": 384,
            "dim_feedforward": 1024,
            "expansion": 1.25,
            "depth_mult": 1.37,
            "use_encoder_idx": [3],
        },
        "decoder": {
            "eval_spatial_size": [640, 640],
            # 384-channel neck output projected down to a 256-wide decoder.
            "feat_channels": [384, 384, 384],
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
    return size.lower()


class LibreTinyFormerModel(nn.Module):
    """SSA backbone + PBM 4-scale encoder + DEIM transformer decoder."""

    def __init__(self, config: str, nb_classes: int = 80):
        super().__init__()
        config = normalize_size(config)
        if config not in SIZE_CONFIGS:
            raise ValueError(f"Unknown TinyFormer size: {config!r}")

        cfg = copy.deepcopy(SIZE_CONFIGS[config])
        self.config = config
        self.uses_imagenet_norm = True

        self.backbone = DINOv3SSAs_4Scale(**cfg["backbone"])
        self.encoder = HybridEncoder_4Scale(**cfg["encoder"])
        self.decoder = DEIMTransformer(num_classes=nb_classes, **cfg["decoder"])

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


class TinyFormerExportWrapper(nn.Module):
    """Tracing-friendly tuple-output wrapper."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.deploy()

    def forward(self, x):
        out = self.model(x)
        return out["pred_logits"], out["pred_boxes"]
