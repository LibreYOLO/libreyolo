"""Top-level LibreECDet model and per-size config table.

Sizes follow upstream ``ecdetseg/configs/ecdet/ecdet_{s,m,l,x}.yml``. All four
share the same decoder (4 layers, num_points=[3,6,3], reg_max=32, reg_scale=4)
and differ in backbone embedding / projector dim and encoder/decoder widths.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .backbone import ViTAdapter
from .decoder import ECTransformer
from .encoder import HybridEncoder


SIZE_CONFIGS: Dict[str, Dict] = {
    "s": {
        # backbone
        "embed_dim": 192,
        "num_heads": 3,
        "proj_dim": None,  # defaults to embed_dim
        # encoder
        "enc_in_channels": (192, 192, 192),
        "enc_hidden_dim": 192,
        "enc_dim_feedforward": 512,
        "enc_expansion": 0.34,
        "enc_depth_mult": 0.67,
        # decoder
        "dec_feat_channels": (192, 192, 192),
        "dec_hidden_dim": 192,
        "dec_dim_feedforward": 512,
    },
    "m": {
        "embed_dim": 256,
        "num_heads": 4,
        "proj_dim": None,
        "enc_in_channels": (256, 256, 256),
        "enc_hidden_dim": 256,
        "enc_dim_feedforward": 512,
        "enc_expansion": 0.75,
        "enc_depth_mult": 0.67,
        "dec_feat_channels": (256, 256, 256),
        "dec_hidden_dim": 256,
        "dec_dim_feedforward": 1024,
    },
    "l": {
        "embed_dim": 384,
        "num_heads": 6,
        "proj_dim": 256,
        "enc_in_channels": (256, 256, 256),
        "enc_hidden_dim": 256,
        "enc_dim_feedforward": 1024,
        "enc_expansion": 0.75,
        "enc_depth_mult": 1.0,
        "dec_feat_channels": (256, 256, 256),
        "dec_hidden_dim": 256,
        "dec_dim_feedforward": 1024,
    },
    "x": {
        "embed_dim": 384,
        "num_heads": 6,
        "proj_dim": 256,
        "enc_in_channels": (256, 256, 256),
        "enc_hidden_dim": 256,
        "enc_dim_feedforward": 2048,
        "enc_expansion": 1.5,
        "enc_depth_mult": 1.0,
        "dec_feat_channels": (256, 256, 256),
        "dec_hidden_dim": 256,
        "dec_dim_feedforward": 2048,
    },
}


class LibreECDetModel(nn.Module):
    """Backbone (ECViT + adapter) + HybridEncoder + ECTransformer."""

    def __init__(
        self,
        config: str,
        nb_classes: int = 80,
        eval_spatial_size: tuple[int, int] | None = (640, 640),
    ):
        super().__init__()
        if config not in SIZE_CONFIGS:
            raise ValueError(f"Unknown ECDet size: {config!r}")
        cfg = SIZE_CONFIGS[config]
        self.config = config

        self.backbone = ViTAdapter(
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            proj_dim=cfg["proj_dim"],
            interaction_indexes=(10, 11),
        )
        self.encoder = HybridEncoder(
            in_channels=cfg["enc_in_channels"],
            hidden_dim=cfg["enc_hidden_dim"],
            dim_feedforward=cfg["enc_dim_feedforward"],
            expansion=cfg["enc_expansion"],
            depth_mult=cfg["enc_depth_mult"],
            eval_spatial_size=list(eval_spatial_size) if eval_spatial_size else None,
        )
        self.decoder = ECTransformer(
            num_classes=nb_classes,
            hidden_dim=cfg["dec_hidden_dim"],
            feat_channels=cfg["dec_feat_channels"],
            dim_feedforward=cfg["dec_dim_feedforward"],
            num_layers=4,
            num_points=(3, 6, 3),
            eval_idx=-1,
            reg_max=32,
            reg_scale=4.0,
            eval_spatial_size=list(eval_spatial_size) if eval_spatial_size else None,
        )

    def forward(self, x: torch.Tensor, targets: List[dict] | None = None):
        feats = self.backbone(x)
        feats = self.encoder(feats)
        return self.decoder(feats, targets=targets)

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy") and m is not self:
                m.convert_to_deploy()
        return self


class ECDetExportWrapper(nn.Module):
    """Tracing-friendly wrapper for ONNX/TorchScript export."""

    def __init__(self, model: LibreECDetModel):
        super().__init__()
        self.model = model
        self.model.deploy()

    def forward(self, x):
        out = self.model(x)
        return out["pred_logits"], out["pred_boxes"]
