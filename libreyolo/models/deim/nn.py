"""Top-level LibreDEIM model and per-size configuration table.

The size configs here are read directly from DEIM's shipped YAMLs:
``configs/deim/deim_hgnetv2_{n,s,m,l,x}_coco.yml`` + the base include
``configs/deim/include/deim_hgnetv2.yml``.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .backbone import HGNetv2
from .decoder import DEIMTransformer
from .encoder import HybridEncoder


# Per-size parameters. Fields map directly to DEIM YAML knobs.
SIZE_CONFIGS: Dict[str, Dict] = {
    "n": {
        "backbone": "B0",
        "use_lab": True,
        "return_idx": (2, 3),
        "freeze_stem_only": True,
        "freeze_at": 0,
        "freeze_norm": False,
        # encoder
        "enc_in_channels": (512, 1024),
        "enc_feat_strides": (16, 32),
        "enc_hidden_dim": 128,
        "enc_dim_feedforward": 512,
        "enc_expansion": 0.34,
        "enc_depth_mult": 0.5,
        "enc_use_encoder_idx": (1,),
        # decoder
        "dec_feat_channels": (128, 128),
        "dec_feat_strides": (16, 32),
        "dec_hidden_dim": 128,
        "dec_dim_feedforward": 512,
        "dec_num_levels": 2,
        "dec_num_layers": 3,
        "dec_num_points": (6, 6),
        "dec_eval_idx": -1,
        "reg_scale": 4.0,
        "activation": "silu",
        "mlp_act": "silu",
    },
    "s": {
        "backbone": "B0",
        "use_lab": True,
        "return_idx": (1, 2, 3),
        "freeze_stem_only": True,
        "freeze_at": 0,
        "freeze_norm": False,
        "enc_in_channels": (256, 512, 1024),
        "enc_feat_strides": (8, 16, 32),
        "enc_hidden_dim": 256,
        "enc_dim_feedforward": 1024,
        "enc_expansion": 0.5,
        "enc_depth_mult": 0.34,
        "enc_use_encoder_idx": (2,),
        "dec_feat_channels": (256, 256, 256),
        "dec_feat_strides": (8, 16, 32),
        "dec_hidden_dim": 256,
        "dec_dim_feedforward": 1024,
        "dec_num_levels": 3,
        "dec_num_layers": 3,
        "dec_num_points": (3, 6, 3),
        "dec_eval_idx": -1,
        "reg_scale": 4.0,
        "activation": "silu",
        "mlp_act": "silu",
    },
    "m": {
        "backbone": "B2",
        "use_lab": True,
        "return_idx": (1, 2, 3),
        "freeze_stem_only": True,
        "freeze_at": 0,
        "freeze_norm": False,
        "enc_in_channels": (384, 768, 1536),
        "enc_feat_strides": (8, 16, 32),
        "enc_hidden_dim": 256,
        "enc_dim_feedforward": 1024,
        "enc_expansion": 1.0,
        "enc_depth_mult": 0.67,
        "enc_use_encoder_idx": (2,),
        "dec_feat_channels": (256, 256, 256),
        "dec_feat_strides": (8, 16, 32),
        "dec_hidden_dim": 256,
        "dec_dim_feedforward": 1024,
        "dec_num_levels": 3,
        "dec_num_layers": 4,
        "dec_num_points": (3, 6, 3),
        "dec_eval_idx": -1,
        "reg_scale": 4.0,
        "activation": "silu",
        "mlp_act": "silu",
    },
    "l": {
        "backbone": "B4",
        "use_lab": False,
        "return_idx": (1, 2, 3),
        "freeze_stem_only": True,
        "freeze_at": 0,
        "freeze_norm": True,
        "enc_in_channels": (512, 1024, 2048),
        "enc_feat_strides": (8, 16, 32),
        "enc_hidden_dim": 256,
        "enc_dim_feedforward": 1024,
        "enc_expansion": 1.0,
        "enc_depth_mult": 1.0,
        "enc_use_encoder_idx": (2,),
        "dec_feat_channels": (256, 256, 256),
        "dec_feat_strides": (8, 16, 32),
        "dec_hidden_dim": 256,
        "dec_dim_feedforward": 1024,
        "dec_num_levels": 3,
        "dec_num_layers": 6,
        "dec_num_points": (3, 6, 3),
        "dec_eval_idx": -1,
        "reg_scale": 4.0,
        "activation": "silu",
        "mlp_act": "silu",
    },
    "x": {
        "backbone": "B5",
        "use_lab": False,
        "return_idx": (1, 2, 3),
        "freeze_stem_only": True,
        "freeze_at": 0,
        "freeze_norm": True,
        "enc_in_channels": (512, 1024, 2048),
        "enc_feat_strides": (8, 16, 32),
        "enc_hidden_dim": 384,
        "enc_dim_feedforward": 2048,
        "enc_expansion": 1.0,
        "enc_depth_mult": 1.0,
        "enc_use_encoder_idx": (2,),
        # X overrides encoder hidden_dim to 384 while inheriting the decoder
        # hidden_dim=256 default from include/deim_hgnetv2.yml. The decoder's
        # input_proj thus has 384->256 convs (unique to X).
        "dec_feat_channels": (384, 384, 384),
        "dec_feat_strides": (8, 16, 32),
        "dec_hidden_dim": 256,
        "dec_dim_feedforward": 1024,
        "dec_num_levels": 3,
        "dec_num_layers": 6,
        "dec_num_points": (3, 6, 3),
        "dec_eval_idx": -1,
        "reg_scale": 8.0,
        "activation": "silu",
        "mlp_act": "silu",
    },
}


class LibreDEIMModel(nn.Module):
    """Backbone + hybrid encoder + DEIM decoder, wired for inference."""

    def __init__(
        self,
        config: str,
        nb_classes: int = 80,
        eval_spatial_size: tuple[int, int] | None = (640, 640),
    ):
        super().__init__()
        if config not in SIZE_CONFIGS:
            raise ValueError(f"Unknown DEIM size: {config!r}")
        cfg = SIZE_CONFIGS[config]
        self.config = config

        self.backbone = HGNetv2(
            name=cfg["backbone"],
            use_lab=cfg["use_lab"],
            return_idx=cfg["return_idx"],
            freeze_stem_only=cfg["freeze_stem_only"],
            freeze_at=cfg["freeze_at"],
            freeze_norm=cfg["freeze_norm"],
            pretrained=False,
        )
        self.encoder = HybridEncoder(
            in_channels=cfg["enc_in_channels"],
            feat_strides=cfg["enc_feat_strides"],
            hidden_dim=cfg["enc_hidden_dim"],
            dim_feedforward=cfg["enc_dim_feedforward"],
            expansion=cfg["enc_expansion"],
            depth_mult=cfg["enc_depth_mult"],
            use_encoder_idx=cfg["enc_use_encoder_idx"],
            eval_spatial_size=eval_spatial_size,
        )
        self.decoder = DEIMTransformer(
            num_classes=nb_classes,
            hidden_dim=cfg["dec_hidden_dim"],
            feat_channels=cfg["dec_feat_channels"],
            feat_strides=cfg["dec_feat_strides"],
            num_levels=cfg["dec_num_levels"],
            num_points=cfg["dec_num_points"],
            num_layers=cfg["dec_num_layers"],
            dim_feedforward=cfg["dec_dim_feedforward"],
            eval_spatial_size=eval_spatial_size,
            eval_idx=cfg["dec_eval_idx"],
            reg_scale=cfg["reg_scale"],
            activation=cfg["activation"],
            mlp_act=cfg["mlp_act"],
        )

    def forward(self, x: torch.Tensor, targets: List[dict] | None = None):
        feats = self.backbone(x)
        feats = self.encoder(feats)
        return self.decoder(feats, targets=targets)

    def deploy(self):
        """Fuse BN into conv + strip non-eval decoder layers for export."""
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy") and m is not self:
                m.convert_to_deploy()
        return self


class DEIMExportWrapper(nn.Module):
    """Tracing-friendly wrapper for ONNX/TorchScript export.

    The raw ``LibreDEIMModel`` returns a dict ``{"pred_logits", "pred_boxes"}``
    which `torch.onnx.export` cannot trace cleanly with named outputs. This
    wrapper deploys the model (BN fusion + prune non-eval layers) once and
    flattens the output to a tuple in declaration order.
    """

    def __init__(self, model: LibreDEIMModel):
        super().__init__()
        self.model = model
        # Idempotent if deploy() has already been called.
        self.model.deploy()

    def forward(self, x):
        out = self.model(x)
        return out["pred_logits"], out["pred_boxes"]
