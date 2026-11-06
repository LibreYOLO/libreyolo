"""Top-level RiO-DETRv2 module: HGNetv2 backbone + RT-DETRv2 OBB encoder + RiO decoder."""

from __future__ import annotations

from typing import Any

from torch import nn

from ..dfine.backbone import HGNetv2
from ..rtdetrv2.obb_encoder import RTDETRv2OBBHybridEncoder
from .decoder import RiODETRv2Transformer

# Released RiO-DETRv2 configurations (``configs/rio-*/riov2_hgnetv2_*``).
# The released L/X configs set ``use_lab: true``, but the released L/X
# checkpoints carry no LearnableAffineBlock tensors: they load strictly only
# with ``use_lab=False`` (the D-FINE default for B4/B5), which is used here.
RIODETRV2_CONFIGS: dict[str, dict[str, Any]] = {
    "n": {
        "backbone": "B0", "use_lab": True, "return_idx": (2, 3),
        "in_channels": (512, 1024), "feat_strides": (16, 32), "hidden_dim": 128,
        "dim_feedforward": 512, "expansion": 0.34, "depth_mult": 0.5,
        "use_encoder_idx": (1,), "num_layers": 3, "num_points": (6, 6), "reg_scale": 4.0,
    },
    "s": {
        "backbone": "B0", "use_lab": True, "return_idx": (1, 2, 3),
        "in_channels": (256, 512, 1024), "feat_strides": (8, 16, 32), "hidden_dim": 224,
        "dim_feedforward": 1024, "expansion": 0.5, "depth_mult": 0.34,
        "use_encoder_idx": (2,), "num_layers": 3, "num_points": (4, 4, 4), "reg_scale": 4.0,
    },
    "m": {
        "backbone": "B2", "use_lab": True, "return_idx": (1, 2, 3),
        "in_channels": (384, 768, 1536), "feat_strides": (8, 16, 32), "hidden_dim": 256,
        "dim_feedforward": 1024, "expansion": 1.0, "depth_mult": 0.67,
        "use_encoder_idx": (2,), "num_layers": 3, "num_points": (4, 4, 4), "reg_scale": 4.0,
    },
    "l": {
        "backbone": "B4", "use_lab": False, "return_idx": (1, 2, 3),
        "in_channels": (512, 1024, 2048), "feat_strides": (8, 16, 32), "hidden_dim": 256,
        "dim_feedforward": 1024, "expansion": 1.0, "depth_mult": 0.67,
        "use_encoder_idx": (2,), "num_layers": 4, "num_points": (4, 4, 4), "reg_scale": 4.0,
    },
    "x": {
        "backbone": "B5", "use_lab": False, "return_idx": (1, 2, 3),
        "in_channels": (512, 1024, 2048), "feat_strides": (8, 16, 32), "hidden_dim": 384,
        "dim_feedforward": 2048, "expansion": 1.0, "depth_mult": 0.67,
        "use_encoder_idx": (2,), "num_layers": 4, "num_points": (4, 4, 4), "reg_scale": 8.0,
    },
}


class RiODETRv2Model(nn.Module):
    """RiO-DETRv2 oriented detector; emits normalized ``(cx, cy, w, h, angle/pi)``."""

    def __init__(self, size: str, num_classes: int = 15, eval_spatial_size=(1024, 1024)):
        super().__init__()
        if size not in RIODETRV2_CONFIGS:
            raise ValueError(f"Unknown RiO-DETRv2 size: {size!r}")
        cfg = RIODETRV2_CONFIGS[size]
        eval_spatial_size = tuple(eval_spatial_size)
        hidden_dim = cfg["hidden_dim"]
        self.backbone = HGNetv2(
            name=cfg["backbone"],
            use_lab=cfg["use_lab"],
            return_idx=cfg["return_idx"],
            freeze_stem_only=False,
            freeze_at=-1,
            freeze_norm=False,
            pretrained=False,
        )
        self.encoder = RTDETRv2OBBHybridEncoder(
            in_channels=list(cfg["in_channels"]),
            feat_strides=list(cfg["feat_strides"]),
            hidden_dim=hidden_dim,
            nhead=8,
            dim_feedforward=cfg["dim_feedforward"],
            expansion=cfg["expansion"],
            depth_mult=cfg["depth_mult"],
            use_encoder_idx=list(cfg["use_encoder_idx"]),
            dropout=0.0,
            eval_spatial_size=eval_spatial_size,
        )
        self.decoder = RiODETRv2Transformer(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=300,
            feat_channels=[hidden_dim] * len(cfg["in_channels"]),
            feat_strides=list(cfg["feat_strides"]),
            num_levels=len(cfg["feat_strides"]),
            num_points=list(cfg["num_points"]),
            nhead=8,
            num_layers=cfg["num_layers"],
            dim_feedforward=1024,
            dropout=0.0,
            num_denoising=100,
            eval_spatial_size=eval_spatial_size,
            eval_idx=-1,
            reg_max=32,
            reg_scale=cfg["reg_scale"],
            angle_bins=36,
            angle_fdr_scale_factor=1.5,
        )

    @property
    def eval_spatial_size(self):
        return self.decoder.eval_spatial_size

    def set_eval_spatial_size(self, eval_spatial_size) -> None:
        """Move the fixed-canvas constants (anchors, AIFI position embedding)."""
        eval_spatial_size = tuple(int(v) for v in eval_spatial_size)
        if tuple(self.decoder.eval_spatial_size) == eval_spatial_size:
            return
        self.decoder.set_eval_spatial_size(eval_spatial_size)
        self.encoder.eval_spatial_size = eval_spatial_size
        self.encoder._pos_embed_cache.clear()
        self.encoder._reset_parameters()

    def forward(self, x, targets=None):
        return self.decoder(self.encoder(self.backbone(x)), targets=targets)
