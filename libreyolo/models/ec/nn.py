"""Top-level LibreEC model and per-size config table.

Sizes follow upstream ``ecseg/configs/ec/ec_{s,m,l,x}.yml``. All four
share the same decoder (4 layers, num_points=[3,6,3], reg_max=32, reg_scale=4)
and differ in backbone embedding / projector dim and encoder/decoder widths.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .backbone import ViTAdapter
from .decoder import ECPoseTransformer, ECTransformer
from .encoder import HybridEncoder


# Pose-specific per-size diffs (overlay on SIZE_CONFIGS' backbone+encoder).
# All four sizes share num_classes=2 (DETRPose criterion uses 2 classes:
# person + bg) and num_keypoints=17 (COCO). Decoder layer counts and
# dim_feedforward differ per size.
POSE_SIZE_OVERRIDES: Dict[str, Dict] = {
    "s": {"dec_num_layers": 3, "dec_dim_feedforward": 512},
    "m": {"dec_num_layers": 4, "dec_dim_feedforward": 512},
    "l": {"dec_num_layers": 6, "dec_dim_feedforward": 1024},
    "x": {"dec_num_layers": 6, "dec_dim_feedforward": 2048},
}


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
        "ffn_ratio": 6.0,  # ecvitsplus uses 6, all other variants use the default 4
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


class LibreECModel(nn.Module):
    """Backbone (ECViT + adapter) + HybridEncoder + ECTransformer."""

    def __init__(
        self,
        config: str,
        nb_classes: int = 80,
        eval_spatial_size: tuple[int, int] | None = (640, 640),
    ):
        super().__init__()
        if config not in SIZE_CONFIGS:
            raise ValueError(f"Unknown EC size: {config!r}")
        cfg = SIZE_CONFIGS[config]
        self.config = config

        self.backbone = ViTAdapter(
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            proj_dim=cfg["proj_dim"],
            ffn_ratio=cfg.get("ffn_ratio", 4.0),
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


class LibreECSegModel(nn.Module):
    """Backbone (ECViT + adapter) + HybridEncoder + ECTransformer with
    segmentation head.

    Sibling of :class:`LibreECModel`. Same backbone+encoder; the decoder
    gains a :class:`SegmentationHead` (downsample_ratio=4) and the forward
    plumbs encoder ``feats[0]`` (highest-resolution stride-8 feature map) into
    the seg head as ``spatial_feat``.
    """

    def __init__(
        self,
        config: str,
        nb_classes: int = 80,
        eval_spatial_size: tuple[int, int] | None = (640, 640),
    ):
        super().__init__()
        if config not in SIZE_CONFIGS:
            raise ValueError(f"Unknown EC seg size: {config!r}")
        cfg = SIZE_CONFIGS[config]
        self.config = config

        self.backbone = ViTAdapter(
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            proj_dim=cfg["proj_dim"],
            ffn_ratio=cfg.get("ffn_ratio", 4.0),
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
            mask_downsample_ratio=4,
        )

    def forward(self, x: torch.Tensor, targets: List[dict] | None = None):
        feats = self.backbone(x)
        feats = self.encoder(feats)
        return self.decoder(feats, targets=targets, spatial_feat=feats[0])

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy") and m is not self:
                m.convert_to_deploy()
        return self


class LibreECPoseModel(nn.Module):
    """Backbone (ECViT + adapter) + HybridEncoder + ECPoseTransformer.

    Sibling of :class:`LibreECModel` for the pose task. Reuses the same
    backbone and encoder; swaps the box-regression decoder for the pose
    DETR transformer with iterative DFL keypoint refinement.
    """

    POSE_NUM_KEYPOINTS = 17
    POSE_NUM_CLASSES = 2  # ECPose's DETRPose criterion uses 2-class logits

    def __init__(
        self,
        config: str,
        eval_spatial_size: tuple[int, int] | None = (640, 640),
    ):
        super().__init__()
        if config not in SIZE_CONFIGS or config not in POSE_SIZE_OVERRIDES:
            raise ValueError(f"Unknown EC pose size: {config!r}")
        cfg = SIZE_CONFIGS[config]
        pcfg = POSE_SIZE_OVERRIDES[config]
        self.config = config

        self.backbone = ViTAdapter(
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            proj_dim=cfg["proj_dim"],
            ffn_ratio=cfg.get("ffn_ratio", 4.0),
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
        self.decoder = ECPoseTransformer(
            hidden_dim=cfg["dec_hidden_dim"],
            num_queries=60,
            num_decoder_layers=pcfg["dec_num_layers"],
            dim_feedforward=pcfg["dec_dim_feedforward"],
            num_feature_levels=3,
            dec_n_points=4,
            num_keypoints=self.POSE_NUM_KEYPOINTS,
            num_classes=self.POSE_NUM_CLASSES,
            feat_strides=(8, 16, 32),
            eval_spatial_size=list(eval_spatial_size) if eval_spatial_size else None,
            reg_max=32,
            reg_scale=4.0,
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


class ECExportWrapper(nn.Module):
    """Tracing-friendly wrapper for ONNX/TorchScript export."""

    def __init__(self, model: LibreECModel):
        super().__init__()
        self.model = model
        self.model.deploy()

    def forward(self, x):
        out = self.model(x)
        return out["pred_logits"], out["pred_boxes"]
