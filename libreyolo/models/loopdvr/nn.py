"""LoopDVR top-level nn.Module definitions (LoopDet, LoopSeg, LoopPose) and export wrapper.

State-dict keys mirror Intellindust-AI-Lab/LoopDVR (commit f2aba1f122029e4db757b69cd37fc5a4909db34e)
verbatim (`looped_vit.*`, `encoder.*`, `decoder.*`) so official release checkpoints
load with `strict=True` and conversion is a pure metadata-wrap.
"""

from __future__ import annotations

import torch
from torch import nn

from .backbone import LoopedViT
from .decoder import DetectionDecoder, PoseDecoder
from .encoder import HybridEncoder

_GROUP_BOUNDARIES = ((0, 2), (2, 4), (4, 6), (6, 9), (9, 11), (11, 12))
_INTERACTION_INDEXES = (10, 11)
_FEAT_STRIDES = (8, 16, 32)
_DECODER_GROUP_SIZES = (2, 2)

# Verified against all 12 official YAML configs in Intellindust-AI-Lab/LoopDVR.
SIZE_CONFIGS: dict[str, dict] = {
    "s": {
        "embed_dim": 192,
        "num_heads": 3,
        "ffn_ratio": 4.0,
        "proj_dim": None,
        "adapter_hidden_dim": 192,
        "enc_hidden_dim": 192,
        "enc_dim_feedforward": 512,
        "enc_expansion": 0.34,
        "enc_depth_mult": 0.67,
        "dec_hidden_dim": 192,
        "dec_dim_feedforward": 512,
        "pose_dec_dim_feedforward": 512,
        "dec_adapter_hidden_dim": 192,
    },
    "m": {
        "embed_dim": 256,
        "num_heads": 4,
        "ffn_ratio": 4.0,
        "proj_dim": None,
        "adapter_hidden_dim": 256,
        "enc_hidden_dim": 256,
        "enc_dim_feedforward": 512,
        "enc_expansion": 0.75,
        "enc_depth_mult": 0.67,
        "dec_hidden_dim": 256,
        "dec_dim_feedforward": 1024,
        # LoopPose-M uses 512 in the pose decoder FFN (looppose-m.yaml),
        # whereas LoopDet-M and LoopSeg-M use 1024.
        "pose_dec_dim_feedforward": 512,
        "dec_adapter_hidden_dim": 256,
    },
    "l": {
        "embed_dim": 384,
        "num_heads": 6,
        "ffn_ratio": 4.0,
        "proj_dim": 256,
        "adapter_hidden_dim": 384,
        "enc_hidden_dim": 256,
        "enc_dim_feedforward": 1024,
        "enc_expansion": 0.75,
        "enc_depth_mult": 1,
        "dec_hidden_dim": 256,
        "dec_dim_feedforward": 1024,
        "pose_dec_dim_feedforward": 1024,
        "dec_adapter_hidden_dim": 256,
    },
    "x": {
        "embed_dim": 384,
        "num_heads": 6,
        "ffn_ratio": 6.0,
        "proj_dim": 256,
        "adapter_hidden_dim": 384,
        "enc_hidden_dim": 256,
        "enc_dim_feedforward": 2048,
        "enc_expansion": 1.5,
        "enc_depth_mult": 1,
        "dec_hidden_dim": 256,
        "dec_dim_feedforward": 2048,
        "pose_dec_dim_feedforward": 2048,
        "dec_adapter_hidden_dim": 256,
    },
}


def _build_backbone_and_encoder(
    cfg: dict, eval_spatial_size: tuple[int, int]
) -> tuple[LoopedViT, HybridEncoder]:
    looped_vit = LoopedViT(
        group_boundaries=_GROUP_BOUNDARIES,
        interaction_indexes=_INTERACTION_INDEXES,
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        ffn_ratio=cfg["ffn_ratio"],
        proj_dim=cfg["proj_dim"],
        adapter_hidden_dim=cfg["adapter_hidden_dim"],
    )
    hidden_dim = cfg["enc_hidden_dim"]
    encoder = HybridEncoder(
        in_channels=(hidden_dim, hidden_dim, hidden_dim),
        feat_strides=_FEAT_STRIDES,
        hidden_dim=hidden_dim,
        nhead=8,
        dim_feedforward=cfg["enc_dim_feedforward"],
        use_encoder_idx=(2,),
        num_encoder_layers=1,
        expansion=cfg["enc_expansion"],
        depth_mult=cfg["enc_depth_mult"],
        act="silu",
        eval_spatial_size=eval_spatial_size,
        dropout=0.0,
        csp_type="csp2",
        fuse_op="sum",
    )
    return looped_vit, encoder


class _BaseLoopDVRModule(nn.Module):
    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy") and m is not self:
                m.convert_to_deploy()
        return self


class LibreLoopDVRModel(_BaseLoopDVRModule):
    """LoopDet: LoopedViT + HybridEncoder + recurrent DetectionDecoder."""

    def __init__(
        self,
        config: str,
        nb_classes: int = 80,
        eval_spatial_size: tuple[int, int] = (640, 640),
    ):
        super().__init__()
        if config not in SIZE_CONFIGS:
            raise ValueError(f"Unknown LoopDVR size: {config!r}")
        cfg = SIZE_CONFIGS[config]
        self.config = config
        self.looped_vit, self.encoder = _build_backbone_and_encoder(
            cfg, eval_spatial_size
        )
        self.decoder = DetectionDecoder(
            hidden_dim=cfg["dec_hidden_dim"],
            dim_feedforward=cfg["dec_dim_feedforward"],
            adapter_hidden_dim=cfg["dec_adapter_hidden_dim"],
            num_queries=300,
            eval_spatial_size=eval_spatial_size,
            num_classes=nb_classes,
            num_points=(3, 6, 3),
            group_sizes=_DECODER_GROUP_SIZES,
            feat_strides=_FEAT_STRIDES,
            nhead=8,
            reg_max=32,
            reg_scale=4,
            activation="silu",
            mask_downsample_ratio=None,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(self.looped_vit(x))
        return self.decoder(features)


class LibreLoopDVRSegModel(_BaseLoopDVRModule):
    """LoopSeg: LoopedViT + HybridEncoder + recurrent DetectionDecoder with mask head."""

    def __init__(
        self,
        config: str,
        nb_classes: int = 80,
        eval_spatial_size: tuple[int, int] = (640, 640),
    ):
        super().__init__()
        if config not in SIZE_CONFIGS:
            raise ValueError(f"Unknown LoopDVR seg size: {config!r}")
        cfg = SIZE_CONFIGS[config]
        self.config = config
        self.looped_vit, self.encoder = _build_backbone_and_encoder(
            cfg, eval_spatial_size
        )
        self.decoder = DetectionDecoder(
            hidden_dim=cfg["dec_hidden_dim"],
            dim_feedforward=cfg["dec_dim_feedforward"],
            adapter_hidden_dim=cfg["dec_adapter_hidden_dim"],
            num_queries=300,
            eval_spatial_size=eval_spatial_size,
            num_classes=nb_classes,
            num_points=(3, 6, 3),
            group_sizes=_DECODER_GROUP_SIZES,
            feat_strides=_FEAT_STRIDES,
            nhead=8,
            reg_max=32,
            reg_scale=4,
            activation="silu",
            mask_downsample_ratio=4,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(self.looped_vit(x))
        return self.decoder(features, spatial_feat=features[0])


class LibreLoopDVRPoseModel(_BaseLoopDVRModule):
    """LoopPose: LoopedViT + HybridEncoder + recurrent PoseDecoder."""

    POSE_NUM_KEYPOINTS = 17
    POSE_NUM_CLASSES = 2

    def __init__(
        self,
        config: str,
        eval_spatial_size: tuple[int, int] = (640, 640),
    ):
        super().__init__()
        if config not in SIZE_CONFIGS:
            raise ValueError(f"Unknown LoopDVR pose size: {config!r}")
        cfg = SIZE_CONFIGS[config]
        self.config = config
        self.looped_vit, self.encoder = _build_backbone_and_encoder(
            cfg, eval_spatial_size
        )
        self.decoder = PoseDecoder(
            hidden_dim=cfg["dec_hidden_dim"],
            dim_feedforward=cfg["pose_dec_dim_feedforward"],
            adapter_hidden_dim=cfg["dec_adapter_hidden_dim"],
            num_queries=60,
            eval_spatial_size=eval_spatial_size,
            num_classes=self.POSE_NUM_CLASSES,
            num_points=4,
            group_sizes=_DECODER_GROUP_SIZES,
            feat_strides=_FEAT_STRIDES,
            nhead=8,
            reg_max=32,
            reg_scale=4,
            activation="relu",
            num_body_points=self.POSE_NUM_KEYPOINTS,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(self.looped_vit(x))
        return self.decoder(features)


class LoopDVRExportWrapper(nn.Module):
    """Tracing-friendly wrapper for ONNX / TorchScript export."""

    def __init__(self, model: _BaseLoopDVRModule, task: str = "detect"):
        super().__init__()
        self.model = model
        self.task = task
        self.model.deploy()

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        if self.task == "segment":
            return out["pred_logits"], out["pred_boxes"], out["pred_masks"]
        if self.task == "pose":
            keypoints = out["pred_keypoints"]
            if keypoints.dim() == 4:
                keypoints = keypoints.flatten(-2)
            return out["pred_logits"], keypoints
        return out["pred_logits"], out["pred_boxes"]
