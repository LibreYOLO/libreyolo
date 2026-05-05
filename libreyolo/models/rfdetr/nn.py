"""Native RF-DETR network assembly for LibreYOLO."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .lwdetr import LWDETR, MLP, PostProcess, build_criterion_and_postprocessors, build_model


@dataclass(frozen=True)
class RFDETRSizeConfig:
    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 16
    num_windows: int = 2
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: tuple[str, ...] = ("P4",)
    out_feature_indexes: tuple[int, ...] = (3, 6, 9, 12)
    resolution: int = 512
    positional_encoding_size: int = 32
    pretrain_weights: str | None = None
    segmentation_head: bool = False
    mask_downsample_ratio: int = 4
    license: str = "Apache-2.0"


RFDETR_CONFIGS: dict[str, RFDETRSizeConfig] = {
    "n": RFDETRSizeConfig(
        dec_layers=2,
        resolution=384,
        positional_encoding_size=24,
        pretrain_weights="rf-detr-nano.pth",
    ),
    "s": RFDETRSizeConfig(
        dec_layers=3,
        resolution=512,
        positional_encoding_size=32,
        pretrain_weights="rf-detr-small.pth",
    ),
    "m": RFDETRSizeConfig(
        dec_layers=4,
        resolution=576,
        positional_encoding_size=36,
        pretrain_weights="rf-detr-medium.pth",
    ),
    "l": RFDETRSizeConfig(
        dec_layers=4,
        resolution=704,
        positional_encoding_size=44,
        pretrain_weights="rf-detr-large-2026.pth",
    ),
}


RFDETR_SEG_CONFIGS: dict[str, RFDETRSizeConfig] = {
    "n": RFDETRSizeConfig(
        patch_size=12,
        num_windows=1,
        dec_layers=4,
        resolution=312,
        positional_encoding_size=26,
        num_queries=100,
        num_select=100,
        pretrain_weights="rf-detr-seg-nano.pt",
        segmentation_head=True,
    ),
    "s": RFDETRSizeConfig(
        patch_size=12,
        num_windows=2,
        dec_layers=4,
        resolution=384,
        positional_encoding_size=32,
        num_queries=100,
        num_select=100,
        pretrain_weights="rf-detr-seg-small.pt",
        segmentation_head=True,
    ),
    "m": RFDETRSizeConfig(
        patch_size=12,
        num_windows=2,
        dec_layers=5,
        resolution=432,
        positional_encoding_size=36,
        num_queries=200,
        num_select=200,
        pretrain_weights="rf-detr-seg-medium.pt",
        segmentation_head=True,
    ),
    "l": RFDETRSizeConfig(
        patch_size=12,
        num_windows=2,
        dec_layers=5,
        resolution=504,
        positional_encoding_size=42,
        num_queries=200,
        num_select=200,
        pretrain_weights="rf-detr-seg-large.pt",
        segmentation_head=True,
    ),
}


_PE_KEY_SUFFIX = "embeddings.position_embeddings"


def interpolate_position_embeddings(checkpoint_state: dict[str, torch.Tensor], pe_size: int) -> None:
    """Resize DINOv2 positional embeddings in-place when checkpoint resolution differs."""
    n_target = pe_size * pe_size
    for key in [k for k in checkpoint_state if k.endswith(_PE_KEY_SUFFIX)]:
        ckpt_pe = checkpoint_state[key]
        n_source = ckpt_pe.shape[1] - 1
        if n_source == n_target:
            continue

        h_src = int(math.isqrt(n_source))
        h_tgt = int(math.isqrt(n_target))
        if h_src * h_src != n_source or h_tgt * h_tgt != n_target:
            continue

        dim = ckpt_pe.shape[-1]
        class_token = ckpt_pe[:, :1]
        patch_pe = ckpt_pe[:, 1:].reshape(1, h_src, h_src, dim).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(
            patch_pe.float(),
            size=(h_tgt, h_tgt),
            mode="bicubic",
            align_corners=False,
            antialias=patch_pe.device.type != "mps",
        ).to(ckpt_pe.dtype)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, n_target, dim)
        checkpoint_state[key] = torch.cat([class_token, patch_pe], dim=1)


def _make_args(
    cfg: RFDETRSizeConfig,
    *,
    nb_classes: int,
    device: str,
    segmentation: bool,
) -> SimpleNamespace:
    cfg_values = {
        f.name: list(getattr(cfg, f.name)) if isinstance(getattr(cfg, f.name), tuple) else getattr(cfg, f.name)
        for f in fields(cfg)
    }
    cfg_values["pretrain_weights"] = "__libreyolo_no_backbone_download__"
    cfg_values["segmentation_head"] = segmentation
    return SimpleNamespace(
        **cfg_values,
        amp=True,
        aux_loss=True,
        backbone_lora=False,
        backbone_only=False,
        bbox_loss_coef=5.0,
        bbox_reparam=True,
        cls_loss_coef=5.0 if segmentation else 1.0,
        decoder_norm="LN",
        dim_feedforward=2048,
        drop_path=0.0,
        dropout=0.0,
        encoder_only=False,
        focal_alpha=0.25,
        force_no_pretrain=False,
        freeze_encoder=False,
        giou_loss_coef=2.0,
        gradient_checkpointing=False,
        group_detr=13,
        ia_bce_loss=True,
        layer_norm=True,
        lite_refpoint_refine=True,
        lr_component_decay=0.7,
        lr_encoder=1.5e-4,
        lr_vit_layer_decay=0.8,
        mask_ce_loss_coef=5.0,
        mask_dice_loss_coef=5.0,
        mask_point_sample_ratio=16,
        num_channels=3,
        num_classes=nb_classes,
        position_embedding="sine",
        pretrained_encoder=None,
        rms_norm=False,
        set_cost_bbox=5.0,
        set_cost_class=2.0,
        set_cost_giou=2.0,
        shape=(cfg.resolution, cfg.resolution),
        sum_group_losses=False,
        two_stage=True,
        use_cls_token=False,
        use_position_supervised_loss=False,
        use_varifocal_loss=False,
        vit_encoder_num_layers=12,
        weight_decay=1e-4,
        window_block_indexes=None,
        device=device,
    )


def _unwrap_state_dict(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    normalized = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = key.removeprefix("module.")
        key = key.removeprefix("model.")
        key = key.removeprefix("_orig_mod.")
        normalized[key] = value
    return normalized


def _resize_query_param(tensor: torch.Tensor, target_rows: int) -> torch.Tensor:
    if tensor.shape[0] == target_rows:
        return tensor
    if tensor.shape[0] > target_rows:
        return tensor[:target_rows]
    repeats = math.ceil(target_rows / tensor.shape[0])
    return tensor.repeat(repeats, *([1] * (tensor.ndim - 1)))[:target_rows]


class LibreRFDETRModel(nn.Module):
    """RF-DETR model built from LibreYOLO-local RF-DETR modules."""

    def __init__(
        self,
        config: str = "s",
        nb_classes: int = 80,
        device: str = "cpu",
        segmentation: bool = False,
    ):
        super().__init__()

        configs = RFDETR_SEG_CONFIGS if segmentation else RFDETR_CONFIGS
        if config not in configs:
            raise ValueError(f"Invalid RF-DETR size: {config}. Must be one of {sorted(configs)}")

        self.config_name = config
        self.config = configs[config]
        self.nb_classes = nb_classes
        self.segmentation = segmentation
        self.args = _make_args(
            self.config,
            nb_classes=nb_classes,
            device=device,
            segmentation=segmentation,
        )

        self.resolution = self.config.resolution
        self.hidden_dim = self.config.hidden_dim
        self.num_queries = self.config.num_queries
        self.num_select = self.config.num_select
        self.patch_size = self.config.patch_size
        self.num_windows = self.config.num_windows

        self.model = build_model(self.args)
        self.postprocess = PostProcess(num_select=self.num_select)

    def forward(self, x: torch.Tensor, targets=None):
        return self.model(x, targets=targets)

    def build_criterion_and_postprocess(self):
        return build_criterion_and_postprocessors(self.args)

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        state_dict = _unwrap_state_dict(state_dict)

        class_bias = state_dict.get("class_embed.bias")
        if class_bias is not None and class_bias.shape[0] != self.model.class_embed.bias.shape[0]:
            self.model.reinitialize_detection_head(int(class_bias.shape[0]))
            self.nb_classes = int(class_bias.shape[0]) - 1
            self.args.num_classes = self.nb_classes

        desired_queries = self.args.num_queries * self.args.group_detr
        for key in ("refpoint_embed.weight", "query_feat.weight"):
            if key in state_dict:
                state_dict[key] = _resize_query_param(state_dict[key], desired_queries)

        interpolate_position_embeddings(state_dict, self.args.positional_encoding_size)
        return self.model.load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)


class RFDETRExportWrapper(nn.Module):
    """Export-facing wrapper that returns RF-DETR tensors as a stable tuple."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.model if isinstance(model, LibreRFDETRModel) else model
        if hasattr(self.model, "export") and not getattr(self.model, "_export", False):
            self.model.export()

    def forward(self, x: torch.Tensor):
        output = self.model(x)
        if isinstance(output, tuple):
            return output
        if "pred_masks" in output:
            return output["pred_boxes"], output["pred_logits"], output["pred_masks"]
        return output["pred_boxes"], output["pred_logits"]


def create_rfdetr_model(
    config: str = "s",
    nb_classes: int = 80,
    device: str = "cpu",
    segmentation: bool = False,
) -> LibreRFDETRModel:
    return LibreRFDETRModel(
        config=config,
        nb_classes=nb_classes,
        device=device,
        segmentation=segmentation,
    )


__all__ = [
    "LibreRFDETRModel",
    "RFDETRExportWrapper",
    "RFDETR_CONFIGS",
    "RFDETR_SEG_CONFIGS",
    "RFDETRSizeConfig",
    "LWDETR",
    "MLP",
    "PostProcess",
    "create_rfdetr_model",
    "interpolate_position_embeddings",
]
