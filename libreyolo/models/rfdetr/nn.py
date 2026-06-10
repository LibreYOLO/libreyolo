"""Native RF-DETR network assembly for LibreYOLO."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .backbone import build_backbone
from .lwdetr import LWDETR, MLP, PostProcess, build_criterion_and_postprocessors, build_model
from .tensors import NestedTensor


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
    "x": RFDETRSizeConfig(
        patch_size=12,
        num_windows=2,
        dec_layers=6,
        resolution=624,
        positional_encoding_size=52,
        num_queries=300,
        num_select=300,
        pretrain_weights="rf-detr-seg-xlarge.pt",
        segmentation_head=True,
    ),
    "xx": RFDETRSizeConfig(
        patch_size=12,
        num_windows=2,
        dec_layers=6,
        resolution=768,
        positional_encoding_size=64,
        num_queries=300,
        num_select=300,
        pretrain_weights="rf-detr-seg-xxlarge.pt",
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
    pose: bool = False,
    obb: bool = False,
    num_keypoints: int = 17,
    oks_sigmas=None,
) -> SimpleNamespace:
    cfg_values = {
        f.name: list(getattr(cfg, f.name)) if isinstance(getattr(cfg, f.name), tuple) else getattr(cfg, f.name)
        for f in fields(cfg)
    }
    cfg_values["pretrain_weights"] = "__libreyolo_no_backbone_download__"
    cfg_values["segmentation_head"] = segmentation
    cfg_values["keypoint_head"] = pose
    cfg_values["obb"] = obb
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
        angle_loss_coef=1.0,
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
        keypoint_l1_loss_coef=10.0,
        keypoint_oks_loss_coef=4.0,
        keypoint_vis_loss_coef=1.0,
        num_keypoints=int(num_keypoints),
        oks_sigmas=oks_sigmas,
        num_channels=3,
        num_classes=max(0, nb_classes - 1) if pose else nb_classes,
        position_embedding="sine",
        pretrained_encoder=None,
        rms_norm=False,
        set_cost_bbox=5.0,
        set_cost_class=2.0,
        set_cost_giou=2.0,
        set_cost_angle=0.5 if obb else 0.0,
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


def _get_arg(args: Any, name: str) -> Any:
    if args is None:
        return None
    if isinstance(args, dict):
        return args.get(name)
    return getattr(args, name, None)


def _slice_query_param_per_group(
    tensor: torch.Tensor,
    *,
    ckpt_num_queries: int,
    ckpt_group_detr: int,
    target_num_queries: int,
    target_group_detr: int,
) -> torch.Tensor:
    """Resize packed Group-DETR query rows without mixing group slots."""
    if ckpt_num_queries <= 0 or ckpt_group_detr <= 0 or target_num_queries <= 0 or target_group_detr <= 0:
        return tensor[: target_num_queries * target_group_detr]

    expected_rows = ckpt_num_queries * ckpt_group_detr
    if tensor.shape[0] != expected_rows:
        return tensor[: target_num_queries * target_group_detr]

    if ckpt_num_queries == target_num_queries and ckpt_group_detr == target_group_detr:
        return tensor

    keep_groups = min(ckpt_group_detr, target_group_detr)
    keep_queries = min(ckpt_num_queries, target_num_queries)
    pieces = [
        tensor[group_idx * ckpt_num_queries : group_idx * ckpt_num_queries + keep_queries]
        for group_idx in range(keep_groups)
    ]
    return torch.cat(pieces, dim=0)


def _resize_query_param_from_checkpoint(
    tensor: torch.Tensor,
    *,
    checkpoint_args: Any,
    target_num_queries: int,
    target_group_detr: int,
) -> torch.Tensor:
    ckpt_num_queries = _get_arg(checkpoint_args, "num_queries")
    ckpt_group_detr = _get_arg(checkpoint_args, "group_detr")
    try:
        ckpt_num_queries = int(ckpt_num_queries) if ckpt_num_queries is not None else None
        ckpt_group_detr = int(ckpt_group_detr) if ckpt_group_detr is not None else None
    except (TypeError, ValueError):
        ckpt_num_queries = None
        ckpt_group_detr = None

    if ckpt_num_queries is not None and ckpt_group_detr is not None:
        return _slice_query_param_per_group(
            tensor,
            ckpt_num_queries=ckpt_num_queries,
            ckpt_group_detr=ckpt_group_detr,
            target_num_queries=target_num_queries,
            target_group_detr=target_group_detr,
        )

    return _resize_query_param(tensor, target_num_queries * target_group_detr)


class RFDETRClassifier(nn.Module):
    """Image-classification model: RF-DETR's DINOv2 backbone + linear head.

    Reuses the same DINOv2 encoder + multi-scale projector that powers RF-DETR
    detection, then collapses the projector's feature map with global average
    pooling and a linear classifier. The backbone is built with patch_size=14
    and a 518px implied resolution so the standard pretrained DINOv2 weights
    load cleanly (a randomly initialized ViT would barely train on small
    datasets); it falls back to random init if the weights cannot be fetched.
    """

    # DINOv2 native settings that keep ``load_dinov2_weights`` enabled.
    _PATCH_SIZE = 14
    _POS_ENC_SIZE = 37  # 37 * 14 == 518, DINOv2's pretrained image_size
    _NUM_WINDOWS = 1

    def __init__(
        self,
        config: str = "s",
        nb_classes: int = 1000,
        device: str = "cpu",
        dropout: float = 0.2,
    ):
        super().__init__()
        if config not in RFDETR_CONFIGS:
            raise ValueError(
                f"Invalid RF-DETR size: {config}. Must be one of {sorted(RFDETR_CONFIGS)}"
            )
        cfg = RFDETR_CONFIGS[config]
        self.config_name = config
        self.nb_classes = nb_classes
        self.hidden_dim = cfg.hidden_dim
        # Surfaced for trainer/transforms that introspect these attributes.
        self.patch_size = self._PATCH_SIZE
        self.num_windows = self._NUM_WINDOWS
        self.resolution = 224

        joiner = self._build_backbone(cfg, device)
        # joiner = Sequential(Backbone, PositionEmbedding); classification only
        # needs the Backbone (DINOv2 encoder + projector) — drop the position
        # encoding so its parameters are not carried around unused.
        self.backbone = joiner[0]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(self.hidden_dim, nb_classes)

    def _build_backbone(self, cfg: RFDETRSizeConfig, device: str):
        kwargs = dict(
            encoder=cfg.encoder,
            vit_encoder_num_layers=12,
            pretrained_encoder=None,
            window_block_indexes=None,
            drop_path=0.0,
            out_channels=cfg.hidden_dim,
            out_feature_indexes=list(cfg.out_feature_indexes),
            projector_scale=list(cfg.projector_scale),
            use_cls_token=False,
            hidden_dim=cfg.hidden_dim,
            position_embedding="sine",
            freeze_encoder=False,
            layer_norm=True,
            target_shape=(self.resolution, self.resolution),
            rms_norm=False,
            backbone_lora=False,
            force_no_pretrain=False,
            gradient_checkpointing=False,
            patch_size=self._PATCH_SIZE,
            num_windows=self._NUM_WINDOWS,
            positional_encoding_size=self._POS_ENC_SIZE,
        )
        try:
            return build_backbone(load_dinov2_weights=True, **kwargs)
        except Exception as exc:  # pragma: no cover - offline / hub failure
            logger = __import__("logging").getLogger(__name__)
            logger.warning(
                "Could not load pretrained DINOv2 weights (%s); building the "
                "classification backbone from random init.",
                exc,
            )
            return build_backbone(load_dinov2_weights=False, **kwargs)

    def forward(self, x: torch.Tensor, targets=None):
        b, _, h, w = x.shape
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=x.device)
        feats = self.backbone(NestedTensor(x, mask))
        # Backbone returns a list of NestedTensor pyramid levels; the last is
        # the coarsest. Global-average-pool it to a per-image embedding.
        feat = feats[-1].tensors
        pooled = self.pool(feat).flatten(1)
        pooled = self.drop(pooled)
        logits = self.linear(pooled)
        if self.training and targets is not None:
            loss = F.cross_entropy(logits, targets.long())
            return {"total_loss": loss, "cls": loss}
        return logits


class RFDETRSemanticSegmenter(nn.Module):
    """Dense semantic-segmentation model: RF-DETR's DINOv2 backbone + decoder.

    Reuses the DINOv2 encoder + multi-scale projector that powers RF-DETR
    detection (no query decoder, no Hungarian matching) and fuses the
    projector pyramid into per-pixel class logits: a 1x1 lateral per level,
    resize-and-sum onto the finest level, a 3x3 smoothing conv, then a 1x1
    projection upsampled to the input resolution. The backbone is built with
    DINOv2-native patch/positional settings so pretrained weights load
    cleanly; it falls back to random init if the weights cannot be fetched.

    Inputs are expected as RGB floats in ``[0, 1]``; ImageNet normalization
    is applied inside ``forward`` so training tensors and inference tensors
    share one contract.
    """

    _PATCH_SIZE = 14
    _POS_ENC_SIZE = 37  # 37 * 14 == 518, DINOv2's pretrained image_size
    _NUM_WINDOWS = 1

    IGNORE_INDEX = 255

    def __init__(
        self,
        config: str = "s",
        nb_classes: int = 19,
        device: str = "cpu",
        dropout: float = 0.1,
    ):
        super().__init__()
        if config not in RFDETR_CONFIGS:
            raise ValueError(
                f"Invalid RF-DETR size: {config}. Must be one of {sorted(RFDETR_CONFIGS)}"
            )
        cfg = RFDETR_CONFIGS[config]
        self.config_name = config
        self.nb_classes = nb_classes
        self.hidden_dim = cfg.hidden_dim
        self.patch_size = self._PATCH_SIZE
        self.num_windows = self._NUM_WINDOWS
        self.resolution = self._POS_ENC_SIZE * self._PATCH_SIZE  # 518

        joiner = self._build_backbone(cfg, device)
        # joiner = Sequential(Backbone, PositionEmbedding); the dense decoder
        # only needs the Backbone (DINOv2 encoder + projector).
        self.backbone = joiner[0]

        num_levels = len(cfg.projector_scale)
        self.laterals = nn.ModuleList(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1) for _ in range(num_levels)
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(32, self.hidden_dim),
            nn.GELU(),
        )
        self.drop = nn.Dropout2d(p=dropout)
        self.predict = nn.Conv2d(self.hidden_dim, nb_classes, 1)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    def _build_backbone(self, cfg: RFDETRSizeConfig, device: str):
        kwargs = dict(
            encoder=cfg.encoder,
            vit_encoder_num_layers=12,
            pretrained_encoder=None,
            window_block_indexes=None,
            drop_path=0.0,
            out_channels=cfg.hidden_dim,
            out_feature_indexes=list(cfg.out_feature_indexes),
            projector_scale=list(cfg.projector_scale),
            use_cls_token=False,
            hidden_dim=cfg.hidden_dim,
            position_embedding="sine",
            freeze_encoder=False,
            layer_norm=True,
            target_shape=(self.resolution, self.resolution),
            rms_norm=False,
            backbone_lora=False,
            force_no_pretrain=False,
            gradient_checkpointing=False,
            patch_size=self._PATCH_SIZE,
            num_windows=self._NUM_WINDOWS,
            positional_encoding_size=self._POS_ENC_SIZE,
        )
        try:
            return build_backbone(load_dinov2_weights=True, **kwargs)
        except Exception as exc:  # pragma: no cover - offline / hub failure
            logger = __import__("logging").getLogger(__name__)
            logger.warning(
                "Could not load pretrained DINOv2 weights (%s); building the "
                "semantic backbone from random init.",
                exc,
            )
            return build_backbone(load_dinov2_weights=False, **kwargs)

    def forward(self, x: torch.Tensor, targets=None):
        b, _, h, w = x.shape
        x = (x - self.pixel_mean) / self.pixel_std
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=x.device)
        feats = self.backbone(NestedTensor(x, mask))

        # Fuse the projector pyramid onto its finest level.
        finest = feats[0].tensors
        fused = self.laterals[0](finest)
        for lateral, level in zip(self.laterals[1:], feats[1:]):
            fused = fused + F.interpolate(
                lateral(level.tensors), size=finest.shape[-2:], mode="bilinear",
                align_corners=False,
            )
        fused = self.smooth(fused)
        logits = self.predict(self.drop(fused))

        if self.training and targets is not None:
            logits = F.interpolate(
                logits, size=targets.shape[-2:], mode="bilinear", align_corners=False
            )
            targets = targets.long()
            if bool((targets != self.IGNORE_INDEX).any()):
                loss = F.cross_entropy(
                    logits, targets, ignore_index=self.IGNORE_INDEX
                )
            else:
                # cross_entropy returns NaN when every pixel is ignored; emit
                # a graph-connected zero so the optimizer step stays sane.
                loss = logits.sum() * 0.0
            return {"total_loss": loss, "sem": loss}

        return F.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )


class LibreRFDETRModel(nn.Module):
    """RF-DETR model built from LibreYOLO-local RF-DETR modules."""

    def __init__(
        self,
        config: str = "s",
        nb_classes: int = 80,
        device: str = "cpu",
        segmentation: bool = False,
        pose: bool = False,
        classification: bool = False,
        obb: bool = False,
        semantic: bool = False,
        num_keypoints: int = 17,
        oks_sigmas=None,
    ):
        super().__init__()

        if sum(bool(x) for x in (segmentation, pose, classification, obb, semantic)) > 1:
            raise ValueError("RF-DETR can enable only one task head at a time")

        self.classification = classification
        self.semantic = semantic
        if classification:
            # Backbone-only classification path: no detection decoder/criterion.
            self.config_name = config
            self.config = RFDETR_CONFIGS[config]
            self.nb_classes = nb_classes
            self.segmentation = False
            self.classifier = RFDETRClassifier(
                config=config, nb_classes=nb_classes, device=device
            )
            self.resolution = self.classifier.resolution
            self.hidden_dim = self.classifier.hidden_dim
            self.patch_size = self.classifier.patch_size
            self.num_windows = self.classifier.num_windows
            self.model = None
            self.postprocess = None
            return

        if semantic:
            # Backbone-only dense path: no detection decoder/criterion.
            self.config_name = config
            self.config = RFDETR_CONFIGS[config]
            self.nb_classes = nb_classes
            self.segmentation = False
            self.segmenter = RFDETRSemanticSegmenter(
                config=config, nb_classes=nb_classes, device=device
            )
            self.resolution = self.segmenter.resolution
            self.hidden_dim = self.segmenter.hidden_dim
            self.patch_size = self.segmenter.patch_size
            self.num_windows = self.segmenter.num_windows
            self.model = None
            self.postprocess = None
            return

        configs = RFDETR_SEG_CONFIGS if segmentation else RFDETR_CONFIGS
        if pose or obb:
            configs = RFDETR_CONFIGS
        if config not in configs:
            raise ValueError(f"Invalid RF-DETR size: {config}. Must be one of {sorted(configs)}")

        self.config_name = config
        self.config = configs[config]
        self.nb_classes = nb_classes
        self.segmentation = segmentation
        self.pose = pose
        self.obb = obb
        self.num_keypoints = int(num_keypoints)
        self.args = _make_args(
            self.config,
            nb_classes=nb_classes,
            device=device,
            segmentation=segmentation,
            pose=pose,
            obb=obb,
            num_keypoints=num_keypoints,
            oks_sigmas=oks_sigmas,
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
        if self.classification:
            return self.classifier(x, targets=targets)
        if self.semantic:
            return self.segmenter(x, targets=targets)
        return self.model(x, targets=targets)

    def build_criterion_and_postprocess(self):
        return build_criterion_and_postprocessors(self.args)

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        if self.classification:
            return self.classifier.load_state_dict(
                _unwrap_state_dict(state_dict), strict=strict
            )
        if self.semantic:
            return self.segmenter.load_state_dict(
                _unwrap_state_dict(state_dict), strict=strict
            )

        checkpoint_args = state_dict.get("args") if isinstance(state_dict, dict) else None
        state_dict = _unwrap_state_dict(state_dict)

        class_bias = state_dict.get("class_embed.bias")
        if class_bias is not None and class_bias.shape[0] != self.model.class_embed.bias.shape[0]:
            out_features = int(class_bias.shape[0])
            self.model.reinitialize_detection_head(out_features)
            if self.pose:
                self.nb_classes = out_features
                self.args.num_classes = max(0, out_features - 1)
            else:
                self.nb_classes = out_features - 1
                self.args.num_classes = self.nb_classes
        keypoint_weight = state_dict.get("keypoint_head.layers.2.weight")
        if keypoint_weight is not None and self.model.keypoint_head is not None:
            ckpt_k = int(keypoint_weight.shape[0]) // 3
            if ckpt_k != self.num_keypoints:
                self.model.reinitialize_keypoint_head(ckpt_k)
                self.num_keypoints = ckpt_k
                self.args.num_keypoints = ckpt_k

        for key in ("refpoint_embed.weight", "query_feat.weight"):
            if key in state_dict:
                state_dict[key] = _resize_query_param_from_checkpoint(
                    state_dict[key],
                    checkpoint_args=checkpoint_args,
                    target_num_queries=self.args.num_queries,
                    target_group_detr=self.args.group_detr,
                )

        interpolate_position_embeddings(state_dict, self.args.positional_encoding_size)
        return self.model.load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args, **kwargs):
        if self.classification:
            return self.classifier.state_dict(*args, **kwargs)
        if self.semantic:
            return self.segmenter.state_dict(*args, **kwargs)
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
        if "pred_keypoints" in output:
            return output["pred_boxes"], output["pred_logits"], output["pred_keypoints"]
        if "pred_angles" in output:
            return output["pred_boxes"], output["pred_logits"], output["pred_angles"]
        return output["pred_boxes"], output["pred_logits"]


def create_rfdetr_model(
    config: str = "s",
    nb_classes: int = 80,
    device: str = "cpu",
    segmentation: bool = False,
    pose: bool = False,
    obb: bool = False,
    num_keypoints: int = 17,
    oks_sigmas=None,
) -> LibreRFDETRModel:
    return LibreRFDETRModel(
        config=config,
        nb_classes=nb_classes,
        device=device,
        segmentation=segmentation,
        pose=pose,
        obb=obb,
        num_keypoints=num_keypoints,
        oks_sigmas=oks_sigmas,
    )


__all__ = [
    "LibreRFDETRModel",
    "RFDETRClassifier",
    "RFDETRSemanticSegmenter",
    "RFDETRExportWrapper",
    "RFDETR_CONFIGS",
    "RFDETR_SEG_CONFIGS",
    "RFDETRSizeConfig",
    "LWDETR",
    "MLP",
    "PostProcess",
    "create_rfdetr_model",
    "interpolate_position_embeddings",
    "_slice_query_param_per_group",
]
