"""LibreRTDETRv2 — RT-DETRv2 detectors."""

from __future__ import annotations

import os
import re
from functools import wraps
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ...postprocess.rtdetr import postprocess_obb
from ...utils.image_loader import ImageInput, ImageLoader
from ...validation.preprocessors import (
    RTDETRv2OBBValPreprocessor,
    RTDETRv2ValPreprocessor,
)
from ..rtdetr.model import LibreRTDETR, RTDETR_CONFIGS
from .nn import RTDETRv2Model


RTDETRV2_OBB_CONFIGS: Dict[str, Dict[str, Any]] = {
    "n": {
        "backbone": "B0",
        "use_lab": True,
        "return_idx": (2, 3),
        "in_channels": (512, 1024),
        "feat_strides": (16, 32),
        "hidden_dim": 128,
        "dim_feedforward": 512,
        "expansion": 0.34,
        "depth_mult": 0.50,
        "use_encoder_idx": (1,),
        "num_layers": 3,
        "num_points": (6, 6),
    },
    "s": {
        "backbone": "B0",
        "use_lab": True,
        "return_idx": (1, 2, 3),
        "in_channels": (256, 512, 1024),
        "feat_strides": (8, 16, 32),
        "hidden_dim": 224,
        "dim_feedforward": 1024,
        "expansion": 0.50,
        "depth_mult": 0.34,
        "use_encoder_idx": (2,),
        "num_layers": 3,
        "num_points": (4, 4, 4),
    },
    "m": {
        "backbone": "B2",
        "use_lab": True,
        "return_idx": (1, 2, 3),
        "in_channels": (384, 768, 1536),
        "feat_strides": (8, 16, 32),
        "hidden_dim": 256,
        "dim_feedforward": 1024,
        "expansion": 1.0,
        "depth_mult": 0.67,
        "use_encoder_idx": (2,),
        "num_layers": 3,
        "num_points": (4, 4, 4),
    },
    "l": {
        "backbone": "B4",
        "use_lab": False,
        "return_idx": (1, 2, 3),
        "in_channels": (512, 1024, 2048),
        "feat_strides": (8, 16, 32),
        "hidden_dim": 256,
        "dim_feedforward": 1024,
        "expansion": 1.0,
        "depth_mult": 0.67,
        "use_encoder_idx": (2,),
        "num_layers": 4,
        "num_points": (4, 4, 4),
    },
    "x": {
        "backbone": "B5",
        "use_lab": False,
        "return_idx": (1, 2, 3),
        "in_channels": (512, 1024, 2048),
        "feat_strides": (8, 16, 32),
        "hidden_dim": 384,
        "dim_feedforward": 2048,
        "expansion": 1.0,
        "depth_mult": 0.67,
        "use_encoder_idx": (2,),
        "num_layers": 4,
        "num_points": (4, 4, 4),
    },
}

RTDETRV2_OBB_NAMES = {
    0: "plane",
    1: "baseball-diamond",
    2: "bridge",
    3: "ground-track-field",
    4: "small-vehicle",
    5: "large-vehicle",
    6: "ship",
    7: "tennis-court",
    8: "basketball-court",
    9: "storage-tank",
    10: "soccer-ball-field",
    11: "roundabout",
    12: "harbor",
    13: "swimming-pool",
    14: "helicopter",
}


class LibreRTDETRv2(LibreRTDETR):
    FAMILY = "rtdetrv2"
    FILENAME_PREFIX = "LibreRTDETRv2"
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    INPUT_SIZES = {"r18": 640, "r34": 640, "r50": 640, "r50m": 640, "r101": 640}
    OBB_INPUT_SIZES = {"n": 1024, "s": 1024, "m": 1024, "l": 1024, "x": 1024}
    SUPPORTED_TASKS = ("detect", "obb")
    DEFAULT_TASK = "detect"
    TASK_INPUT_SIZES = {"detect": INPUT_SIZES, "obb": OBB_INPUT_SIZES}
    val_preprocessor_class = RTDETRv2ValPreprocessor

    @classmethod
    def is_obb_state_dict(cls, weights_dict: dict) -> bool:
        query_pos = weights_dict.get("decoder.query_pos_head.layers.0.weight")
        enc_bbox = weights_dict.get("decoder.enc_bbox_head.layers.2.weight")
        has_hgnet = any(k.startswith("backbone.stages.") for k in weights_dict)
        has_v2_sampling = any("cross_attn.num_points_scale" in k for k in weights_dict)
        return bool(
            has_hgnet
            and has_v2_sampling
            and query_pos is not None
            and enc_bbox is not None
            and int(query_pos.shape[1]) == 5
            and int(enc_bbox.shape[0]) == 5
        )

    @classmethod
    def detect_checkpoint_task(cls, weights_dict: dict) -> Optional[str]:
        return "obb" if cls.is_obb_state_dict(weights_dict) else None

    @classmethod
    def default_checkpoint_names(
        cls, nc: int, task: Optional[str] = None
    ) -> Optional[Dict[int, str]]:
        return dict(RTDETRV2_OBB_NAMES) if nc == 15 else None

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        if cls.is_obb_state_dict(weights_dict):
            return True
        # State-dict shape is identical to v1's, so we delegate to v1's check.
        # Disambiguation against v1 in the factory happens via:
        #   (1) the ``model_family`` metadata gate (converted ckpts);
        #   (2) the ``rtdetrv2_`` filename hint (raw upstream ckpts);
        #   (3) registry order — ``LibreRTDETR`` is imported BEFORE
        #       ``LibreRTDETRv2`` so that ckpts lacking both signals route
        #       to v1 by default. v1 cannot be silently shadowed.
        return LibreRTDETR.can_load(weights_dict)

    @classmethod
    def convert_upstream_state_dict(cls, weights_dict: dict) -> Optional[dict]:
        """Remap upstream RT-DETRv2 ResNet checkpoints to the v2 key layout.

        The discrete-sampling buffer (``num_points_scale``) is unique to v2
        and is what separates these from v1 claims. v2 HGNetv2 checkpoints
        are intentionally not claimed here — LibreYOLO ships those under the
        v1 family (LibreRTDETR l/x).
        """
        from ..rtdetr.convert import (
            V2_SAMPLING_FRAGMENT,
            convert_to_v2,
            has_upstream_input_proj_keys,
        )

        if not has_upstream_input_proj_keys(weights_dict):
            return None
        if cls.is_obb_state_dict(weights_dict):
            return convert_to_v2(weights_dict)
        if not any(k.startswith("backbone.res_layers") for k in weights_dict):
            return None
        if not any(V2_SAMPLING_FRAGMENT in k for k in weights_dict):
            return None
        return convert_to_v2(weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        if cls.is_obb_state_dict(weights_dict):
            encoder_weight = weights_dict.get("encoder.input_proj.0.conv.weight")
            if encoder_weight is None:
                encoder_weight = weights_dict.get("encoder.input_proj.0.0.weight")
            if encoder_weight is None:
                return None
            hidden_dim = int(encoder_weight.shape[0])
            if hidden_dim == 128:
                return "n"
            if hidden_dim == 224:
                return "s"
            if hidden_dim == 384:
                return "x"
            if hidden_dim == 256:
                stem_weight = weights_dict.get("backbone.stem.stem1.conv.weight")
                if stem_weight is not None:
                    return "m" if int(stem_weight.shape[0]) == 24 else "l"
            return None
        return super().detect_size(weights_dict)

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        basename = os.path.basename(filename).lower()
        pattern = cls._filename_regex()
        match = pattern.search(basename) if pattern is not None else None
        if match:
            return match.group("size")
        m = re.search(r"rtdetrv2_r(\d+)vd(_m)?_", basename)
        if m:
            depth, m_suffix = m.group(1), m.group(2)
            return f"r{depth}m" if m_suffix else f"r{depth}"
        return None

    @classmethod
    def _get_trainer_class(cls):
        from .trainer import RTDETRv2Trainer

        return RTDETRv2Trainer

    def _init_model(self) -> nn.Module:
        if self.task == "obb":
            from ..dfine.backbone import HGNetv2

            if self.size not in RTDETRV2_OBB_CONFIGS:
                raise ValueError(f"Unknown RT-DETRv2 OBB size: {self.size!r}")
            cfg = RTDETRV2_OBB_CONFIGS[self.size]
            backbone = HGNetv2(
                name=cfg["backbone"],
                use_lab=cfg["use_lab"],
                return_idx=cfg["return_idx"],
                freeze_stem_only=False,
                freeze_at=-1,
                freeze_norm=False,
                pretrained=False,
            )
            return RTDETRv2Model(
                num_classes=self.nb_classes,
                backbone=backbone,
                hidden_dim=cfg["hidden_dim"],
                dim_feedforward=cfg["dim_feedforward"],
                expansion=cfg["expansion"],
                encoder_depth_mult=cfg["depth_mult"],
                encoder_in_channels=cfg["in_channels"],
                encoder_use_encoder_idx=cfg["use_encoder_idx"],
                decoder_hidden_dim=cfg["hidden_dim"],
                decoder_dim_feedforward=1024,
                num_decoder_layers=cfg["num_layers"],
                num_decoder_points=list(cfg["num_points"]),
                feat_strides=cfg["feat_strides"],
                num_levels=len(cfg["feat_strides"]),
                eval_idx=-1,
                eval_spatial_size=(self.input_size, self.input_size),
                obb=True,
            )
        if self.size not in RTDETR_CONFIGS:
            raise ValueError(f"Unknown RT-DETRv2 size: {self.size!r}")
        cfg: Dict[str, Any] = RTDETR_CONFIGS[self.size]
        # v2 ResNet sizes only — HGNetv2 backbones are skipped at this layer
        # (v1 already ships HGNetv2-l/x; v2's HGNetv2 numbers are within ~0.1
        # AP and not worth the duplicate weights).
        if cfg.get("backbone_type") == "hgnetv2":
            raise ValueError(
                f"LibreRTDETRv2 size {self.size!r} uses an HGNetv2 backbone; "
                f"use LibreRTDETR for HGNetv2 variants."
            )
        scratch = self._is_scratch_build()
        return RTDETRv2Model(
            num_classes=self.nb_classes,
            backbone_depth=cfg["backbone_depth"],
            backbone_freeze_at=-1 if scratch else cfg["backbone_freeze_at"],
            backbone_freeze_norm=False if scratch else cfg["backbone_freeze_norm"],
            backbone_pretrained=False,
            hidden_dim=cfg["encoder_hidden_dim"],
            dim_feedforward=cfg["encoder_dim_feedforward"],
            expansion=cfg["encoder_expansion"],
            decoder_hidden_dim=cfg["decoder_hidden_dim"],
            decoder_dim_feedforward=cfg.get("decoder_dim_feedforward", 1024),
            num_decoder_layers=cfg["num_decoder_layers"],
            eval_idx=cfg["eval_idx"],
            eval_spatial_size=(self.input_size, self.input_size),
        )

    def _strict_loading(self) -> bool:
        return self.task == "obb" or super()._strict_loading()

    def _validate_loaded_state_dict_for_task(
        self,
        state_dict: dict,
        checkpoint: dict | None = None,
    ) -> None:
        is_obb = self.is_obb_state_dict(state_dict)
        if self.task == "obb" and not is_obb:
            raise ValueError("RT-DETRv2 OBB requires a five-coordinate OBB checkpoint")
        if self.task == "detect" and is_obb:
            raise ValueError("RT-DETRv2 OBB checkpoints must be loaded with task='obb'")

    @wraps(LibreRTDETR.train)
    def train(self, *args, **kwargs):
        if getattr(self, "task", "detect") == "obb":
            raise NotImplementedError(
                "RT-DETRv2 OBB is inference-only in LibreYOLO; training is not implemented"
            )
        return super().train(*args, **kwargs)

    def _get_val_preprocessor(self, img_size: int | None = None):
        if self.task != "obb":
            return super()._get_val_preprocessor(img_size=img_size)
        if img_size is None:
            img_size = self._get_input_size()
        return RTDETRv2OBBValPreprocessor(img_size=(img_size, img_size))

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ):
        if self.task != "obb":
            return super()._preprocess(image, color_format, input_size)

        effective_size = input_size if input_size is not None else self.input_size
        if isinstance(effective_size, int):
            target_h = target_w = int(effective_size)
        else:
            target_h, target_w = int(effective_size[0]), int(effective_size[1])

        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (target_w, target_h), color=0)
        canvas.paste(resized, (0, 0))

        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)
        return tensor, img, (orig_w, orig_h), scale

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        if self.task != "obb":
            return super()._postprocess(
                output,
                conf_thres,
                iou_thres,
                original_size,
                max_det=max_det,
                **kwargs,
            )
        return postprocess_obb(
            output,
            conf_thres,
            iou_thres,
            original_size,
            max_det=max_det,
            input_size=kwargs.get("input_size", self.input_size),
        )
