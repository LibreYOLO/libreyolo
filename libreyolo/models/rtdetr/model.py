"""LibreYOLORTDETR implementation for LibreYOLO."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ..base import BaseModel
from ...utils.image_loader import ImageInput
from .nn import RTDETRModel
from .config import RTDETRConfig
from ...validation.preprocessors import RTDETRValPreprocessor

# Single source of truth for training defaults
_TRAIN_DEFAULTS = RTDETRConfig()


# Model configs — derived from official RT-DETR YAML configs
RTDETR_CONFIGS = {
    "r18": {
        "backbone_depth": 18,
        "backbone_freeze_at": 0,
        "backbone_freeze_norm": False,
        "backbone_pretrained": True,
        "encoder_hidden_dim": 256,
        "encoder_dim_feedforward": 1024,
        "encoder_expansion": 0.5,
        "decoder_hidden_dim": 256,
        "num_decoder_layers": 3,
        "eval_idx": -1,
    },
    "r34": {
        "backbone_depth": 34,
        "backbone_freeze_at": 0,
        "backbone_freeze_norm": False,
        "backbone_pretrained": True,
        "encoder_hidden_dim": 256,
        "encoder_dim_feedforward": 1024,
        "encoder_expansion": 0.5,
        "decoder_hidden_dim": 256,
        "num_decoder_layers": 4,
        "eval_idx": -1,
    },
    "r50": {
        "backbone_depth": 50,
        "backbone_freeze_at": 0,
        "backbone_freeze_norm": True,
        "backbone_pretrained": True,
        "encoder_hidden_dim": 256,
        "encoder_dim_feedforward": 1024,
        "encoder_expansion": 1.0,
        "decoder_hidden_dim": 256,
        "num_decoder_layers": 6,
        "eval_idx": -1,
    },
    "r50m": {
        "backbone_depth": 50,
        "backbone_freeze_at": 0,
        "backbone_freeze_norm": True,
        "backbone_pretrained": True,
        "encoder_hidden_dim": 256,
        "encoder_dim_feedforward": 1024,
        "encoder_expansion": 0.5,
        "decoder_hidden_dim": 256,
        "num_decoder_layers": 6,
        "eval_idx": 2,
    },
    "r101": {
        "backbone_depth": 101,
        "backbone_freeze_at": 0,
        "backbone_freeze_norm": True,
        "backbone_pretrained": True,
        "encoder_hidden_dim": 384,
        "encoder_dim_feedforward": 2048,
        "encoder_expansion": 1.0,
        "decoder_hidden_dim": 256,
        "decoder_dim_feedforward": 1024,
        "num_decoder_layers": 6,
        "eval_idx": -1,
    },
    "l": {
        "backbone_type": "hgnetv2",
        "backbone_arch": "L",
        "backbone_freeze_at": 0,
        "backbone_freeze_norm": True,
        "backbone_pretrained": True,
        "encoder_hidden_dim": 256,
        "encoder_dim_feedforward": 1024,
        "encoder_expansion": 1.0,
        "decoder_hidden_dim": 256,
        "num_decoder_layers": 6,
        "eval_idx": -1,
    },
    "x": {
        "backbone_type": "hgnetv2",
        "backbone_arch": "X",
        "backbone_freeze_at": 0,
        "backbone_freeze_norm": True,
        "backbone_pretrained": True,
        "encoder_hidden_dim": 384,
        "encoder_dim_feedforward": 2048,
        "encoder_expansion": 1.0,
        "decoder_hidden_dim": 256,
        "decoder_dim_feedforward": 1024,
        "num_decoder_layers": 6,
        "eval_idx": -1,
    },
}


class LibreYOLORTDETR(BaseModel):
    """RT-DETR model for object detection.

    RT-DETR is a real-time Detection Transformer using ResNet backbone with
    hybrid encoder and multi-scale deformable attention decoder.

    Args:
        model_path: Path to weights, pre-loaded state_dict, or None for fresh model.
        size: Model size variant ("r18", "r34", "r50", "r50m", "r101").
        nb_classes: Number of classes (default: 80 for COCO).
        device: Device for inference.

    Example::

        >>> model = LibreYOLORTDETR(size="r50")
        >>> detections = model.predict("path/to/image.jpg")
    """

    # Class-level metadata
    FAMILY = "rtdetr"
    FILENAME_PREFIX = "LibreRTDETR"
    INPUT_SIZES = {
        "r18": 640,
        "r34": 640,
        "r50": 640,
        "r50m": 640,
        "r101": 640,
        "l": 640,
        "x": 640,
    }
    TRAIN_CONFIG = RTDETRConfig
    val_preprocessor_class = RTDETRValPreprocessor

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Detect RT-DETR-specific keys in the state dict.

        RT-DETR and D-FINE share several decoder head names, so require the
        RT-DETR decoder input projection path that D-FINE does not have.
        """
        keys = set(weights_dict.keys())
        has_backbone = any(
            k.startswith(("backbone.res_layers", "backbone.stages")) for k in keys
        )
        has_encoder_input_proj = any(k.startswith("encoder.input_proj") for k in keys)
        has_decoder_input_proj = any(k.startswith("decoder.input_proj") for k in keys)
        has_decoder_head = any(
            k.startswith(("decoder.dec_score_head", "decoder.enc_score_head"))
            for k in keys
        )
        return (
            has_backbone
            and has_encoder_input_proj
            and has_decoder_input_proj
            and has_decoder_head
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Detect model size from weights."""
        keys = set(weights_dict.keys())

        # HGNetv2 backbone (l/x): distinguish by encoder hidden_dim (256 vs 384).
        if any(k.startswith("backbone.stages") for k in keys):
            for k, v in weights_dict.items():
                if k == "encoder.input_proj.0.0.weight":
                    return "x" if v.shape[0] == 384 else "l"
            return "l"

        enc0 = weights_dict.get("encoder.input_proj.0.0.weight")
        if enc0 is not None and int(enc0.shape[0]) == 384:
            return "r101"

        # PResNet bottleneck checkpoints use branch2c, not a "conv3" name.
        has_bottleneck = any(
            k.startswith("backbone.res_layers.") and ".branch2c." in k for k in keys
        )

        if not has_bottleneck:
            # r18 or r34 — check by layer count
            # r18: [2,2,2,2], r34: [3,4,6,3]
            # Count unique layer indices in backbone.res_layers.0
            stage0_keys = [
                k
                for k in weights_dict.keys()
                if k.startswith("backbone.res_layers.0.blocks.")
            ]
            block_indices = set()
            for k in stage0_keys:
                parts = k.split(".")
                if len(parts) > 5:
                    try:
                        block_indices.add(int(parts[4]))
                    except ValueError:
                        pass
            if len(block_indices) <= 2:
                return "r18"
            else:
                return "r34"

        stage2_blocks = {
            int(k.split(".")[4])
            for k in keys
            if k.startswith("backbone.res_layers.2.blocks.")
            and len(k.split(".")) > 5
            and k.split(".")[4].isdigit()
        }
        if len(stage2_blocks) > 6:
            return "r101"

        # r50m uses encoder_expansion=0.5, so the first CSPRepLayer projects to
        # 128 channels; r50 uses expansion=1.0 and keeps 256 channels.
        fpn_key = "encoder.fpn_blocks.0.conv1.conv.weight"
        if fpn_key in weights_dict and int(weights_dict[fpn_key].shape[0]) == 128:
            return "r50m"
        return "r50"

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> int:
        """Detect number of classes from the classification head."""
        # The classification head is decoder.dec_score_head.{last_layer}.bias
        # Find the last dec_score_head layer
        score_head_keys = [
            k
            for k in weights_dict.keys()
            if "dec_score_head" in k and k.endswith(".bias")
        ]
        if score_head_keys:
            # Get the last layer's bias shape
            last_key = sorted(score_head_keys)[-1]
            return weights_dict[last_key].shape[0]
        return 80  # default COCO

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        """Override to handle multi-char size codes like r18, r34, r50, r50m, r101."""
        sizes = list(cls.INPUT_SIZES.keys())
        # Sort by length descending to match r50m before r50
        sizes_sorted = sorted(sizes, key=len, reverse=True)
        basename = os.path.basename(filename)
        for size in sizes_sorted:
            pattern = rf"{cls.FILENAME_PREFIX}[-_]?{re.escape(size)}[^a-z0-9]"
            if re.search(pattern, basename):
                return size
            # Also try just the size code anywhere in the filename
            if (
                f"-{size}" in basename
                or f"_{size}" in basename
                or basename.startswith(f"{size}")
            ):
                return size
        return None

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        model_path=None,
        size: str = "r50",
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )

        if isinstance(model_path, str):
            self._load_weights(model_path)

    # =========================================================================
    # Model lifecycle
    # =========================================================================

    def _init_model(self) -> nn.Module:
        """Initialize the RTDETR model."""
        cfg = RTDETR_CONFIGS[self.size]
        backbone_kwargs: Dict[str, Any] = {}
        if cfg.get("backbone_type") == "hgnetv2":
            from .hgnetv2 import HGNetv2

            backbone_kwargs["backbone"] = HGNetv2(
                name=cfg["backbone_arch"],
                return_idx=[1, 2, 3],
                freeze_at=cfg["backbone_freeze_at"],
                freeze_norm=cfg["backbone_freeze_norm"],
                pretrained=cfg["backbone_pretrained"],
            )
        else:
            backbone_kwargs.update(
                backbone_depth=cfg["backbone_depth"],
                backbone_freeze_at=cfg["backbone_freeze_at"],
                backbone_freeze_norm=cfg["backbone_freeze_norm"],
                backbone_pretrained=cfg["backbone_pretrained"],
            )
        return RTDETRModel(
            num_classes=self.nb_classes,
            **backbone_kwargs,
            hidden_dim=cfg["encoder_hidden_dim"],
            dim_feedforward=cfg["encoder_dim_feedforward"],
            expansion=cfg["encoder_expansion"],
            decoder_hidden_dim=cfg["decoder_hidden_dim"],
            decoder_dim_feedforward=cfg.get("decoder_dim_feedforward", 1024),
            num_decoder_layers=cfg["num_decoder_layers"],
            eval_idx=cfg["eval_idx"],
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        """Return mapping of layer names to module objects."""
        return {
            "backbone": self.model.backbone,
            "encoder": self.model.encoder,
            "decoder": self.model.decoder,
        }

    def _strict_loading(self) -> bool:
        """RTDETR uses non-strict loading to handle variable layer counts."""
        return False

    # =========================================================================
    # Inference pipeline
    # =========================================================================

    @staticmethod
    def _get_preprocess_numpy():
        from .utils import preprocess_numpy

        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        """Preprocess image for RTDETR: resize to 640x640, normalize to [0,1], no letterbox."""
        import cv2
        from ...utils.image_loader import ImageLoader

        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        original_size = (orig_w, orig_h)

        # Convert PIL to numpy (RGB)
        img_np = np.array(img)

        effective_size = input_size if input_size is not None else self.input_size

        # Resize to square
        img_resized = cv2.resize(img_np, (effective_size, effective_size))

        # Normalize to [0, 1]
        img_float = img_resized.astype(np.float32) / 255.0

        # HWC -> CHW
        img_chw = img_float.transpose(2, 0, 1)

        # To tensor and add batch dimension
        input_tensor = torch.from_numpy(img_chw).unsqueeze(0)

        ratio = 1.0  # RTDETR uses direct resize, not letterbox
        return input_tensor, img, original_size, ratio

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        """Run model forward pass."""
        with torch.no_grad():
            outputs = self.model(input_tensor)
        return outputs

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        """Convert RTDETR outputs to detection results.

        Args:
            output: dict with pred_logits [1, Q, C] and pred_boxes [1, Q, 4] (cxcywh normalized)
            conf_thres: confidence threshold
            iou_thres: IoU threshold (not used for RTDETR - NMS-free)
            original_size: (width, height)
            max_det: maximum detections
            ratio: aspect ratio (1.0 for RTDETR)

        Returns:
            Dict with boxes, scores, classes, num_detections
        """
        pred_logits = output["pred_logits"]  # [1, Q, C]
        pred_boxes = output["pred_boxes"]  # [1, Q, 4] cxcywh normalized

        # Get scores and labels
        scores = torch.sigmoid(pred_logits[0])  # [Q, C]
        max_scores, labels = scores.max(dim=-1)  # [Q], [Q]

        # Filter by confidence
        mask = max_scores > conf_thres
        scores = max_scores[mask]
        labels = labels[mask]
        boxes = pred_boxes[0][mask]  # [N, 4] cxcywh normalized

        # Convert cxcywh normalized to xyxy pixel coords
        orig_w, orig_h = original_size
        cx, cy, w, h = boxes.unbind(-1)
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        # Clamp to image bounds
        boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0, orig_w)
        boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0, orig_h)

        # Limit to max_det (sort by score)
        if len(scores) > max_det:
            topk_indices = scores.argsort(descending=True)[:max_det]
            scores = scores[topk_indices]
            labels = labels[topk_indices]
            boxes_xyxy = boxes_xyxy[topk_indices]

        return {
            "boxes": boxes_xyxy.cpu(),
            "scores": scores.cpu(),
            "classes": labels.cpu(),
            "num_detections": len(boxes_xyxy),
        }

    # =========================================================================
    # Public API
    # =========================================================================

    def export(self, format: str = "onnx", *, opset: int = 17, **kwargs) -> str:
        """Export model. RTDETR requires opset >= 17 for deformable attention (F.grid_sample)."""
        return super().export(format, opset=opset, **kwargs)

    def train(
        self,
        data: str,
        *,
        epochs: int = _TRAIN_DEFAULTS.epochs,
        batch: int = _TRAIN_DEFAULTS.batch,
        imgsz: int = _TRAIN_DEFAULTS.imgsz,
        lr0: float = _TRAIN_DEFAULTS.lr0,
        lr_backbone: float = _TRAIN_DEFAULTS.lr_backbone,
        optimizer: str = _TRAIN_DEFAULTS.optimizer,
        scheduler: str = _TRAIN_DEFAULTS.scheduler,
        device: str = "",
        workers: int = _TRAIN_DEFAULTS.workers,
        seed: int = _TRAIN_DEFAULTS.seed,
        project: str = _TRAIN_DEFAULTS.project,
        name: str = _TRAIN_DEFAULTS.name,
        exist_ok: bool = _TRAIN_DEFAULTS.exist_ok,
        pretrained: bool = True,
        resume: bool = _TRAIN_DEFAULTS.resume,
        amp: bool = _TRAIN_DEFAULTS.amp,
        patience: int = _TRAIN_DEFAULTS.patience,
        allow_download_scripts: bool = False,
        **kwargs,
    ) -> dict:
        """Train the RT-DETR model on a dataset.

        Args:
            data: Path to data.yaml file (required).
            epochs: Number of epochs to train.
            batch: Batch size.
            imgsz: Input image size.
            lr0: Initial learning rate for encoder/decoder.
            lr_backbone: Initial learning rate for backbone (typically 10x lower).
            optimizer: Optimizer name ('AdamW', 'Adam', 'SGD').
            scheduler: LR scheduler ('linear', 'cos', 'warmcos').
            device: Device to train on ('' = auto-detect).
            workers: Number of dataloader workers.
            seed: Random seed for reproducibility.
            project: Root directory for training runs.
            name: Experiment name.
            exist_ok: If True, overwrite existing experiment directory.
            pretrained: Use pretrained weights if available.
            resume: If True, resume training from the loaded checkpoint.
            amp: Enable automatic mixed precision training.
            patience: Early stopping patience.

        Returns:
            Training results dict with final_loss, best_mAP50, best_mAP50_95, etc.
        """
        from .trainer import RTDETRTrainer
        from libreyolo.data import load_data_config

        try:
            data_config = load_data_config(
                data,
                autodownload=True,
                allow_scripts=allow_download_scripts,
            )
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        yaml_nc = data_config.get("nc")
        yaml_names = data_config.get("names")
        if yaml_nc is not None and yaml_nc != self.nb_classes:
            self._rebuild_for_new_classes(yaml_nc)

        # Apply custom class names from data config
        if yaml_names is not None:
            if isinstance(yaml_names, list):
                yaml_names = {i: n for i, n in enumerate(yaml_names)}
            self.names = self._sanitize_names(yaml_names, self.nb_classes)

        if seed > 0:
            import random
            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer = RTDETRTrainer(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=self.nb_classes,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            lr_backbone=lr_backbone,
            optimizer=optimizer.lower(),
            scheduler=scheduler.lower(),
            device=device if device else "auto",
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            amp=amp,
            patience=patience,
            allow_download_scripts=allow_download_scripts,
            **kwargs,
        )

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreYOLORTDETR('path/to/last.pt'); model.train(data=..., resume=True)"
                )
            trainer.resume(str(self.model_path))

        results = trainer.train()

        if Path(results["best_checkpoint"]).exists():
            self._load_weights(results["best_checkpoint"])

        return results
