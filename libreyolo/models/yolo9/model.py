"""LibreYOLO9 inference and training wrapper."""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ..base import BaseModel
from ...utils.image_loader import ImageInput
from .nn import LibreYOLO9Model
from .utils import preprocess_image, postprocess, process_mask
from ...validation.preprocessors import YOLO9ValPreprocessor


class LibreYOLO9(BaseModel):
    """YOLOv9 model for object detection and instance segmentation.

    Args:
        model_path: Path to weights, pre-loaded state_dict, or None for fresh model.
        size: Model size variant ("t", "s", "m", "c").
        reg_max: Regression max value for DFL (default: 16).
        nb_classes: Number of classes (default: 80 for COCO).
        device: Device for inference.
        segmentation: If True, use segmentation head with proto masks.

    Example::

        >>> model = LibreYOLO9(model_path="path/to/weights.pt", size="s")
        >>> detections = model(image=image_path, save=True)
    """

    # Class-level metadata
    FAMILY = "yolo9"
    FILENAME_PREFIX = "LibreYOLO9"
    INPUT_SIZES = {"t": 640, "s": 640, "m": 640, "c": 640}
    val_preprocessor_class = YOLO9ValPreprocessor

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys_lower = [k.lower() for k in weights_dict]
        return any(
            "repncspelan" in k or "adown" in k or "sppelan" in k for k in keys_lower
        ) or any("backbone.elan" in k or "neck.elan" in k for k in weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        key = "backbone.conv0.conv.weight"
        if key not in weights_dict:
            return None
        first_channel = weights_dict[key].shape[0]
        if first_channel == 16:
            return "t"
        if first_channel == 64:
            return "c"
        if first_channel == 32:
            secondary_key = "backbone.elan1.cv1.conv.weight"
            if secondary_key in weights_dict:
                mid_channel = weights_dict[secondary_key].shape[0]
                if mid_channel == 64:
                    return "s"
                elif mid_channel == 128:
                    return "m"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        for key, tensor in weights_dict.items():
            if re.match(r"head\.cv3\.\d+\.2\.weight", key):
                return tensor.shape[0]
        return None

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        model_path,
        size: str,
        reg_max: int = 16,
        nb_classes: int = 80,
        device: str = "auto",
        segmentation: bool = False,
        **kwargs,
    ):
        self.reg_max = reg_max
        self._is_segmentation = segmentation

        # Auto-detect segmentation from weights
        if not segmentation and isinstance(model_path, (str, dict)):
            self._is_segmentation = self._detect_segmentation(model_path)

        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )

        if isinstance(model_path, str):
            self._load_weights(model_path)

    @staticmethod
    def _detect_segmentation(model_path) -> bool:
        """Check if weights contain a segmentation head (proto + mask coeffs)."""
        if isinstance(model_path, dict):
            return any(k.startswith("head.proto") or k.startswith("head.cv4") for k in model_path)
        if not isinstance(model_path, str):
            return False
        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt)
            return any(k.startswith("head.proto") or k.startswith("head.cv4") for k in state)
        except Exception:
            return False

    # =========================================================================
    # Model lifecycle
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return LibreYOLO9Model(
            config=self.size, reg_max=self.reg_max, nb_classes=self.nb_classes,
            segmentation=self._is_segmentation,
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone_conv0": self.model.backbone.conv0,
            "backbone_conv1": self.model.backbone.conv1,
            "backbone_elan1": self.model.backbone.elan1,
            "backbone_down2": self.model.backbone.down2,
            "backbone_elan2": self.model.backbone.elan2,
            "backbone_down3": self.model.backbone.down3,
            "backbone_elan3": self.model.backbone.elan3,
            "backbone_down4": self.model.backbone.down4,
            "backbone_elan4": self.model.backbone.elan4,
            "backbone_spp": self.model.backbone.spp,
            "neck_elan_up1": self.model.neck.elan_up1,
            "neck_elan_up2": self.model.neck.elan_up2,
            "neck_elan_down1": self.model.neck.elan_down1,
            "neck_elan_down2": self.model.neck.elan_down2,
        }

    def _strict_loading(self) -> bool:
        return False

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        """Remap legacy 'detect.*' keys to 'head.*' for backward compatibility."""
        remapped = {}
        for key, value in state_dict.items():
            new_key = (
                key.replace("detect.", "head.", 1) if key.startswith("detect.") else key
            )
            remapped[new_key] = value
        return remapped

    def _rebuild_for_new_classes(self, new_nc: int):
        """Replace only the final classification layers for different number of classes."""
        self.nb_classes = new_nc
        self.model.nc = new_nc
        detect = self.model.head
        detect.nc = new_nc
        detect.no = new_nc + detect.reg_max * 4

        for seq in detect.cv3:
            old_final = seq[-1]
            in_channels = old_final.weight.shape[1]
            seq[-1] = nn.Conv2d(in_channels, new_nc, 1)

        detect._init_bias()
        detect._loss_fn = None
        detect.to(next(self.model.parameters()).device)

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
        effective_size = input_size if input_size is not None else 640
        tensor, img, size = preprocess_image(
            image, input_size=effective_size, color_format=color_format
        )
        return tensor, img, size, 1.0

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        actual_input_size = kwargs.get("input_size", 640)
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            max_det=max_det,
            letterbox=kwargs.get("letterbox", False),
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def train(
        self,
        data: str,
        *,
        epochs: int = 300,
        batch: int = 16,
        imgsz: int = 640,
        lr0: float = 0.01,
        optimizer: str = "SGD",
        device: str = "",
        workers: int = 8,
        seed: int = 0,
        project: str = "runs/train",
        name: str = "yolo9_exp",
        exist_ok: bool = False,
        resume: bool = False,
        amp: bool = True,
        patience: int = 50,
        **kwargs,
    ) -> dict:
        """Train the YOLOv9 model on a dataset.

        Args:
            data: Path to data.yaml file (required).
            epochs: Number of epochs to train.
            batch: Batch size.
            imgsz: Input image size.
            lr0: Initial learning rate.
            optimizer: Optimizer name ('SGD', 'Adam', 'AdamW').
            device: Device to train on ('' = auto-detect).
            workers: Number of dataloader workers.
            seed: Random seed for reproducibility.
            project: Root directory for training runs.
            name: Experiment name.
            exist_ok: If True, overwrite existing experiment directory.
            resume: If True, resume training from checkpoint.
            amp: Enable automatic mixed precision training.
            patience: Early stopping patience.

        Returns:
            Training results dict with final_loss, best_mAP50, best_mAP50_95, etc.
        """
        from .trainer import YOLO9Trainer
        from libreyolo.data import load_data_config

        try:
            data_config = load_data_config(data, autodownload=True)
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        yaml_nc = data_config.get("nc")
        if yaml_nc is not None and yaml_nc != self.nb_classes:
            self._rebuild_for_new_classes(yaml_nc)

        if seed > 0:
            import random
            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer = YOLO9Trainer(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=self.nb_classes,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            optimizer=optimizer.lower(),
            device=device if device else "auto",
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            amp=amp,
            patience=patience,
            **kwargs,
        )

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreYOLO9('path/to/last.pt', size='t'); model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))

        results = trainer.train()

        if Path(results["best_checkpoint"]).exists():
            self._load_weights(results["best_checkpoint"])

        return results
