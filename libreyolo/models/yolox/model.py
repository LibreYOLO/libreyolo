"""LibreYOLOX implementation for LibreYOLO."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ..base import BaseModel
from ...utils.image_loader import ImageInput
from .nn import LibreYOLOXModel
from .utils import preprocess_image as _yolox_preprocess, postprocess
from ...validation.preprocessors import YOLOXValPreprocessor


class LibreYOLOX(BaseModel):
    """YOLOX model for object detection.

    Args:
        model_path: Path to weights, pre-loaded state_dict, or None for fresh model.
        size: Model size variant ("n", "t", "s", "m", "l", "x").
        nb_classes: Number of classes (default: 80 for COCO).
        device: Device for inference.

    Examples::

        >>> model = LibreYOLO("LibreYOLOXs.pt")
        >>> detections = model(image="image.jpg", save=True)

        >>> model = LibreYOLOX(size="s")
        >>> results = model.train(data="coco128.yaml", epochs=100)
    """

    # Class-level metadata
    FAMILY = "yolox"
    FILENAME_PREFIX = "LibreYOLOX"
    INPUT_SIZES = {"n": 416, "t": 416, "s": 640, "m": 640, "l": 640, "x": 640}
    val_preprocessor_class = YOLOXValPreprocessor

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return any("backbone.backbone" in k or "head.stems" in k for k in weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        key = "backbone.backbone.stem.conv.conv.weight"
        if key not in weights_dict:
            return None
        ch = weights_dict[key].shape[0]
        return {16: "n", 24: "t", 32: "s", 48: "m", 64: "l", 80: "x"}.get(ch)

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        key = "head.cls_preds.0.weight"
        return weights_dict[key].shape[0] if key in weights_dict else None

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        model_path=None,
        size: str = "s",
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

        # Nano-specific BatchNorm settings (must run after weight loading)
        if self.size == "n":
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

    # =========================================================================
    # Model lifecycle
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return LibreYOLOXModel(config=self.size, nb_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone_stem": self.model.backbone.stem,
            "backbone_dark2": self.model.backbone.dark2,
            "backbone_dark3": self.model.backbone.dark3,
            "backbone_dark4": self.model.backbone.dark4,
            "backbone_dark5": self.model.backbone.dark5,
        }

    def _strict_loading(self) -> bool:
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
        effective_size = input_size if input_size is not None else self.input_size
        return _yolox_preprocess(
            image, input_size=effective_size, color_format=color_format
        )

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

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
        actual_input_size = kwargs.get("input_size", self.input_size)

        # Recompute ratio if caller passed default (batch validation path)
        if ratio == 1.0 and original_size is not None:
            orig_w, orig_h = original_size
            ratio = min(actual_input_size / orig_h, actual_input_size / orig_w)

        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            ratio=ratio,
            max_det=max_det,
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def train(
        self,
        data: str,
        *,
        epochs: int = 100,
        batch: int = 16,
        imgsz: int = 640,
        lr0: float = 0.01,
        optimizer: str = "SGD",
        device: str = "",
        workers: int = 8,
        seed: int = 0,
        project: str = "runs/train",
        name: str = "exp",
        exist_ok: bool = False,
        pretrained: bool = True,
        resume: bool = False,
        amp: bool = True,
        patience: int = 50,
        **kwargs,
    ) -> dict:
        """Train the YOLOX model on a dataset.

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
            pretrained: Use pretrained weights if available.
            resume: If True, resume training from the loaded checkpoint.
            amp: Enable automatic mixed precision training.
            patience: Early stopping patience.

        Returns:
            Training results dict with final_loss, best_mAP50, best_mAP50_95, etc.
        """
        from .trainer import YOLOXTrainer
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

        trainer = YOLOXTrainer(
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
                    "model = LibreYOLOX('path/to/last.pt'); model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))

        results = trainer.train()

        if Path(results["best_checkpoint"]).exists():
            self._load_weights(results["best_checkpoint"])

        return results
