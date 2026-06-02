"""LibreFOMO point localizer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ...training.config import FOMOConfig
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import LibreFOMOValPreprocessor
from ..base import BaseModel
from .nn import CONFIGS, LibreFOMOModel, detect_size_from_state_dict
from .utils import decode_points_from_logits, postprocess as _postprocess
from .utils import preprocess_image, preprocess_numpy


class LibreFOMO(BaseModel):
    """FOMO-style point localizer.

    The in-tree model code is Apache-2.0. Official prepared weights are hosted
    externally at ``bdanko/LibreFOMO`` under cc-by-nc-4.0 due to ImageNet-derived
    weight licensing ambiguity.
    """

    FAMILY = "librefomo"
    FILENAME_PREFIX = "LibreFOMO"
    INPUT_SIZES = {k: int(v["imgsz"]) for k, v in CONFIGS.items()}
    SUPPORTED_TASKS = ("point",)
    DEFAULT_TASK = "point"
    TRAIN_CONFIG = FOMOConfig
    val_preprocessor_class = LibreFOMOValPreprocessor
    TTA_ENABLED = False

    _LICENSE_NOTICE_SHOWN = False
    _WEIGHTS_REPO = "bdanko/LibreFOMO"

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return "head.weight" in weights_dict and any(k.startswith("backbone.block_6_expand") for k in weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return detect_size_from_state_dict(weights_dict)

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        weight = weights_dict.get("head.weight")
        if weight is None:
            return None
        return max(int(weight.shape[0]) - 1, 1)

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        if cls.detect_size_from_filename(filename) is None:
            return None
        return f"https://huggingface.co/{cls._WEIGHTS_REPO}/resolve/main/{filename}"

    @classmethod
    def _notify_license_once(cls) -> None:
        if cls._LICENSE_NOTICE_SHOWN:
            return
        cls._LICENSE_NOTICE_SHOWN = True
        print(
            "\n"
            "----------------------------------------------------------------\n"
            f"LibreFOMO weights are hosted externally at {cls._WEIGHTS_REPO}\n"
            "under cc-by-nc-4.0.\n"
            "They are treated as externally hosted, ImageNet-derived weights and\n"
            "are not redistributed by LibreYOLO. By downloading them you accept\n"
            "the Hugging Face repository license terms.\n"
            "----------------------------------------------------------------\n"
        )

    def __init__(
        self,
        model_path=None,
        size: str = "m",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=task,
            **kwargs,
        )
        if isinstance(model_path, str):
            self._load_weights(model_path)

    def _init_model(self) -> nn.Module:
        return LibreFOMOModel(size=self.size, nc=self.nb_classes, head_channels=self.nb_classes + 1)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {"backbone": self.model.backbone, "head": self.model.head}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        return preprocess_image(image, input_size or self.input_size, color_format=color_format)

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
        return _postprocess(
            output,
            conf_thres=conf_thres,
            input_size=kwargs.get("input_size", self.input_size),
            original_size=original_size,
            nms_radius=int(kwargs.get("nms_radius", 1)),
            max_det=max_det,
        )

    def _point_metric_shape(self, output: Any, imgsz: int) -> tuple[int, int]:
        return tuple(int(v) for v in output.shape[-2:])

    def _decode_point_predictions(
        self,
        output: Any,
        conf_thres: float,
        max_det: int = 300,
        nms_radius: int = 1,
        **kwargs,
    ) -> list[torch.Tensor]:
        decoded = decode_points_from_logits(
            output.detach().cpu(),
            conf_threshold=conf_thres,
            nms_radius=nms_radius,
            max_points=max_det,
        )
        # Utility rows are (x, y, class_channel, confidence); validator rows
        # use dataset class ids, so subtract the background channel offset.
        return [
            torch.cat((rows[:, :2], (rows[:, 2:3] - 1), rows[:, 3:4]), dim=1)
            if len(rows)
            else rows
            for rows in decoded
        ]

    def _point_targets_from_boxes(
        self,
        target: torch.Tensor,
        metric_shape: tuple[int, int],
        imgsz: int,
    ) -> torch.Tensor:
        valid = target[:, 2] > target[:, 0]
        target = target[valid]
        if target.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.float32)
        metric_h, metric_w = metric_shape
        cx = (target[:, 0] + target[:, 2]) * 0.5 * (metric_w / imgsz)
        cy = (target[:, 1] + target[:, 3]) * 0.5 * (metric_h / imgsz)
        return torch.stack((cx, cy, target[:, 4]), dim=1).float()

    def _load_weights(self, model_path: str):
        self._notify_license_once()
        return super()._load_weights(model_path)

    def train(self, allow_experimental: bool = False, **kwargs):
        """Train this LibreFOMO model.

        Training is marked experimental in v1. Pass ``allow_experimental=True``
        to enable. Stable inference and validation do not require this flag.

        Args:
            allow_experimental: Must be ``True`` to start training.
            **kwargs: Training config overrides forwarded to
                :class:`~libreyolo.models.librefomo.trainer.LibreFOMOTrainer`.
                Notable keys: ``data`` (YOLO data.yaml path), ``epochs``, ``batch``,
                ``lr0``, ``fg_weight``, ``device``, ``project``, ``name``.

        Returns:
            Training results dict (see :meth:`BaseTrainer.train`).
        """
        if not allow_experimental:
            raise NotImplementedError(
                "LibreFOMO training is experimental in this version. "
                "Pass allow_experimental=True to model.train() to enable it."
            )

        from .trainer import LibreFOMOTrainer

        # imgsz is determined by the model size unless the caller overrides it
        if "imgsz" not in kwargs:
            kwargs["imgsz"] = self.input_size
        if "size" not in kwargs:
            kwargs["size"] = self.size
        if "num_classes" not in kwargs:
            kwargs["num_classes"] = self.nb_classes

        trainer = LibreFOMOTrainer(
            model=self.model,
            wrapper_model=self,
            **kwargs,
        )
        results = trainer.train()
        # Reload the best (or last) checkpoint so the wrapper is the trained model
        from pathlib import Path

        best_ckpt = results.get("best_checkpoint", "")
        last_ckpt = results.get("last_checkpoint", "")
        reload_path = best_ckpt if Path(best_ckpt).exists() else last_ckpt
        if reload_path and Path(reload_path).exists():
            self._load_weights(reload_path)
        return results

