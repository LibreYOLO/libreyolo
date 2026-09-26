"""LibreFOMO — FOMO point-localizer family wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...training.callbacks import TrainCallbacks
from ..base.model import BaseModel
from .nn import CONFIGS, LibreFOMOModel, detect_size_from_state_dict
from .utils import postprocess as postprocess_fomo
from .utils import preprocess_numpy
from ...training.config import FOMOConfig
from ...training.ddp_spawn import ddp_aware
from ...validation.preprocessors import FOMOValPreprocessor
from ...validation.fomo_validator import FOMOValidator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LibreFOMO — BaseModel subclass
# ---------------------------------------------------------------------------


class LibreFOMO(BaseModel):
    """LibreFOMO point localizer: image → (x, y, class, confidence) keypoint predictions."""

    FAMILY = "fomo"
    FILENAME_PREFIX = "LibreFOMO"
    WEIGHT_EXT = ".pt"

    INPUT_SIZES: ClassVar[Dict[str, int]] = {k: int(v["imgsz"]) for k, v in CONFIGS.items()}

    SUPPORTED_TASKS = ("point",)
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    DEFAULT_TASK = "point"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = FOMOConfig
    RESUME_RESTORES_TRAIN_ARGS = True
    val_preprocessor_class = FOMOValPreprocessor
    validator_class = FOMOValidator

    TTA_ENABLED = False

    # -------------------------------------------------------------------------
    # Registry / can_load interface
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_state_dict_keys(weights_dict: dict) -> dict:
        return {k.removeprefix("module."): v for k, v in weights_dict.items()}

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        weights_dict = cls._normalize_state_dict_keys(weights_dict)
        return (
            "head.weight" in weights_dict
            and any(k.startswith("backbone.block_6_expand") for k in weights_dict)
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return detect_size_from_state_dict(weights_dict)

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        weights_dict = cls._normalize_state_dict_keys(weights_dict)
        weight = weights_dict.get("head.weight")
        if weight is None:
            return None
        return max(int(weight.shape[0]) - 1, 1)

    @classmethod
    def get_download_url(cls, _filename: str) -> Optional[str]:
        # LibreFOMO pretrained weights are not redistributed or auto-downloaded.
        # Users who accept third-party terms must provide a local checkpoint path.
        return None

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    def __init__(
        self,
        model_path: str | Path | None = None,
        size: str = "m",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=task,
            **kwargs,
        )
        if isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))

    # -------------------------------------------------------------------------
    # BaseModel abstract surface
    # -------------------------------------------------------------------------

    def _init_model(self) -> nn.Module:
        return LibreFOMOModel(size=self.size, nc=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {"backbone": self.model.backbone, "head": self.model.head}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: Any,
        color_format: str = "auto",
        input_size: int | None = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        import numpy as np
        from PIL import Image as _Image

        from ...utils.image_loader import ImageLoader

        img = ImageLoader.load(image, color_format=color_format)
        pil_img = img.convert("RGB")
        isz = int(input_size or self._get_input_size())
        pil_resized = pil_img.resize((isz, isz), _Image.Resampling.BILINEAR)
        arr = np.asarray(pil_resized, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        chw = np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)
        tensor = torch.from_numpy(chw).unsqueeze(0)
        return tensor, pil_img, pil_img.size, 1.0

    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if output.dim() == 3:
            output = output.unsqueeze(0)
        return postprocess_fomo(
            output,
            conf_thres=conf_thres,
            input_size=self._get_input_size(),
            original_size=original_size,
            nms_radius=int(kwargs.get("nms_radius", 1)),
            max_det=max_det,
        )

    # -------------------------------------------------------------------------
    # PointValidator integration hooks
    # -------------------------------------------------------------------------

    def _parse_gt_points(
        self,
        gt_row: Any,
        orig_h: int,
        orig_w: int,
        validator: Any,
    ) -> Tuple[Any, Any]:
        return validator.parse_gt_points_from_boxes(gt_row, orig_h, orig_w)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs: Any,
    ) -> Dict:
        """Train LibreFOMO.

        Args:
            callbacks: Optional training callback or iterable of callbacks.
            loggers: Optional built-in experiment loggers: a registered name,
                a configured logger instance, or an iterable mixing both.
        """
        from .trainer import FOMOTrainer

        if "imgsz" not in kwargs:
            kwargs["imgsz"] = self._get_input_size()
        else:
            native_size = self._get_input_size()
            if int(kwargs["imgsz"]) != native_size:
                raise ValueError(
                    f"LibreFOMO size '{self.size}' only supports imgsz={native_size}. "
                    f"Got imgsz={kwargs['imgsz']}."
                )
        if "size" not in kwargs:
            kwargs["size"] = self.size
        if "num_classes" not in kwargs:
            kwargs["num_classes"] = self.nb_classes

        seed = kwargs.get("seed", 0)
        device = kwargs.get("device", "")
        if seed >= 0:
            import random

            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if str(device).lower() not in ("cpu", "mps") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer = FOMOTrainer(
            model=self.model,
            wrapper_model=self,
            data=data,
            callbacks=callbacks,
            loggers=loggers,
            **kwargs,
        )
        if kwargs.get("resume"):
            trainer.setup()
            trainer.resume(self._resume_checkpoint(kwargs["resume"]))

        results = trainer.train()

        reload_path = None
        for key in ("best_checkpoint", "last_checkpoint"):
            path = results.get(key)
            if path and Path(path).exists():
                reload_path = str(path)
                break

        if reload_path:
            self._load_weights(reload_path)

        return results
