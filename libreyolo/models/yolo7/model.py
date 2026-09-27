"""LibreYOLO7 — YOLOv7 detector, native port of MIT MultimediaTechLab/YOLO.

Source provenance: architecture and weights derive from MultimediaTechLab/YOLO
(MIT, (c) 2024 Kin-Yiu Wong & Hao-Tang Tsui) — the authors' own MIT re-release
of YOLOv7. NOT the GPL-3.0 ``WongKinYiu/yolov7``. The native modules mirror the
upstream names so ``v7.pt`` loads with no remapping (see ``blocks.py`` / ``net.py``).

Single size ``b`` (upstream ships one v7 model). Inference reproduces upstream
``Anc2Box`` exactly; training uses LibreYOLO's SimOTA loss (see :mod:`.loss`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from libreyolo.training.ddp_spawn import ddp_aware
from PIL import Image

from ...training.callbacks import TrainCallbacks
from ...training.config import YOLOv7Config
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import YOLO9ValPreprocessor
from ..base import BaseModel
from .net import YOLOv7Model

# Single source of truth for training defaults.
_TRAIN_DEFAULTS = YOLOv7Config()


class LibreYOLO7(BaseModel):
    """YOLOv7 object detector (anchor-based, implicit-knowledge head)."""

    FAMILY = "yolo7"
    FILENAME_PREFIX = "LibreYOLO7"
    INPUT_SIZES = {"b": 640}
    SUPPORTED_TASKS = ("detect",)
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    DEFAULT_TASK = "detect"
    # Letterbox + RGB + /255 + gray(114) pad — same contract as YOLO9.
    val_preprocessor_class = YOLO9ValPreprocessor
    # Machine-readable "trainable" flag: the CLI discovers family training
    # defaults through this (cli/config.py); None would mean inference-only.
    TRAIN_CONFIG = YOLOv7Config
    RESUME_RESTORES_TRAIN_ARGS = True

    # =====================================================================
    # Registry classmethods
    # =====================================================================
    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # ImplicitA/M (YOLOR implicit knowledge) heads are unique to YOLOv7.
        return any("implicit_a.implicit" in k for k in weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return "b" if cls.can_load(weights_dict) else None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        for k, v in weights_dict.items():
            if k.endswith("heads.0.head_conv.weight"):
                return int(v.shape[0]) // 3 - 5
        return None

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> Optional[dict]:
        """Recognize + remap a raw upstream MMT ``v7.pt`` for auto-conversion.

        Upstream ``v7.pt`` is a flat state dict with numbered keys
        (``0.conv.weight`` ... ``105.heads.0.implicit_a.implicit``) and no
        ``layers.`` prefix. The native model nests the graph under ``layers.``,
        so ``LibreYOLO("v7.pt")`` (metadata-less) must add that prefix here —
        otherwise the default hook would claim the checkpoint and strict-load
        the un-prefixed keys, failing with missing/unexpected key errors.
        """
        if not any("implicit_a.implicit" in k for k in state_dict):
            return None
        # Our own converted checkpoints already carry the layers.* prefix.
        if any(k.startswith("layers.") for k in state_dict):
            return dict(state_dict)
        return {f"layers.{k}": v for k, v in state_dict.items()}

    # =====================================================================
    # Init
    # =====================================================================
    def __init__(self, model_path=None, size: str = "b", nb_classes: int = 80,
                 device: str = "auto", **kwargs):
        super().__init__(model_path=model_path, size=size, nb_classes=nb_classes,
                         device=device, **kwargs)
        if isinstance(model_path, str):
            self._load_weights(model_path)

    def _init_model(self) -> nn.Module:
        return YOLOv7Model(num_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {"model": self.model}

    def _strict_loading(self) -> bool:
        return True  # native names match v7.pt exactly (verified 564/564)

    # =====================================================================
    # Inference pipeline
    # =====================================================================
    @staticmethod
    def _get_preprocess_numpy():
        from .utils import preprocess_numpy
        return preprocess_numpy

    def _preprocess(self, image: ImageInput, color_format: str = "auto",
                    input_size: Optional[int] = None):
        from .utils import preprocess_image
        size = input_size if input_size is not None else self.input_size
        return preprocess_image(image, input_size=size, color_format=color_format)

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(self, output: Any, conf_thres: float, iou_thres: float,
                     original_size: Tuple[int, int], max_det: int = 300,
                     ratio: float = 1.0, **kwargs) -> Dict:
        from ...postprocess.yolo7 import postprocess as _pp
        input_size = kwargs.get("input_size", self.input_size)
        return _pp(
            output,
            self.model.anchors,
            self.model.strides,
            self.nb_classes,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=input_size,
            original_size=original_size,
            ratio=ratio,
            max_det=max_det,
        )

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        epochs: int = _TRAIN_DEFAULTS.epochs,
        batch: int = _TRAIN_DEFAULTS.batch,
        imgsz: int = _TRAIN_DEFAULTS.imgsz,
        lr0: float = _TRAIN_DEFAULTS.lr0,
        optimizer: str = _TRAIN_DEFAULTS.optimizer,
        device: str = "",
        workers: int = _TRAIN_DEFAULTS.workers,
        seed: int = _TRAIN_DEFAULTS.seed,
        project: str = _TRAIN_DEFAULTS.project,
        name: str = _TRAIN_DEFAULTS.name,
        exist_ok: bool = _TRAIN_DEFAULTS.exist_ok,
        pretrained: "bool | str | Path | None" = None,
        resume: bool = _TRAIN_DEFAULTS.resume,
        amp: bool = _TRAIN_DEFAULTS.amp,
        patience: int = _TRAIN_DEFAULTS.patience,
        allow_download_scripts: bool = False,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> dict:
        """Train the YOLOv7 model on a dataset.

        v7's anchor-based head is trained with LibreYOLO's SimOTA loss
        (:class:`.loss.YOLOv7Loss`, adapted from the in-repo Apache-2.0 YOLOX
        assignment). Same API as the other detectors.

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
            pretrained: Optional training initialization weights. Use True to
                warm-start from the default COCO checkpoint
                (``LibreYOLO7b.pt``, auto-downloaded), or pass a path/name.
                Incompatible with ``resume=True``. Default None trains the
                currently loaded weights (or from scratch).
            resume: If True, resume training from the loaded checkpoint.
            amp: Enable automatic mixed precision training.
            patience: Early stopping patience.
            callbacks: Optional training callback or iterable of callbacks.
            loggers: Optional built-in experiment loggers.

        Returns:
            Training results dict with final_loss, best_mAP50, best_mAP50_95, etc.
        """
        from .trainer import YOLOv7Trainer
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

        if resume and pretrained:
            raise ValueError("pretrained transfer cannot be combined with resume=True.")

        did_pretrained = False
        if pretrained:
            weights = (
                pretrained
                if isinstance(pretrained, (str, Path))
                else f"{self.FILENAME_PREFIX}{self.size}.pt"
            )
            # The published v7 checkpoints are nc=80 (COCO); load at that
            # geometry, then the yaml_nc rebuild below transfers every
            # shape-matching tensor (backbone/neck) into the target head count.
            if self.nb_classes != 80:
                self._rebuild_for_new_classes(80)
            self._load_weights(str(weights))
            did_pretrained = True

        yaml_nc = data_config.get("nc")
        yaml_names = data_config.get("names")
        if yaml_nc is None and yaml_names is not None:
            yaml_nc = len(yaml_names)
        rebuilt = yaml_nc is not None and int(yaml_nc) != self.nb_classes
        if rebuilt:
            self._rebuild_for_new_classes(int(yaml_nc))

        if yaml_names is not None:
            if isinstance(yaml_names, list):
                yaml_names = {i: n for i, n in enumerate(yaml_names)}
            self.names = self._sanitize_names(yaml_names, self.nb_classes)

        # Bias-prior rules (focal prior on obj/cls logits, idempotent):
        # - nc rebuild leaves a fresh random head regardless of how the
        #   backbone was initialized -> always prime it.
        # - from-scratch run (no checkpoint loaded, no pretrained) -> prime.
        # - warm start with matching nc -> keep the trained priors.
        if rebuilt or (not did_pretrained and self.model_path is None):
            self.model.initialize_biases(0.01)

        if seed >= 0:
            import random
            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if str(device).lower() not in ("cpu", "mps") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer = YOLOv7Trainer(
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
            allow_download_scripts=allow_download_scripts,
            callbacks=callbacks,
            loggers=loggers,
            **kwargs,
        )

        if resume:
            trainer.setup()
            trainer.resume(self._resume_checkpoint(resume))

        results = trainer.train()

        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self._load_weights(best_ckpt)

        return results
