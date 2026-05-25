"""LibreRTMDet: BaseModel subclass wiring RTMDet into the LibreYOLO factory.

Cleanroom port of RTMDet (Lyu et al., 2022) from open-mmlab/mmdetection
(Apache-2.0). Sizes: t / s / m / l / x. Detection-only in the first PR;
RTMDet-Ins (segmentation) lands as a follow-up.

Training is wired but experimental: ``model.train(..., allow_experimental=True)``.
Inference is bit-equivalent to upstream mmdet on the same checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from libreyolo.training.ddp_spawn import ddp_aware
from PIL import Image

from ...training.config import RTMDetConfig
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import RTMDetValPreprocessor
from ..base import BaseModel
from .nn import LibreRTMDetModel
from .utils import postprocess as _postprocess
from .utils import preprocess_image as _rtmdet_preprocess
from .utils import preprocess_numpy as _preprocess_numpy

_TRAIN_DEFAULTS = RTMDetConfig()


class LibreRTMDet(BaseModel):
    """RTMDet detector (CSPNeXt backbone + CSPNeXtPAFPN neck + decoupled SepBN head).

    Args:
        model_path: path to a LibreRTMDet weight file, or None for a fresh model.
        size: one of {"t", "s", "m", "l", "x"}.
        nb_classes: number of classes (default 80 for COCO).
        device: inference device.

    Examples::

        >>> model = LibreYOLO("LibreRTMDett.pt")
        >>> result = model("image.jpg", save=True)
    """

    FAMILY = "rtmdet"
    FILENAME_PREFIX = "LibreRTMDet"
    INPUT_SIZES = {"t": 640, "s": 640, "m": 640, "l": 640, "x": 640}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = RTMDetConfig
    val_preprocessor_class = RTMDetValPreprocessor

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # `rtm_cls` / `rtm_reg` are unique to RTMDet (no other family in the
        # registry uses these prefixes). Both `bbox_head.rtm_cls` (upstream)
        # and `head.rtm_cls` (LibreRTMDet checkpoints) match.
        return any(
            "rtm_cls" in k or "rtm_reg" in k for k in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        # Stem first conv has channels = 64 * widen_factor // 2.
        # tiny=12, s=16, m=24, l=32, x=40.
        for key in ("backbone.stem.0.conv.weight",):
            if key in weights_dict:
                ch = int(weights_dict[key].shape[0])
                return {12: "t", 16: "s", 24: "m", 32: "l", 40: "x"}.get(ch)
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        for key in ("head.rtm_cls.0.weight", "bbox_head.rtm_cls.0.weight"):
            if key in weights_dict:
                return int(weights_dict[key].shape[0])
        return None

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

    # =========================================================================
    # Model lifecycle
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return LibreRTMDetModel(size=self.size, nc=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "neck": self.model.neck,
            "head": self.model.head,
        }

    def _strict_loading(self) -> bool:
        # share_conv aliasing means the saved state_dict has fewer keys than the
        # model exposes (cls_convs[0] / reg_convs[0] only). Strict loading would
        # complain about the missing aliased keys.
        return False

    # =========================================================================
    # Inference pipeline
    # =========================================================================

    @staticmethod
    def _get_preprocess_numpy():
        return _preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_size = input_size if input_size is not None else self.input_size
        return _rtmdet_preprocess(
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

        # Validation path passes ratio=1.0; recompute from original_size if so.
        if ratio == 1.0 and original_size is not None:
            orig_w, orig_h = original_size
            ratio = min(actual_input_size / orig_h, actual_input_size / orig_w)

        return _postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            ratio=ratio,
            max_det=max_det,
        )

    # =========================================================================
    # Training (experimental)
    # =========================================================================

    @ddp_aware(experimental_key="allow_experimental")
    def train(
        self,
        data: str,
        *,
        allow_experimental: bool = False,
        epochs: int = _TRAIN_DEFAULTS.epochs,
        batch: int = _TRAIN_DEFAULTS.batch,
        imgsz: int | None = None,
        lr0: float = _TRAIN_DEFAULTS.lr0,
        optimizer: str = _TRAIN_DEFAULTS.optimizer,
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
        **kwargs: Any,
    ) -> dict:
        """Fine-tune LibreRTMDet on a YOLO-format dataset.

        **EXPERIMENTAL.** The QualityFocalLoss + GIoU + BatchDynamicSoftLabelAssigner
        components are cleanroom-ported from mmyolo and the trainer runs
        end-to-end. What is NOT validated:

        - small-dataset fine-tune convergence (RF1-floor parity)
        - paper-parity training-from-scratch (reproducing the 41.1 val mAP)
        - cached Mosaic + MixUp throughput (we use the standard non-cached pair)
        - the strict two-stage pipeline switch (we approximate via the shared
          ``no_aug_epochs`` mechanism)
        - paramwise weight decay overrides (norm_decay_mult=0, bias_decay_mult=0)

        What IS validated: forward + ONNX export bit-equivalent to upstream
        mmdet, postprocess matches mmdet's output to within 0.001 mAP on
        val2017 subsets. See the family docstring for the full contract.

        Pass ``allow_experimental=True`` to acknowledge.
        """
        if not allow_experimental:
            raise RuntimeError(
                "RTMDet training is experimental. The loss + assigner mirror "
                "mmyolo's BatchDynamicSoftLabelAssigner + QualityFocalLoss + "
                "GIoULoss recipe and the trainer runs end-to-end, but small-"
                "dataset fine-tune convergence and from-scratch paper parity "
                "have NOT been verified. Pass allow_experimental=True to "
                "proceed.\n"
                "Validated: inference, ONNX export, bit-equivalent to upstream "
                "mmdet on val2017 subsets within 0.001 mAP. "
                "Not validated: training convergence, multi-GPU, the strict "
                "two-stage pipeline switch, cached Mosaic/MixUp throughput."
            )
        from libreyolo.data import load_data_config

        from .trainer import RTMDetTrainer

        if imgsz is None:
            imgsz = self.input_size

        try:
            data_config = load_data_config(
                data, autodownload=True, allow_scripts=allow_download_scripts,
            )
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        yaml_nc = data_config.get("nc")
        yaml_names = data_config.get("names")
        if yaml_nc is not None and yaml_nc != self.nb_classes:
            self._rebuild_for_new_classes(yaml_nc)
        if yaml_names is not None:
            if isinstance(yaml_names, list):
                yaml_names = {i: n for i, n in enumerate(yaml_names)}
            self.names = self._sanitize_names(yaml_names, self.nb_classes)

        if seed >= 0:
            import random
            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer = RTMDetTrainer(
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
            **kwargs,
        )

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreRTMDet('path/to/last.pt'); model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))

        results = trainer.train()
        if Path(results["best_checkpoint"]).exists():
            self._load_weights(results["best_checkpoint"])
        return results
