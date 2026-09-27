"""LibreRTMDet: BaseModel subclass wiring RTMDet into the LibreYOLO factory.

Port of RTMDet (Lyu et al., 2022) from open-mmlab/mmdetection
(Apache-2.0). Sizes: t / s / m / l / x. Supports detection and RTMDet-Ins
instance segmentation inference and validation.

Detection training is implemented. RTMDet-Ins training is not implemented.
Inference is compatible with official mmdetection checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from libreyolo.training.ddp_spawn import ddp_aware
from PIL import Image

from ...training.callbacks import TrainCallbacks
from ...training.config import RTMDetConfig
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import RTMDetValPreprocessor
from ..base import BaseModel
from .nn import LibreRTMDetModel
from ...postprocess.rtmdet import postprocess as _postprocess
from .utils import preprocess_image as _rtmdet_preprocess
from .utils import preprocess_numpy as _preprocess_numpy

_TRAIN_DEFAULTS = RTMDetConfig()


class LibreRTMDet(BaseModel):
    """RTMDet detection and RTMDet-Ins instance segmentation.

    Args:
        model_path: path to a LibreRTMDet weight file, or None for a fresh model.
        size: one of {"t", "s", "m", "l", "x"}.
        nb_classes: number of classes (default 80 for COCO).
        device: inference device.

    Examples::

        >>> model = LibreYOLO("LibreRTMDett.pt")
        >>> result = model("image.jpg", save=True)
        >>> segmenter = LibreYOLO("LibreRTMDett-seg.pt")
        >>> masks = segmenter("image.jpg")[0].masks
    """

    FAMILY = "rtmdet"
    FILENAME_PREFIX = "LibreRTMDet"
    INPUT_SIZES = {"t": 640, "s": 640, "m": 640, "l": 640, "x": 640}
    SUPPORTED_TASKS = ("detect", "segment")
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    DEFAULT_TASK = "detect"
    TASK_INPUT_SIZES = {
        "detect": INPUT_SIZES,
        "segment": INPUT_SIZES,
    }
    TRAIN_CONFIG = RTMDetConfig
    RESUME_RESTORES_TRAIN_ARGS = True
    val_preprocessor_class = RTMDetValPreprocessor

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # `rtm_cls` / `rtm_reg` are unique to RTMDet (no other family in the
        # registry uses these prefixes). Both `bbox_head.rtm_cls` (upstream)
        # and `head.rtm_cls` (LibreRTMDet checkpoints) match.
        return any("rtm_cls" in k or "rtm_reg" in k for k in weights_dict)

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

    @classmethod
    def detect_checkpoint_task(cls, weights_dict: dict) -> Optional[str]:
        if any("rtm_kernel" in key or ".mask_head." in key for key in weights_dict):
            return "segment"
        return None

    @classmethod
    def convert_upstream_state_dict(cls, weights_dict: dict) -> Optional[dict]:
        """Remap mm-series ``bbox_head`` naming to LibreRTMDet's ``head``."""
        from .convert import convert_upstream, is_upstream_state_dict

        if not is_upstream_state_dict(weights_dict):
            return None
        return convert_upstream(weights_dict)

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        model_path=None,
        size: str = "s",
        nb_classes: int = 80,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ):
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

    # =========================================================================
    # Model lifecycle
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return LibreRTMDetModel(
            size=self.size,
            nc=self.nb_classes,
            enable_mask_head=self.task == "segment",
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        layers = {
            "backbone": self.model.backbone,
            "neck": self.model.neck,
            "head": self.model.head,
        }
        if self.task == "segment":
            layers["mask_head"] = self.model.head.mask_head
        return layers

    @property
    def _is_segmentation(self) -> bool:
        return self.task == "segment"

    def _validate_loaded_state_dict_for_task(
        self,
        state_dict: dict,
        checkpoint: dict | None = None,
    ) -> None:
        detected_task = self.detect_checkpoint_task(state_dict)
        if detected_task == "segment" and self.task != "segment":
            raise RuntimeError(
                "RTMDet-Ins segmentation checkpoints must be loaded with "
                "task='segment' or a '-seg' filename suffix."
            )
        if self.task == "segment" and detected_task is None:
            raise RuntimeError(
                "This RTMDet checkpoint has no instance-segmentation mask head, "
                "but the model was initialized for task='segment'. Use an "
                "RTMDet-Ins '-seg' checkpoint."
            )

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
            if isinstance(actual_input_size, (list, tuple)):
                actual_h, actual_w = int(actual_input_size[0]), int(actual_input_size[1])
            else:
                actual_h = actual_w = int(actual_input_size)
            ratio = min(actual_w / orig_w, actual_h / orig_h)

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
    # Training
    # =========================================================================

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
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
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs: Any,
    ) -> dict:
        """Fine-tune LibreRTMDet on a YOLO-format dataset.

        The QualityFocalLoss + GIoU + DynamicSoftLabelAssigner components are
        ported from mmdetection (Apache-2.0) and the trainer runs end-to-end.
        The following have not been validated:

        - small-dataset fine-tune convergence (RF1-floor parity)
        - paper-parity training-from-scratch (reproducing the 41.1 val mAP)
        - cached Mosaic + MixUp throughput (we use the standard non-cached pair)
        - the strict two-stage pipeline switch (we approximate via the shared
          ``no_aug_epochs`` mechanism)
        - paramwise weight decay overrides (norm_decay_mult=0, bias_decay_mult=0)

        What IS validated: forward + ONNX export bit-equivalent to upstream
        mmdet, postprocess matches mmdet's output to within 0.001 mAP on
        val2017 subsets. See the family docstring for the full contract.

        Args:
            callbacks: Optional training callback or iterable of callbacks.
            loggers: Optional built-in experiment loggers: a registered name,
                a configured logger instance, or an iterable mixing both.
        """
        if self.task == "segment":
            raise NotImplementedError(
                "RTMDet-Ins training is not implemented yet. Instance "
                "segmentation currently supports inference and validation."
            )
        from libreyolo.data import load_data_config

        from .trainer import RTMDetTrainer

        if imgsz is None:
            imgsz = self.input_size

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
        # If no nc in data.yaml, infer it by counting.
        if yaml_nc is None and yaml_names is not None:
            yaml_nc = len(yaml_names)
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
            if str(device).lower() not in ("cpu", "mps") and torch.cuda.is_available():
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
