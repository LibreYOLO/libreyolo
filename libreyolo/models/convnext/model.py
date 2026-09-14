"""LibreConvNeXt: BaseModel subclass wiring ConvNeXt classification into the factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ...training.ddp_spawn import ddp_aware
from ...training.callbacks import TrainCallbacks
from ...postprocess.convnext import postprocess as _cnx_postprocess
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .config import ConvNeXtConfig
from .nn import ConvNeXt
from .utils import preprocess_image as _cnx_preprocess

_TRAIN_DEFAULTS = ConvNeXtConfig()


class LibreConvNeXt(BaseModel):
    """ConvNeXt V1 image classifier (tiny/small/base).

    Examples::

        >>> model = LibreYOLO("LibreConvNeXtt-cls.pt")
        >>> result = model.predict("cat.jpg")[0]
        >>> result.probs.top1, result.probs.top5

        >>> model = LibreConvNeXt(size="t")
        >>> model.train(data="imagenette160", epochs=5)
    """

    FAMILY = "convnext"
    FILENAME_PREFIX = "LibreConvNeXt"
    INPUT_SIZES = {"t": 224, "s": 224, "b": 224}
    SUPPORTED_TASKS = ("classify",)
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    DEFAULT_TASK = "classify"
    REQUIRE_TASK_SUFFIX = True  # canonical weights are LibreConvNeXt<size>-cls.pt
    TRAIN_CONFIG = ConvNeXtConfig

    # timm eval crop_pct per checkpoint — convnext_*.fb_in1k all use 0.875.
    CROP_PCT = {"t": 0.875, "s": 0.875, "b": 0.875}

    # ---- registry --------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # ConvNeXt signature: patch stem + layer-scale gamma + LayerNorm head fc.
        # The gamma layer-scale parameter is unique to ConvNeXt among all
        # families (detectors and MobileNetV4 have no per-block ``gamma``).
        return (
            "stem.0.weight" in weights_dict
            and "head.fc.weight" in weights_dict
            and any(k.endswith(".gamma") and k.startswith("stages.") for k in weights_dict)
            # Only claim it if the exact (dim, depth) signature is a shipped V1
            # size — keeps autoconvert from mis-claiming ConvNeXt-L/XL.
            and cls.detect_size(weights_dict) is not None
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        if "stem.0.weight" not in weights_dict or "head.fc.weight" not in weights_dict:
            return None
        import re

        dim0 = int(weights_dict["stem.0.weight"].shape[0])
        s2 = -1
        for k in weights_dict:
            m = re.match(r"^stages\.2\.blocks\.(\d+)\.gamma$", k)
            if m:
                s2 = max(s2, int(m.group(1)))
        # (stem dim, stage-2 depth) -> exact shipped V1 size. tiny/small share
        # dims (96) and differ by stage-2 depth (9 vs 27); base widens to 128.
        # Rejects ConvNeXt-L/XL (dims 192/256) by returning None.
        return {(96, 9): "t", (96, 27): "s", (128, 27): "b"}.get((dim0, s2 + 1))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        key = "head.fc.weight"
        if key not in weights_dict:
            return None
        return int(weights_dict[key].shape[0])

    # ---- init ------------------------------------------------------------

    def __init__(
        self,
        model_path=None,
        size: str = "t",
        nb_classes: int = 1000,
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
        self.crop_pct = self.CROP_PCT[self.size]
        self.interpolation = "bicubic"
        if isinstance(model_path, str):
            self._load_weights(model_path)

    def _init_model(self) -> nn.Module:
        return ConvNeXt(size=self.size, num_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "stem": self.model.stem,
            "stages": self.model.stages,
            "head_norm": self.model.head.norm,
            "classifier": self.model.head.fc,
        }

    def _rebuild_for_new_classes(self, new_nb_classes: int) -> None:
        """Swap the final Linear for a new class count (backbone preserved)."""
        self.nb_classes = new_nb_classes
        self.names = {i: f"class_{i}" for i in range(new_nb_classes)}
        self.model.reset_classifier(new_nb_classes)
        self.model.to(self.device)

    def _prepare_model_for_state_dict(self, state_dict: dict) -> None:
        """Replay LoRA injection for adapter checkpoints (lora=True training
        saves block MLPs under peft keys; rebuild the adapted graph so the
        keys line up before the strict load)."""
        from ...training.lora import (
            apply_lora_to_convnext,
            module_has_lora,
            state_dict_has_lora,
        )

        if state_dict_has_lora(state_dict) and not module_has_lora(self.model):
            apply_lora_to_convnext(self.model)

    # ---- inference -------------------------------------------------------

    def _get_preprocess_numpy(self):
        # Instance method (not staticmethod) so the per-variant crop_pct is bound.
        # The base contract calls this as ``preprocess_numpy(img, input_size)``
        # (e.g. the exporter's INT8 calibration), so binding crop_pct here avoids
        # relying on the default matching this family's value.
        from functools import partial

        from .utils import preprocess_numpy

        return partial(preprocess_numpy, crop_pct=self.crop_pct)

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        eff = input_size if input_size is not None else self.input_size
        return _cnx_preprocess(
            image, input_size=eff, crop_pct=self.crop_pct, color_format=color_format
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
        return _cnx_postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=max_det,
            ratio=ratio,
        )

    # ---- training --------------------------------------------------------

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
        resume: bool = _TRAIN_DEFAULTS.resume,
        amp: bool = _TRAIN_DEFAULTS.amp,
        patience: int = _TRAIN_DEFAULTS.patience,
        callbacks: TrainCallbacks = None,
        **kwargs: Any,
    ) -> dict:
        """Fine-tune the classifier on an ImageFolder-style dataset.

        ``data`` is a dataset root (``train/`` + ``val/`` folder-per-class), a
        known name (e.g. ``"imagenette160"``), or a ``.zip`` URL. The head is
        rebuilt to the dataset's class count automatically. Cross-entropy +
        AdamW + cosine; the ImageNet-pretrained backbone transfers cleanly.

        ``cls_pw`` (classification only, float in [0, 1], default 0) controls
        inverse-frequency weighting strength with mean-one class weights.
        ``class_weights=True`` retains legacy sample-normalized weighting and
        cannot be combined with ``cls_pw>0``. Neither option changes sampling.
        See docs/classification_training.md for compatibility and resume rules.
        """
        from .trainer import ConvNeXtTrainer

        if imgsz is None:
            imgsz = self.input_size

        trainer = ConvNeXtTrainer(
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
            callbacks=callbacks,
            **kwargs,
        )

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreConvNeXt('path/to/last.pt', size='t'); "
                    "model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))

        results = trainer.train()
        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self._load_weights(best_ckpt)
        return results
