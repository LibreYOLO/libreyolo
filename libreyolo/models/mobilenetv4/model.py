"""LibreMobileNetV4: BaseModel subclass wiring MobileNetV4 classification into the factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ...training.ddp_spawn import ddp_aware
from ...training.callbacks import TrainCallbacks
from ...postprocess.mobilenetv4 import postprocess as _mnv4_postprocess
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .config import MobileNetV4Config
from .nn import MobileNetV4
from .utils import preprocess_image as _mnv4_preprocess

_TRAIN_DEFAULTS = MobileNetV4Config()


class LibreMobileNetV4(BaseModel):
    """MobileNetV4 image classifier (conv-small/medium/large).

    Examples::

        >>> model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        >>> result = model.predict("cat.jpg")[0]
        >>> result.probs.top1, result.probs.top5

        >>> model = LibreMobileNetV4(size="s")
        >>> model.train(data="imagenette160", epochs=5)
    """

    FAMILY = "mobilenetv4"
    FILENAME_PREFIX = "LibreMobileNetV4"
    INPUT_SIZES = {"s": 224, "m": 224, "l": 256}
    SUPPORTED_TASKS = ("classify",)
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    DEFAULT_TASK = "classify"
    REQUIRE_TASK_SUFFIX = True  # canonical weights are LibreMobileNetV4<size>-cls.pt
    TRAIN_CONFIG = MobileNetV4Config

    # timm eval crop_pct per checkpoint — matches the upstream benchmark preprocessing.
    CROP_PCT = {"s": 0.875, "m": 0.95, "l": 0.95}

    # ---- registry --------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # MobileNetV4-conv signature: timm stem/head + UIB pointwise expansion.
        # Distinct from every detector family (no conv_stem+conv_head+pw_exp combo).
        return (
            "conv_stem.weight" in weights_dict
            and "conv_head.weight" in weights_dict
            and any(k.endswith(".pw_exp.conv.weight") for k in weights_dict)
            # Only claim it if the exact stem+stage-0 signature is a shipped conv
            # size — rejects the hybrid (MQA) and 035/050-width variants.
            and cls.detect_size(weights_dict) is not None
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        if "conv_stem.weight" not in weights_dict or "classifier.weight" not in weights_dict:
            return None
        # Conv-only family: reject the hybrid (Mobile MQA attention) variants.
        if any(".attn." in k or "mqa" in k.lower() for k in weights_dict):
            return None
        # conv s/m/l all end at a 960-wide stage 4 (conv_head in_channels == 960);
        # the 035/050 width variants (and fix_stem keeps their stem at 32) do not.
        if "conv_head.weight" not in weights_dict or int(weights_dict["conv_head.weight"].shape[1]) != 960:
            return None
        stem_chs = int(weights_dict["conv_stem.weight"].shape[0])
        is_er = "blocks.0.0.conv_exp.weight" in weights_dict  # EdgeResidual stage 0 (m/l)
        is_cn = "blocks.0.0.conv.weight" in weights_dict  # ConvBnAct stage 0 (s)
        # small/medium use a 32-ch stem, large a 24-ch stem; 035/050 differ.
        if stem_chs == 24 and is_er:
            return "l"
        if stem_chs == 32 and is_er:
            return "m"
        if stem_chs == 32 and is_cn:
            return "s"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        key = "classifier.weight"
        if key not in weights_dict:
            return None
        return int(weights_dict[key].shape[0])

    # ---- init ------------------------------------------------------------

    def __init__(
        self,
        model_path=None,
        size: str = "s",
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
        return MobileNetV4(size=self.size, num_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "conv_stem": self.model.conv_stem,
            "blocks": self.model.blocks,
            "conv_head": self.model.conv_head,
            "classifier": self.model.classifier,
        }

    def _rebuild_for_new_classes(self, new_nb_classes: int) -> None:
        """Swap the final Linear for a new class count (backbone preserved)."""
        self.nb_classes = new_nb_classes
        self.names = {i: f"class_{i}" for i in range(new_nb_classes)}
        self.model.reset_classifier(new_nb_classes)
        self.model.to(self.device)

    # ---- inference -------------------------------------------------------

    def _get_preprocess_numpy(self):
        # Instance method (not staticmethod) so the per-variant crop_pct is bound.
        # The base contract calls this as ``preprocess_numpy(img, input_size)``
        # (e.g. the exporter's INT8 calibration), so binding crop_pct here keeps
        # medium/large from silently falling back to the 0.875 default.
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
        return _mnv4_preprocess(
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
        return _mnv4_postprocess(
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
        from .trainer import MobileNetV4Trainer

        if imgsz is None:
            imgsz = self.input_size

        trainer = MobileNetV4Trainer(
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
                    "model = LibreMobileNetV4('path/to/last.pt', size='s'); "
                    "model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))

        results = trainer.train()
        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self._load_weights(best_ckpt)
        return results
