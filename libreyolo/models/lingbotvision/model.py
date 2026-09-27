"""LibreYOLO wrapper for LingBot-Vision semantic segmentation.

LingBot-Vision ("Vision Pretraining for Dense Spatial Perception", Fu et al.,
2026, arXiv:2607.05247) is a family of self-supervised ViT backbones trained
with boundary-centric masked modeling, released by Robbyant under Apache-2.0
(code and weights). The backbones ship without task heads; this family pairs
them with a 1x1 dense head following the report's linear-probing protocol and
trains that head (or the full network) on a semantic dataset via the shared
LibreYOLO semantic pipeline.

Sizes s/b/l are distilled from the 1.1B ViT-g/16 teacher; g is the teacher
itself (supported for loading/fine-tuning, no LibreYOLO-hosted weights).

Weights hosted under ``LibreYOLO/`` are Apache-2.0 backbone weights plus a
LibreYOLO-trained head; see the family ``NOTICE`` for attribution and for the
architectural-lineage note (the ViT mirrors the DINOv3 architecture as
published by the upstream Apache-2.0 release).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...tasks import normalize_task
from ...training.callbacks import TrainCallbacks
from ...training.ddp_spawn import ddp_aware
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.serialization import load_trusted_torch_file
from ..base.model import BaseModel
from .nn import SIZE_CONFIGS, LingBotVisionSemanticSegmenter

logger = logging.getLogger(__name__)

_EMBED_DIM_TO_SIZE = {cfg.embed_dim: size for size, cfg in SIZE_CONFIGS.items()}


def _input_size_hw(input_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(input_size, int):
        return input_size, input_size
    if len(input_size) != 2:
        raise ValueError(f"input_size must be int or (height, width), got {input_size!r}")
    return int(input_size[0]), int(input_size[1])


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int | tuple[int, int] = 512,
) -> tuple[np.ndarray, float]:
    """Stretch-resize RGB image to the square canvas as CHW float32 in [0, 1].

    Matches the training/validation pipeline (``semantic_resize_mode =
    "stretch"``): plain resize, ``/255`` only. ImageNet standardization is
    applied inside ``LingBotVisionSemanticSegmenter.forward`` on the raw
    ``[0, 1]`` tensor, so it must not be duplicated here (the RF-DETR semantic
    house convention).
    """
    import cv2

    input_h, input_w = _input_size_hw(input_size)
    resized = cv2.resize(img_rgb_hwc, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    arr = np.ascontiguousarray(resized, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), 1.0


class LibreLingBotVision(BaseModel):
    """LingBot-Vision ViT (s/b/l/g) + dense head for semantic segmentation."""

    FAMILY: ClassVar[str] = "lingbotvision"
    FILENAME_PREFIX: ClassVar[str] = "LibreLingBotVision"
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    WEIGHT_EXT: ClassVar[str] = ".pt"
    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("semantic",)
    DEFAULT_TASK: ClassVar[str] = "semantic"
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = True
    INPUT_SIZES: ClassVar[Dict[str, int]] = {size: 512 for size in SIZE_CONFIGS}
    # g is the 1.1B teacher: loadable and fine-tunable, but no LibreYOLO-hosted
    # checkpoint exists for it (see the module docstring). Kept as data so
    # get_download_url and the error message cannot drift apart.
    UNPUBLISHED_SIZES: ClassVar[Tuple[str, ...]] = ("g",)

    # ViT square canvas: stretch-resize (like LibreDINOv2), patch-16 grid.
    semantic_resize_mode: ClassVar[str] = "stretch"
    semantic_imgsz_divisor: ClassVar[int] = 16
    # The linear-probe recipe uses no photometric jitter; SemanticDataset
    # defaults to 0.5, so declare it explicitly (see LibreSegformer precedent).
    semantic_hsv_prob: ClassVar[float] = 0.0

    # ------------------------------------------------------------------
    # Weight download
    # ------------------------------------------------------------------

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        """Refuse to auto-download the sizes that have no published weights.

        Sizes s/b/l are hosted under ``LibreYOLO/``; g is the 1.1B teacher and
        is not. Left to the base implementation, ``lingbotvision-g`` builds a
        URL for a repo that does not exist and the user gets an HTTP failure
        against ``LibreYOLO/LibreLingBotVisiong-sem``, which reads like an
        outage rather than a checkpoint that was never published. Raise with
        the reason instead.

        The size guard matters: ``download_weights`` asks every registered
        family in turn and takes the first non-None answer, so raising on a
        filename that is not ours would hijack another family's download.
        """
        name = Path(filename).name
        size = cls.detect_size_from_filename(name)
        if size is None:
            return None
        if size in cls.UNPUBLISHED_SIZES:
            published = ", ".join(
                f"{cls.FAMILY}-{s}" for s in SIZE_CONFIGS if s not in cls.UNPUBLISHED_SIZES
            )
            raise FileNotFoundError(
                f"{name}: LibreYOLO publishes no weights for {cls.FAMILY}-{size}. "
                f"Size {size} is the upstream teacher backbone, supported for loading "
                f"and fine-tuning from a local checkpoint but never mirrored under "
                f"LibreYOLO/. Pass a local .pt path, or use one of the published "
                f"sizes: {published}."
            )
        return super().get_download_url(name)

    # ------------------------------------------------------------------
    # Registry / can_load interface
    # ------------------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys = set(weights_dict)
        return (
            "backbone.rope_embed.periods" in keys
            and "backbone.storage_tokens" in keys
            and "predict.weight" in keys
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        cls_token = weights_dict.get("backbone.cls_token")
        if cls_token is None or getattr(cls_token, "ndim", 0) != 3:
            return None
        return _EMBED_DIM_TO_SIZE.get(int(cls_token.shape[-1]))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        head = weights_dict.get("predict.weight")
        if head is not None and getattr(head, "ndim", 0) >= 1:
            return int(head.shape[0])
        return None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_path=None,
        size: str = "s",
        nb_classes: int = 150,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        resolved_task = normalize_task(task) if task is not None else "semantic"
        if resolved_task != "semantic":
            raise ValueError(f"LibreLingBotVision supports only task='semantic'; got {task!r}.")
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **kwargs,
        )
        self.model.eval()
        if self.model_path is not None:
            self._load_weights(str(self.model_path))

    def _init_model(self) -> nn.Module:
        return LingBotVisionSemanticSegmenter(size=self.size, num_classes=self.nb_classes)

    def _rebuild_for_new_size(self, new_size: str) -> None:
        """Re-instantiate the net at the checkpoint's size (s/b/l/g)."""
        self.size = new_size
        self.input_size = self.INPUT_SIZES[new_size]
        self.model = self._init_model()
        self.model.to(self.device)

    def _rebuild_for_new_classes(self, new_nb_classes: int) -> None:
        in_channels = self.model.predict.in_channels
        head = nn.Conv2d(in_channels, new_nb_classes, kernel_size=1)
        self.model._init_head(head)
        self.model.predict = head
        self.model.num_classes = new_nb_classes
        self.nb_classes = new_nb_classes
        self.names = {i: f"class_{i}" for i in range(new_nb_classes)}
        self.model.to(self.device)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {"backbone": self.model.backbone, "head": self.model.predict}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    # ------------------------------------------------------------------
    # Inference pipeline
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: int | tuple[int, int] | None = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_res = input_size if input_size is not None else self._get_input_size()
        divisor = self.semantic_imgsz_divisor
        if any(side % divisor for side in _input_size_hw(effective_res)):
            raise ValueError(
                f"LibreLingBotVision semantic imgsz={effective_res} must be divisible "
                f"by {divisor} (ViT patch grid)."
            )
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        chw, ratio = preprocess_numpy(np.asarray(img), effective_res)
        return torch.from_numpy(chw).unsqueeze(0), img, (orig_w, orig_h), ratio

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess_semantic_logits(
        self,
        output: Any,
        original_size: Tuple[int, int],
        **kwargs,
    ) -> torch.Tensor:
        """Interpolate raw semantic logits to ``original_size``, pre-argmax.

        Shared by ``_postprocess`` and ``BaseModel._predict_augment_semantic``
        (flip TTA), which averages pre-argmax logits across views. Stretch
        resize means the whole canvas is valid content — no crop needed.
        """
        logits = output
        if isinstance(logits, dict):
            logits = logits.get("semantic_logits", logits.get("predictions"))
            if logits is None:
                raise RuntimeError(
                    "LingBot-Vision forward output carries neither "
                    "'semantic_logits' nor 'predictions'; got keys "
                    f"{sorted(output)}"
                )
        orig_w, orig_h = original_size
        return F.interpolate(logits.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        logits = self._postprocess_semantic_logits(output, original_size, **kwargs)
        return {"semantic": logits.argmax(dim=1)[0].cpu()}

    # ------------------------------------------------------------------
    # Weights I/O
    # ------------------------------------------------------------------

    def _strict_loading(self) -> bool:
        return True

    def _validate_loaded_state_dict_for_task(
        self,
        state_dict: dict,
        checkpoint: dict | None = None,
    ) -> None:
        if not self.can_load(state_dict):
            raise RuntimeError(
                "Checkpoint does not look like a LingBot-Vision semantic segmentation model."
            )

    def _load_weights(self, model_path: str | dict[str, Any]) -> None:
        # A checkpoint here is either produced by weights/convert_lingbotvision_
        # weights.py (full LibreYOLO metadata), self-produced by model.train(),
        # or the raw unwrapped state dict that DDP training round-trips through
        # a tempfile — metadata fields are checked when present, never required.
        if isinstance(model_path, str):
            if not Path(model_path).exists():
                from ...utils.download import download_weights

                download_weights(model_path, self.size)
            loaded = load_trusted_torch_file(
                model_path, map_location="cpu", context="LingBot-Vision semantic weights"
            )
        else:
            loaded = model_path

        if not isinstance(loaded, dict):
            raise TypeError("LibreLingBotVision checkpoints must be dictionaries")

        ckpt_family = loaded.get("model_family")
        if isinstance(ckpt_family, str) and ckpt_family and ckpt_family != self.FAMILY:
            raise RuntimeError(
                f"Checkpoint was trained with model_family='{ckpt_family}' "
                f"but is being loaded into '{self.FAMILY}'."
            )

        ckpt_task = loaded.get("task")
        if isinstance(ckpt_task, str) and normalize_task(ckpt_task) != "semantic":
            raise RuntimeError(
                f"Checkpoint was trained for task={normalize_task(ckpt_task)!r}, "
                "but LibreLingBotVision is semantic-only."
            )

        if isinstance(loaded.get("model"), dict):
            state = loaded["model"]
        elif isinstance(loaded.get("state_dict"), dict):
            state = loaded["state_dict"]
        else:
            state = loaded

        # Size determines every layer shape; the state dict is authoritative.
        ckpt_size = self.detect_size(state) or loaded.get("size")
        if ckpt_size is not None and ckpt_size != self.size:
            self._rebuild_for_new_size(str(ckpt_size))

        ckpt_nc = loaded.get("nc") or self.detect_nb_classes(state)
        if ckpt_nc is not None and int(ckpt_nc) != self.nb_classes:
            self._rebuild_for_new_classes(int(ckpt_nc))

        if not self.can_load(state):
            raise RuntimeError(
                "Checkpoint does not look like a LingBot-Vision semantic segmentation model."
            )
        self.model.load_state_dict(state, strict=True)

        ckpt_names = loaded.get("names")
        if ckpt_names is not None:
            self.names = self._sanitize_names(ckpt_names, self.nb_classes)
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        epochs: int = 20,
        batch: int = 16,
        imgsz: Optional[int] = None,
        lr0: Optional[float] = None,
        device: str = "",
        workers: int = 4,
        seed: int = 0,
        project: str = "runs/train",
        name: str = "lingbotvision_exp",
        exist_ok: bool = False,
        resume: bool = False,
        amp: bool = True,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> Dict:
        """Train on a semantic dataset.

        Fine-tuning keeps the report's linear-probe default. Scratch training
        trains the random backbone unless ``freeze_backbone`` is explicit.
        """
        from .trainer import LingBotVisionTrainer

        if self._is_scratch_build():
            kwargs.setdefault("freeze_backbone", False)

        train_kwargs = dict(
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz if imgsz is not None else self.input_size,
            size=self.size,
            num_classes=self.nb_classes,
            device=device,
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            amp=amp,
            **kwargs,
        )
        if lr0 is not None:
            train_kwargs["lr0"] = lr0

        trainer = LingBotVisionTrainer(
            model=self.model,
            wrapper_model=self,
            callbacks=callbacks,
            loggers=loggers,
            **train_kwargs,
        )
        result = trainer.train()
        self._restore_after_training(result)
        return result

    def _restore_after_training(self, result: dict) -> None:
        checkpoint = None
        for key in ("best_checkpoint", "last_checkpoint"):
            path = result.get(key)
            if path and Path(path).exists():
                checkpoint = str(path)
                break
        if checkpoint is not None:
            self.model_path = checkpoint
            self._load_weights(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, format: str = "onnx", *, opset: int = 17, **kwargs) -> str:
        # Dense logits through the shared semantic runtime contract
        # (_SemanticExportWrapper); SDPA needs opset >= 14, use the house 17.
        return super().export(format=format, opset=opset, **kwargs)


__all__ = ["LibreLingBotVision", "preprocess_numpy"]
