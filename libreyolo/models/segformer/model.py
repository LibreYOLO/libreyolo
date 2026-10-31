"""LibreYOLO wrapper for SegFormer semantic segmentation.

SegFormer ("Simple and Efficient Design for Semantic Segmentation with
Transformers", Xie et al., NeurIPS 2021) is a lightweight ViT-style encoder
(MiT, Mix Transformer) paired with an all-MLP decode head. This family covers
all six standard sizes, b0 through b5.

Weights: the pretrained ADE20K checkpoints derive from NVIDIA's SegFormer
release, which is distributed under the NVIDIA Source Code License. That
license permits redistribution (with the license and attribution attached) but
restricts *use* to non-commercial research or evaluation. LibreYOLO therefore
hosts converted checkpoints and prints a non-commercial notice before each
auto-download (see ``get_download_notice``), exactly as it does for the
VisDrone research-preview weights. The architecture and LibreYOLO itself stay
permissively licensed; only these pretrained weights are restricted. For
unrestricted use, train from scratch via ``LibreSegformer(...).train(...)``.
See ``NOTICE`` in this directory for the full licensing rationale.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...tasks import normalize_task
from ...training.callbacks import TrainCallbacks
from ...training.config import SegformerConfig
from ...training.ddp_spawn import ddp_aware
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.serialization import load_trusted_torch_file
from ..base.model import BaseModel
from .nn import SIZE_CONFIGS, LibreSegformerNet

logger = logging.getLogger(__name__)


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
    """Letterbox RGB image to SegFormer's canvas as CHW float32 in ``[0, 1]``.

    Matches the training/validation pipeline: ``/255`` only. ImageNet
    standardization (the distribution the MiT encoder was pretrained and
    fine-tuned on) is applied inside ``LibreSegformerNet.forward`` on the raw
    ``[0, 1]`` tensor, so it must not be duplicated here (see the family docs
    and ``nn.py``; this mirrors the RF-DETR semantic house convention).
    """
    orig_h, orig_w = img_rgb_hwc.shape[:2]
    input_h, input_w = _input_size_hw(input_size)
    ratio = min(input_h / orig_h, input_w / orig_w)
    new_h = max(int(orig_h * ratio), 1)
    new_w = max(int(orig_w * ratio), 1)

    resized = cv2.resize(img_rgb_hwc, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    arr = np.ascontiguousarray(padded, dtype=np.float32) / 255.0
    chw = arr.transpose(2, 0, 1)
    return np.ascontiguousarray(chw, dtype=np.float32), ratio


class LibreSegformer(BaseModel):
    """SegFormer b0-b5 family for dense semantic segmentation (ADE20K, 150 classes)."""

    FAMILY: ClassVar[str] = "segformer"
    FILENAME_PREFIX: ClassVar[str] = "LibreSegformer"
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    WEIGHT_EXT: ClassVar[str] = ".pt"
    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("semantic",)
    DEFAULT_TASK: ClassVar[str] = "semantic"
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = True
    # b0-b4 are fine-tuned on ADE20K at 512; b5 is the one trained at 640.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {**{size: 512 for size in SIZE_CONFIGS}, "b5": 640}
    TRAIN_CONFIG: ClassVar[type[SegformerConfig]] = SegformerConfig

    # Fine-tune recipe (SegFormer paper / mmsegmentation ADE20K config):
    #  - resize_crop: training resizes the short side to imgsz*ratio and random-
    #    crops imgsz^2 (dense full-res crops, cat_max_ratio=0.75), matching mmseg
    #    rather than the fit-long-side+pad letterbox. Validation/inference still
    #    letterbox to a fixed imgsz^2 canvas (mIoU scored at that resolution).
    #  - the MiT encoder is pretrained with ImageNet mean/std; that
    #    standardization is applied inside LibreSegformerNet.forward on the raw
    #    [0, 1] tensor, so the dataset/validator/preprocess feed [0, 1] only and
    #    never pre-normalize (matching the RF-DETR semantic house convention).
    #    Scale jitter spans 0.5..2.0.
    # Read by SegformerTrainer._setup_semantic_data and SemanticValidator.
    semantic_resize_mode: ClassVar[str] = "resize_crop"
    semantic_imgsz_divisor: ClassVar[int] = 32
    semantic_scale_jitter: ClassVar[Tuple[float, float]] = (0.5, 2.0)
    # The reference ADE20K recipe uses no photometric jitter. SemanticDataset
    # defaults to 0.5, so this has to be declared explicitly or training quietly
    # runs a recipe nobody asked for.
    semantic_hsv_prob: ClassVar[float] = 0.0

    # ------------------------------------------------------------------
    # Registry / can_load interface
    # ------------------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys = set(weights_dict)
        return (
            "decode_head.linear_fuse.weight" in keys
            and "decode_head.classifier.weight" in keys
            and "encoder.stages.0.patch_embeddings.proj.weight" in keys
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        stem = weights_dict.get("encoder.stages.0.patch_embeddings.proj.weight")
        if stem is None or getattr(stem, "ndim", 0) < 1:
            return None
        embed_dim0 = int(stem.shape[0])
        if embed_dim0 == 32:
            return "b0"
        if embed_dim0 != 64:
            return None

        depth_by_size = {2: "b1", 6: "b2", 18: "b3", 27: "b4", 40: "b5"}
        stage2_prefix = "encoder.stages.2.blocks."
        block_indices = set()
        for key in weights_dict:
            if key.startswith(stage2_prefix):
                remainder = key[len(stage2_prefix) :]
                idx_str = remainder.split(".", 1)[0]
                if idx_str.isdigit():
                    block_indices.add(int(idx_str))
        depth = len(block_indices)
        return depth_by_size.get(depth)

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        head = weights_dict.get("decode_head.classifier.weight")
        if head is not None and getattr(head, "ndim", 0) >= 1:
            return int(head.shape[0])
        return None

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> Optional[dict]:
        """Map a raw upstream SegFormer state dict onto LibreSegformer keys.

        The module tree matches the Apache-2.0 reference exactly, so the only
        difference is the encoder prefix (upstream ``segformer.``, here
        ``encoder.``). Learned parameters are unchanged. Returns None when the
        input is not an upstream SegFormer checkpoint.
        """
        keys = set(state_dict)
        if not any(k.startswith("segformer.") for k in keys):
            return None
        if "decode_head.classifier.weight" not in keys:
            return None
        converted = {}
        for key, value in state_dict.items():
            if key.startswith("segformer."):
                converted["encoder." + key[len("segformer.") :]] = value
            elif key.startswith("decode_head."):
                converted[key] = value
            else:
                return None  # unknown layout; refuse rather than half-convert
        return converted

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> Optional[str]:
        return (
            f"{Path(filename).name} derives from NVIDIA's SegFormer checkpoints, "
            "released under the NVIDIA Source Code License: these weights are for "
            "NON-COMMERCIAL use (research or evaluation) only and are NOT covered "
            "by LibreYOLO's permissive license. The architecture and LibreYOLO "
            "itself remain freely usable; only these pretrained weights are "
            "restricted. Train from scratch for unrestricted use."
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_path=None,
        size: str = "b0",
        nb_classes: int = 150,
        device: str = "auto",
        task: str | None = None,
        pretrained_encoder: Optional[str] = None,
        **kwargs,
    ) -> None:
        resolved_task = normalize_task(task) if task is not None else "semantic"
        if resolved_task != "semantic":
            raise ValueError(f"LibreSegformer supports only task='semantic'; got {task!r}.")
        if model_path is not None and pretrained_encoder is not None:
            raise ValueError(
                "Pass only one of model_path= (a full checkpoint) or "
                "pretrained_encoder= (an encoder-only ImageNet pretraining "
                "checkpoint) -- combining them is ambiguous."
            )
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
        elif pretrained_encoder is not None:
            self._load_pretrained_encoder(pretrained_encoder)

    def _init_model(self) -> nn.Module:
        return LibreSegformerNet(size=self.size, num_classes=self.nb_classes)

    def _rebuild_for_new_size(self, new_size: str) -> None:
        """Re-instantiate the net at the checkpoint's size (b0-b5).

        Size determines every layer shape, so this must run before any other
        rebuild. Without it, loading e.g. LibreSegformerb1-sem.pt into the
        default b0 model dies on a wall of shape mismatches.
        """
        self.size = new_size
        self.input_size = self.INPUT_SIZES[new_size]
        self.model = self._init_model()
        self.model.to(self.device)

    def _rebuild_for_new_classes(self, new_nb_classes: int) -> None:
        decode_head = self.model.decode_head
        in_channels = decode_head.classifier.in_channels
        classifier = nn.Conv2d(in_channels, new_nb_classes, kernel_size=1)
        # A fresh nn.Conv2d carries torch's default init, not the reference one.
        # This is the fine-tuning path (pretrained encoder, new class head), so
        # the head would start ~2x too wide with non-zero bias.
        self.model._init_weights(classifier)
        decode_head.classifier = classifier
        self.model.num_classes = new_nb_classes
        self.nb_classes = new_nb_classes
        self.names = {i: f"class_{i}" for i in range(new_nb_classes)}
        self.model.to(self.device)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {"encoder": self.model.encoder, "decode_head": self.model.decode_head}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

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
                f"LibreSegformer semantic imgsz={effective_res} must be divisible "
                f"by {divisor} (encoder stride product)."
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
        ratio: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Interpolate raw semantic logits to ``original_size``, pre-argmax.

        Shared by ``_postprocess`` (single-view predict/val) and
        ``BaseModel._predict_augment_semantic`` (flip TTA), which needs the
        pre-argmax logits to average across augmented views.
        """
        logits = output
        if isinstance(logits, dict):
            logits = logits.get("semantic_logits", logits.get("predictions"))
        orig_w, orig_h = original_size
        input_size = kwargs.get("input_size", self._get_input_size())
        input_h, input_w = _input_size_hw(input_size)
        scale_y = logits.shape[-2] / input_h
        scale_x = logits.shape[-1] / input_w
        valid_h = min(logits.shape[-2], max(int(round(orig_h * ratio * scale_y)), 1))
        valid_w = min(logits.shape[-1], max(int(round(orig_w * ratio * scale_x)), 1))
        logits = logits[..., :valid_h, :valid_w]
        return F.interpolate(logits.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False)

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
        logits = self._postprocess_semantic_logits(output, original_size, ratio=ratio, **kwargs)
        return {"semantic": logits.argmax(dim=1)[0].cpu()}

    def _strict_loading(self) -> bool:
        return True

    def _validate_loaded_state_dict_for_task(
        self,
        state_dict: dict,
        checkpoint: dict | None = None,
    ) -> None:
        if not self.can_load(state_dict):
            raise RuntimeError("Checkpoint does not look like a SegFormer semantic segmentation model.")

    def _load_weights(self, model_path: str | dict[str, Any]) -> None:
        # A checkpoint here is either converted from upstream
        # (weights/convert_segformer_weights.py, full LibreYOLO metadata),
        # self-produced by model.train(), or the raw unwrapped state dict that
        # DDP training round-trips through a tempfile (libreyolo/training/
        # ddp_spawn.py). All three must load cleanly, so — like LibreDINOv2 —
        # this only checks metadata fields when present, never requires them.
        if isinstance(model_path, str):
            if not Path(model_path).exists():
                from ...utils.download import download_weights

                download_weights(model_path, self.size)
            loaded = load_trusted_torch_file(
                model_path, map_location="cpu", context="SegFormer semantic weights"
            )
        else:
            loaded = model_path

        if not isinstance(loaded, dict):
            raise TypeError("LibreSegformer checkpoints must be dictionaries")

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
                "but LibreSegformer is semantic-only."
            )

        if isinstance(loaded.get("model"), dict):
            state = loaded["model"]
        elif isinstance(loaded.get("state_dict"), dict):
            state = loaded["state_dict"]
        else:
            state = loaded

        # Size determines every layer shape, so resolve it before any other
        # rebuild. The state dict is authoritative; the metadata is a hint.
        ckpt_size = self.detect_size(state) or loaded.get("size")
        if ckpt_size is not None and ckpt_size != self.size:
            self._rebuild_for_new_size(str(ckpt_size))

        ckpt_nc = loaded.get("nc") or self.detect_nb_classes(state)
        if ckpt_nc is not None and int(ckpt_nc) != self.nb_classes:
            self._rebuild_for_new_classes(int(ckpt_nc))

        if not self.can_load(state):
            raise RuntimeError("Checkpoint does not look like a SegFormer semantic segmentation model.")
        self.model.load_state_dict(state, strict=True)

        ckpt_names = loaded.get("names")
        if ckpt_names is not None:
            self.names = self._sanitize_names(ckpt_names, self.nb_classes)
        self.model.to(self.device).eval()

    def _load_pretrained_encoder(self, path: str) -> None:
        """Partial-load an encoder-only checkpoint (e.g. one produced by your own
        MiT pretraining run). Only ``self.model.encoder`` is populated; the decode
        head stays at random init, ready for a fresh semantic fine-tune.

        Most users want ``model_path=`` instead: that loads a full pretrained
        checkpoint and ``nb_classes=`` re-heads it for a new dataset.
        """
        payload = load_trusted_torch_file(path, map_location="cpu", context="SegFormer pretrained encoder")
        if not isinstance(payload, dict) or "encoder" not in payload:
            raise ValueError(f"{path} is not a valid pretrained-encoder checkpoint (missing 'encoder' key).")
        ckpt_size = payload.get("size")
        if ckpt_size is not None and ckpt_size != self.size:
            raise ValueError(
                f"Pretrained encoder checkpoint was produced for size={ckpt_size!r} "
                f"but this model is size={self.size!r}."
            )
        self.model.encoder.load_state_dict(payload["encoder"], strict=True)
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Training — fine-tune the pretrained ADE20K checkpoints on a new dataset,
    # or train from scratch (the only route to weights without the upstream
    # non-commercial restriction).
    # ------------------------------------------------------------------

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        epochs: int = 160,
        batch: int = 8,
        imgsz: Optional[int] = None,
        lr0: Optional[float] = None,
        device: str = "",
        workers: int = 4,
        seed: int = 0,
        project: str = "runs/train",
        name: str = "segformer_exp",
        exist_ok: bool = False,
        resume: bool = False,
        amp: bool = True,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> Dict:
        """Fine-tune LibreSegformer, or train it from scratch.

        Pass ``model_path=`` a pretrained checkpoint to fine-tune it (``nb_classes``
        re-heads it for a new dataset). Training from scratch produces weights that
        carry none of the upstream non-commercial restriction.
        """
        from .trainer import SegformerTrainer

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

        trainer = SegformerTrainer(
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

    def export(self, format: str = "onnx", **kwargs) -> str:
        return super().export(format=format, **kwargs)


__all__ = ["LibreSegformer", "preprocess_numpy"]
