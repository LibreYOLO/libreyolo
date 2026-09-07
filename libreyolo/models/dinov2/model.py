"""LibreDINOv2 — semantic, classification, and whole-image embedding.

LibreDINOv2 is the honest home for the tasks that are really "DINOv2 backbone +
a task head" rather than the RF-DETR detector:

* ``semantic`` — reuses ``RFDETRSemanticSegmenter`` (DINOv2 encoder +
  multi-scale projector pyramid + 1x1 dense head).
* ``classify`` — reuses ``RFDETRClassifier`` (DINOv2 encoder + projector +
  global-average-pool + linear head). The DETR decoder/queries are never built;
  this is a linear probe on the pretrained DINOv2 backbone, not the RF-DETR
  detector, so it does not belong in the RF-DETR family.
* ``embed`` — bypasses every task head and projector, returning the final
  normalized DINOv2 CLS token as one whole-image row.

All expose themselves as the ``dinov2`` family: checkpoints save
``model_family="dinov2"`` and the factory routes ``backbone.*`` plus
``predict.*`` (semantic) / ``linear.*`` (classify) keys here. LibreRFDETR no
longer registers ``semantic`` or ``classify``.

Training delegates to :class:`DINOv2Trainer`, which inherits the semantic and
classify branches from ``RFDETRTrainer`` and overrides only the family metadata.

Backward compatibility: pre-split semantic checkpoints carry
``model_family="rfdetr"`` / ``task="semantic"`` and are still accepted.
Classify is intentionally a clean break — legacy ``LibreRFDETR*-cls``
checkpoints are not loaded (retrain/republish as ``LibreDINOv2*-cls``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ...training.callbacks import TrainCallbacks
from ...tasks import normalize_task
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.serialization import load_trusted_torch_file
from ..base.model import BaseModel
from .config import DINOv2Config
from libreyolo.training.ddp_spawn import ddp_aware

logger = logging.getLogger(__name__)

_TRAIN_DEFAULTS = DINOv2Config()


class _DINOv2ModelWrapper(nn.Module):
    """Thin wrapper making ``RFDETRSemanticSegmenter`` compatible with
    ``RFDETRTrainer``'s semantic training path.

    ``RFDETRTrainer`` inspects ``self.model`` (= ``wrapper_model.model``) for:
    * ``self.model.model is None``  → activates the classify/semantic branch in
      ``get_freeze_groups`` (looks for ``self.model.segmenter``).
    * ``self.model.segmenter``      → the actual segmenter module.
    * ``self.model.patch_size``     → for transform block-size validation.
    * ``self.model.num_windows``    → ditto.
    * ``self.model.nb_classes``     → class count synced from config.
    * ``self.model(imgs, targets)`` → returns the loss dict in training mode.
    * ``self.model.state_dict()``   → delegated to segmenter.
    * ``self.model.load_state_dict(...)`` → delegated to segmenter.
    """

    def __init__(
        self,
        config: str,
        nb_classes: int,
        device: str = "cpu",
        load_dinov2_weights: bool = True,
    ) -> None:
        super().__init__()
        # Lazy import so the outer module-level import stays rfdetr-free.
        from ..rfdetr.nn import RFDETRSemanticSegmenter

        # ``model = None`` signals RFDETRTrainer.get_freeze_groups that this is
        # the dense-head path (uses the segmenter freeze layout).
        self.model: nn.Module | None = None
        self.segmenter = RFDETRSemanticSegmenter(
            config=config,
            nb_classes=nb_classes,
            device=device,
            load_dinov2_weights=load_dinov2_weights,
        )
        self.nb_classes = nb_classes
        self.resolution = self.segmenter.resolution
        self.hidden_dim = self.segmenter.hidden_dim
        self.patch_size = self.segmenter.patch_size
        self.num_windows = self.segmenter.num_windows

    def forward(self, x: torch.Tensor, targets=None):
        return self.segmenter(x, targets=targets)

    def state_dict(self, *args, **kwargs):
        return self.segmenter.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: dict, strict: bool = False):
        # Strip outer checkpoint wrappers; the segmenter expects its own keys.
        if isinstance(state_dict, dict):
            if "model" in state_dict and isinstance(state_dict["model"], dict):
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict and isinstance(
                state_dict["state_dict"], dict
            ):
                state_dict = state_dict["state_dict"]
        return self.segmenter.load_state_dict(state_dict, strict=strict)


class _DINOv2ClassifierWrapper(nn.Module):
    """Exposes ``RFDETRClassifier`` to ``RFDETRTrainer``'s classify path.

    ``RFDETRTrainer.get_freeze_groups`` activates the classify branch when
    ``self.model.model is None`` and reads ``self.model.classifier`` (backbone +
    linear head). Mirrors :class:`_DINOv2ModelWrapper` but wraps the linear-probe
    classifier instead of the dense segmenter. The DETR decoder is never built —
    this is a DINOv2 backbone + global-average-pool + linear head.
    """

    def __init__(
        self,
        config: str,
        nb_classes: int,
        device: str = "cpu",
        load_dinov2_weights: bool = True,
    ) -> None:
        super().__init__()
        # Lazy import so the outer module-level import stays rfdetr-free.
        from ..rfdetr.nn import RFDETRClassifier

        # ``model = None`` signals RFDETRTrainer.get_freeze_groups that this is
        # the linear-head path (uses the classifier freeze layout).
        self.model: nn.Module | None = None
        self.classifier = RFDETRClassifier(
            config=config,
            nb_classes=nb_classes,
            device=device,
            load_dinov2_weights=load_dinov2_weights,
        )
        self.nb_classes = nb_classes
        self.resolution = self.classifier.resolution
        self.hidden_dim = self.classifier.hidden_dim
        self.patch_size = self.classifier.patch_size
        self.num_windows = self.classifier.num_windows

    def forward(self, x: torch.Tensor, targets=None):
        return self.classifier(x, targets=targets)

    def state_dict(self, *args, **kwargs):
        return self.classifier.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: dict, strict: bool = False):
        if isinstance(state_dict, dict):
            if "model" in state_dict and isinstance(state_dict["model"], dict):
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict and isinstance(
                state_dict["state_dict"], dict
            ):
                state_dict = state_dict["state_dict"]
        return self.classifier.load_state_dict(state_dict, strict=strict)


class _DINOv2EmbedderWrapper(nn.Module):
    """DINOv2 backbone-only wrapper returning the final CLS token."""

    def __init__(
        self,
        config: str,
        device: str = "cpu",
        load_dinov2_weights: bool = True,
    ) -> None:
        super().__init__()
        from ..rfdetr.nn import RFDETRClassifier

        classifier = RFDETRClassifier(
            config=config,
            nb_classes=1,
            device=device,
            dropout=0.0,
            load_dinov2_weights=load_dinov2_weights,
        )
        # Retain only the encoder. The classifier backbone's projector is not
        # used for CLS extraction; keeping its random parameters would waste
        # memory and make state-dict fingerprints differ between otherwise
        # identical zero-config embedders.
        self.backbone = nn.Module()
        self.backbone.encoder = classifier.backbone.encoder
        dino = self.backbone.encoder.encoder
        self.embedding_dim = int(dino.config.hidden_size)
        self.resolution = classifier.resolution
        self.patch_size = classifier.patch_size
        self.num_windows = classifier.num_windows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The backbone normally reshapes selected patch stages for its
        # projector. Embedding bypasses that projector and returns the final
        # normalized CLS token.
        dino = self.backbone.encoder.encoder
        hidden = dino.embeddings(x)
        encoded = dino.encoder(
            hidden,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=False,
        )
        sequence = dino.layernorm(encoded[0])
        return sequence[:, 0, :]

    def load_state_dict(self, state_dict: dict, strict: bool = False):
        if isinstance(state_dict, dict):
            if "model" in state_dict and isinstance(state_dict["model"], dict):
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict and isinstance(
                state_dict["state_dict"], dict
            ):
                state_dict = state_dict["state_dict"]
        backbone_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
        return super().load_state_dict(backbone_state, strict=strict)


# ---------------------------------------------------------------------------
# LibreDINOv2
# ---------------------------------------------------------------------------


class LibreDINOv2(BaseModel):
    """DINOv2 backbone for semantic, classification, and embedding tasks.

    Semantic inputs are resized to a square divisible by the DINOv2 patch grid
    (14). Classification and embedding use the 224-pixel classification
    transform; embedding bypasses the projector and task heads.

    Args:
        model_path: Path to a ``.pt`` checkpoint, a pre-loaded state dict, or
                    ``None`` to build from the DINOv2 pretrained backbone.
        size: Size variant — ``"n"``, ``"s"``, ``"m"``, or ``"l"`` — controls
              the RF-DETR projector width (not the backbone; all sizes share
              the same DINOv2-S encoder).
        nb_classes: Number of semantic classes.
        device: Torch device string or ``"auto"``.
        task: ``"semantic"``, ``"classify"``, or ``"embed"``. The default is
            ``"semantic"``.
    """

    FAMILY: ClassVar[str] = "dinov2"
    FILENAME_PREFIX: ClassVar[str] = "LibreDINOv2"
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    WEIGHT_EXT: ClassVar[str] = ".pt"

    # Semantic runs at the DINOv2-native 518 (37 × 14); classify runs at 224.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"n": 518, "s": 518, "m": 518, "l": 518}
    TASK_INPUT_SIZES: ClassVar[Dict[str, Dict[str, int]]] = {
        "classify": {"n": 224, "s": 224, "m": 224, "l": 224},
        "embed": {"n": 224, "s": 224, "m": 224, "l": 224},
    }
    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = (
        "semantic",
        "classify",
        "embed",
    )
    WEIGHT_TASKS: ClassVar[Tuple[str, ...]] = ("semantic", "classify")
    DEFAULT_TASK: ClassVar[str] = "semantic"

    TRAIN_CONFIG: ClassVar[type] = DINOv2Config

    # Stretch-resize and DINOv2 patch divisibility.
    semantic_resize_mode: ClassVar[str] = "stretch"
    semantic_imgsz_divisor: ClassVar[int] = 14

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Match LibreDINOv2 checkpoints: backbone + dense head (semantic) or
        backbone + linear head (classify).

        Semantic also accepts old ``model_family="rfdetr"`` checkpoints for
        backward compatibility; classify is a clean break (legacy
        ``LibreRFDETR*-cls`` checkpoints are not matched as new dinov2 weights).
        """
        if not any(k.startswith("backbone.") for k in weights_dict):
            return False
        # LibreLingBotVision checkpoints also pair backbone.* with a
        # predict.weight dense head; their RoPE buffer is the discriminator
        # (DINOv2 backbones carry learned position embeddings, never RoPE).
        if "backbone.rope_embed.periods" in weights_dict:
            return False
        is_semantic = "predict.weight" in weights_dict
        is_classify = (
            "linear.weight" in weights_dict and "predict.weight" not in weights_dict
        )
        return is_semantic or is_classify

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        # All semantic sizes share the same DINOv2-S backbone; size must be
        # read from the checkpoint metadata or filename.
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        pw = weights_dict.get("predict.weight")
        if pw is not None:
            return int(pw.shape[0])
        lw = weights_dict.get("linear.weight")
        if lw is not None:
            return int(lw.shape[0])
        return None

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        self,
        model_path=None,
        size: str = "s",
        nb_classes: int = 19,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        resolved_task = normalize_task(task) if task is not None else "semantic"
        if resolved_task not in self.SUPPORTED_TASKS:
            raise ValueError(
                "LibreDINOv2 supports task in ('semantic', 'classify', 'embed'); "
                f"got {task!r}."
            )

        if isinstance(model_path, dict) and not model_path:
            weight_source = None
        elif model_path is None:
            # No default checkpoint: build from DINOv2 pretrained backbone.
            weight_source = None
        elif isinstance(model_path, str):
            weight_source = self._resolve_weights_path(model_path)
        else:
            weight_source = model_path  # pre-loaded state dict

        self._weight_source = weight_source

        # Detect size from checkpoint when not given explicitly.
        if size is None and weight_source is not None:
            detected = self._detect_size_from_source(weight_source)
            size = detected or "s"
        if size is None:
            size = "s"

        self._model_num_classes = nb_classes

        # Detect nb_classes from checkpoint state dict.
        if isinstance(weight_source, dict):
            state = self._extract_state(weight_source)
            detected_nc = self.detect_nb_classes(state)
            if detected_nc is not None:
                self._model_num_classes = detected_nc

        super().__init__(
            model_path=None,
            size=size,
            nb_classes=self._model_num_classes,
            device=device,
            task=resolved_task,
            **kwargs,
        )

        if weight_source is not None:
            self._load_weights(weight_source)
            if isinstance(weight_source, str) and Path(weight_source).is_file():
                self.model_path = str(weight_source)
        if self.task == "embed":
            self.names = {}
        self.model.eval()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_state(ckpt: dict) -> dict:
        """Strip LibreYOLO checkpoint wrappers to get raw tensor keys."""
        for key in ("model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt

    @classmethod
    def _detect_size_from_source(cls, source: str | dict | None) -> Optional[str]:
        if source is None:
            return None
        if isinstance(source, str):
            try:
                ckpt = load_trusted_torch_file(
                    source, map_location="cpu", context="DINOv2 size detection"
                )
            except Exception:
                return cls.detect_size_from_filename(source)
        else:
            ckpt = source
        if isinstance(ckpt, dict) and isinstance(ckpt.get("size"), str):
            return ckpt["size"]
        if isinstance(ckpt, dict):
            detected = cls.detect_size(cls._extract_state(ckpt))
            if detected:
                return detected
        if isinstance(source, str):
            return cls.detect_size_from_filename(source)
        return None

    # =========================================================================
    # Model lifecycle
    # =========================================================================

    def _init_model(self) -> nn.Module:
        load_dinov2_weights = not self._is_scratch_build()
        if self.task == "classify":
            return _DINOv2ClassifierWrapper(
                config=self.size,
                nb_classes=self._model_num_classes,
                device=str(self.device),
                load_dinov2_weights=load_dinov2_weights,
            )
        if self.task == "embed":
            return _DINOv2EmbedderWrapper(
                config=self.size,
                device=str(self.device),
                load_dinov2_weights=load_dinov2_weights,
            )
        return _DINOv2ModelWrapper(
            config=self.size,
            nb_classes=self._model_num_classes,
            device=str(self.device),
            load_dinov2_weights=load_dinov2_weights,
        )

    def _rebuild_for_new_classes(self, new_nc: int) -> None:
        if self.task == "embed":
            raise NotImplementedError(
                "DINOv2 embedding has no class-dependent head to rebuild."
            )
        self.nb_classes = new_nc
        self._model_num_classes = new_nc
        if self.task == "classify":
            classifier = self.model.classifier
            in_features = classifier.linear.in_features
            classifier.linear = nn.Linear(in_features, new_nc)
            classifier.nb_classes = new_nc
        else:
            segmenter = self.model.segmenter
            in_channels = segmenter.predict.in_channels
            segmenter.predict = nn.Conv2d(in_channels, new_nc, 1)
            segmenter.nb_classes = new_nc
        self.model.nb_classes = new_nc
        self.model.to(self.device)
        self.names = {i: f"class_{i}" for i in range(new_nc)}

    def _strict_loading(self) -> bool:
        return False

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        if self.task == "embed":
            return {"backbone": self.model.backbone}
        core = (
            self.model.classifier if self.task == "classify" else self.model.segmenter
        )
        layers: Dict[str, nn.Module] = {}
        backbone = getattr(core, "backbone", None)
        if backbone is not None:
            layers["backbone"] = backbone
        head = (
            getattr(core, "linear", None)
            if self.task == "classify"
            else getattr(core, "predict", None)
        )
        if head is not None:
            layers["head"] = head
        return layers

    @staticmethod
    def _get_preprocess_numpy():
        """Return a stretch-resize + [0,1] normalisation callable."""
        import cv2
        import numpy as _np

        def _preprocess_numpy(img_rgb_hwc, input_size=518):
            h = input_size if isinstance(input_size, int) else input_size[0]
            w = input_size if isinstance(input_size, int) else input_size[1]
            resized = cv2.resize(img_rgb_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
            arr = _np.ascontiguousarray(resized, dtype=_np.float32) / 255.0
            return arr.transpose(2, 0, 1), 1.0

        return _preprocess_numpy

    # =========================================================================
    # Inference pipeline
    # =========================================================================

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        """Stretch-resize to square; the model applies ImageNet norm."""
        effective_res = input_size if input_size is not None else self.input_size
        if self.task in ("classify", "embed"):
            from ...data.classify_dataset import build_classify_transforms

            img = ImageLoader.load(image, color_format=color_format)
            orig_w, orig_h = img.size
            transform = build_classify_transforms(effective_res, augment=False)
            return transform(img).unsqueeze(0), img, (orig_w, orig_h), 1.0
        if effective_res % self.semantic_imgsz_divisor:
            raise ValueError(
                f"LibreDINOv2 semantic imgsz={effective_res} must be divisible "
                f"by {self.semantic_imgsz_divisor} (DINOv2 patch grid)."
            )
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        resized = img.resize((effective_res, effective_res), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return img_tensor, img, (orig_w, orig_h), 1.0

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        if self.task == "embed":
            return self._postprocess_embeddings(
                output,
                gallery=kwargs.get("gallery"),
                threshold=kwargs.get("threshold"),
            )
        if self.task == "classify":
            logits = output
            if isinstance(logits, dict):
                logits = logits.get("logits", logits.get("predictions"))
            probs = torch.softmax(logits.float(), dim=1)[0]
            return {"probs": probs.cpu()}
        logits = self._postprocess_semantic_logits(output, original_size, **kwargs)
        return {"semantic": logits.argmax(dim=1)[0].cpu()}

    def _postprocess_semantic_logits(
        self,
        output: Any,
        original_size: Tuple[int, int],
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
        return torch.nn.functional.interpolate(
            logits.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

    # =========================================================================
    # Weights I/O
    # =========================================================================

    def _load_weights(self, model_path: str | dict[str, Any]) -> None:
        """Load a LibreDINOv2 checkpoint.

        Semantic also accepts legacy ``model_family="rfdetr"`` checkpoints;
        classify is a clean break (see :meth:`_load_classify_weights`).
        """
        if isinstance(model_path, str):
            if not Path(model_path).exists():
                from ...utils.download import download_weights

                download_weights(model_path, self.size)
            loaded = load_trusted_torch_file(
                model_path,
                map_location="cpu",
                context=f"DINOv2 {self.task} weights",
            )
        else:
            loaded = model_path

        if not isinstance(loaded, dict):
            raise TypeError("LibreDINOv2 checkpoints must be dictionaries")

        if self.task == "embed":
            return self._load_embed_weights(loaded)
        if self.task == "classify":
            return self._load_classify_weights(loaded)

        # Accept both model_family="dinov2" (new) and "rfdetr" (legacy semantic).
        ckpt_family = loaded.get("model_family", "")
        if ckpt_family and ckpt_family not in (self.FAMILY, "rfdetr"):
            raise RuntimeError(
                f"Checkpoint was trained with model_family='{ckpt_family}' "
                f"but is being loaded into '{self.FAMILY}'."
            )

        ckpt_task = loaded.get("task")
        if isinstance(ckpt_task, str) and normalize_task(ckpt_task) != "semantic":
            raise RuntimeError(
                f"Checkpoint was trained for task={normalize_task(ckpt_task)!r}, "
                "but is being loaded into a LibreDINOv2 semantic model."
            )

        # Detect and rebuild for checkpoint class count.
        ckpt_nc = loaded.get("nc")
        if ckpt_nc is None:
            names = loaded.get("names")
            ckpt_nc = len(names) if names else None
        if ckpt_nc is None:
            state = self._extract_state(loaded)
            pw = state.get("predict.weight")
            ckpt_nc = int(pw.shape[0]) if pw is not None else None
        if ckpt_nc is not None and ckpt_nc != self.nb_classes:
            self._rebuild_for_new_classes(int(ckpt_nc))

        result = self.model.load_state_dict(loaded, strict=False)
        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        if any(k.startswith("predict.") for k in missing) or any(
            ("class_embed" in k or "transformer" in k or "query" in k)
            for k in unexpected
        ):
            raise RuntimeError(
                "Checkpoint does not look like a LibreDINOv2 semantic model "
                "(weights do not match backbone + dense decoder)."
            )

        ckpt_names = loaded.get("names")
        if ckpt_names is not None:
            self.names = self._sanitize_names(ckpt_names, self.nb_classes)
        self.model.to(self.device)

    def _load_embed_weights(self, loaded: dict) -> None:
        """Load only the DINOv2 backbone from a semantic/classify artifact."""
        checkpoint_family = loaded.get("model_family", "")
        if checkpoint_family and checkpoint_family not in (self.FAMILY, "rfdetr"):
            raise RuntimeError(
                f"Checkpoint was trained with model_family={checkpoint_family!r}, "
                "but DINOv2 embedding requires a DINOv2-family backbone."
            )
        checkpoint_task = loaded.get("task")
        if (
            isinstance(checkpoint_task, str)
            and normalize_task(checkpoint_task) not in self.SUPPORTED_TASKS
        ):
            raise RuntimeError(
                f"Checkpoint task={normalize_task(checkpoint_task)!r} is not "
                "compatible with LibreDINOv2 embedding."
            )

        state = self._extract_state(loaded)
        if not any(key.startswith("backbone.") for key in state):
            raise RuntimeError("Checkpoint does not contain a LibreDINOv2 backbone.")
        result = self.model.load_state_dict(state, strict=False)
        missing = list(getattr(result, "missing_keys", []) or [])
        critical_prefixes = (
            "backbone.encoder.encoder.embeddings.",
            "backbone.encoder.encoder.encoder.",
            "backbone.encoder.encoder.layernorm.",
        )
        if any(key.startswith(critical_prefixes) for key in missing):
            raise RuntimeError(
                "Checkpoint is missing required DINOv2 encoder weights for "
                "task='embed'."
            )
        self.model.to(self.device)
        self.names = {}

    def _load_classify_weights(self, loaded: dict) -> None:
        """Load a LibreDINOv2 classification checkpoint.

        Clean break: legacy ``LibreRFDETR*-cls`` checkpoints
        (``model_family="rfdetr"``) are rejected with a clear message.
        """
        ckpt_family = loaded.get("model_family", "")
        if ckpt_family and ckpt_family != self.FAMILY:
            raise RuntimeError(
                f"Checkpoint was trained with model_family='{ckpt_family}', but "
                f"LibreDINOv2 classification only loads model_family='{self.FAMILY}' "
                "checkpoints. Legacy LibreRFDETR*-cls weights are not supported — "
                "retrain/republish as LibreDINOv2*-cls."
            )
        ckpt_task = loaded.get("task")
        if isinstance(ckpt_task, str) and normalize_task(ckpt_task) != "classify":
            raise RuntimeError(
                f"Checkpoint was trained for task={normalize_task(ckpt_task)!r}, "
                "but is being loaded into a LibreDINOv2 classification model."
            )

        state = self._extract_state(loaded)
        lw = state.get("linear.weight")
        if lw is None:
            raise RuntimeError(
                "Checkpoint does not look like a LibreDINOv2 classification model "
                "(no 'linear.weight' classifier head found)."
            )
        ckpt_nc = int(lw.shape[0])
        if ckpt_nc != self.nb_classes:
            self._rebuild_for_new_classes(ckpt_nc)

        result = self.model.load_state_dict(loaded, strict=False)
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        if any(
            ("class_embed" in k or "transformer" in k or k.startswith("predict."))
            for k in unexpected
        ):
            raise RuntimeError(
                "Checkpoint does not look like a LibreDINOv2 classification model "
                "(weights match a detector or dense head, not backbone + linear)."
            )

        ckpt_names = loaded.get("names")
        if ckpt_names is not None:
            self.names = self._sanitize_names(ckpt_names, self.nb_classes)
        self.model.to(self.device)

    # =========================================================================
    # Training
    # =========================================================================

    @ddp_aware(batch_key="batch_size")
    def train(
        self,
        data: str,
        epochs: int = 100,
        batch_size: int | None = None,
        lr: float | None = None,
        output_dir: str | None = None,
        resume: str | Path | bool | None = None,
        callbacks: TrainCallbacks = None,
        **kwargs,
    ) -> Dict:
        """Fine-tune LibreDINOv2 for semantic segmentation or classification.

        Task is taken from ``self.task``; ``DINOv2Trainer`` (via ``RFDETRTrainer``)
        routes the classify vs semantic data/loss branches accordingly.

        ``output_dir``, when given, splits into ``project`` (parent) and
        ``name`` (leaf); ``project=`` / ``name=`` kwargs take precedence.
        Defaults to ``<DINOv2Config.project>/<DINOv2Config.name>`` when omitted.

        ``resume``: checkpoint path, or True to resume from this run's own
        ``weights/last.pt``.
        """
        if self.task == "embed":
            raise NotImplementedError(
                "LibreDINOv2 task='embed' is a backbone feature extractor; "
                "training is not implemented."
            )
        from pathlib import Path as _Path

        from .trainer import DINOv2Trainer

        train_kwargs = dict(kwargs)
        project = train_kwargs.pop("project", None)
        name = train_kwargs.pop("name", None)
        exist_ok = train_kwargs.pop("exist_ok", _TRAIN_DEFAULTS.exist_ok)
        batch = train_kwargs.pop("batch", None)
        lr0 = train_kwargs.pop("lr0", None)
        if output_dir is not None:
            output_path = _Path(output_dir)
            if project is None:
                project = output_path.parent
            if name is None:
                name = output_path.name
        else:
            if project is None:
                project = _TRAIN_DEFAULTS.project
            if name is None:
                name = _TRAIN_DEFAULTS.name
        run_dir = _Path(project) / str(name)
        if resume is True:
            # resume=True reads weights/last.pt from this exact run_dir below;
            # never let _get_save_dir() increment away from it mid-resume.
            exist_ok = True

        if batch is not None and batch_size is not None and batch != batch_size:
            raise ValueError(
                f"Conflicting DINOv2 batch values: batch={batch} and batch_size={batch_size}"
            )
        if lr0 is not None and lr is not None and lr0 != lr:
            raise ValueError(f"Conflicting DINOv2 LR values: lr0={lr0} and lr={lr}")

        resolved_batch = batch if batch is not None else batch_size
        resolved_lr0 = lr0 if lr0 is not None else lr
        if resolved_batch is None:
            resolved_batch = 4
        if resolved_lr0 is None:
            resolved_lr0 = 1e-4

        # A caller-supplied device must win over the wrapper's; passing both
        # through to the trainer raised "multiple values for 'device'".
        resolved_device = train_kwargs.pop("device", None)
        if resolved_device is None:
            resolved_device = str(self.device)

        resume_path = None
        if resume:
            resume_path = run_dir / "weights" / "last.pt" if resume is True else resume

        trainer = DINOv2Trainer(
            model=self.model,
            wrapper_model=self,
            data=data,
            epochs=epochs,
            batch_size=resolved_batch,
            lr0=resolved_lr0,
            imgsz=train_kwargs.pop("imgsz", self.input_size),
            size=self.size,
            num_classes=self.nb_classes,
            project=str(project),
            name=str(name),
            exist_ok=exist_ok,
            resume=resume,
            device=resolved_device,
            callbacks=callbacks,
            **train_kwargs,
        )
        if resume:
            trainer.setup()
            trainer.resume(str(resume_path))

        result = trainer.train()
        self._restore_after_training(result)
        return result

    def _restore_after_training(self, result: dict) -> None:
        """Reload the best checkpoint after training completes."""
        checkpoint = None
        for key in ("best_checkpoint", "last_checkpoint"):
            path = result.get(key)
            if path and Path(path).exists():
                checkpoint = str(path)
                break
        if checkpoint is not None:
            self.model_path = checkpoint
            self._load_weights(checkpoint)
        model = getattr(self, "model", None)
        device = getattr(self, "device", None)
        if model is not None and device is not None and hasattr(model, "to"):
            model.to(device)
        if model is not None and hasattr(model, "eval"):
            model.eval()

    # =========================================================================
    # Export
    # =========================================================================

    def val(self, *args, **kwargs) -> Dict:
        if self.task == "embed":
            raise NotImplementedError(
                "LibreDINOv2 retrieval validation is not implemented."
            )
        return super().val(*args, **kwargs)

    def export(self, format: str = "onnx", *, opset: int = 17, **kwargs) -> str:
        if self.task == "embed":
            if format.lower() in {
                "onnx",
                "torchscript",
                "executorch",
                "tensorrt",
                "openvino",
                "tflite",
            }:
                return super().export(format=format, opset=opset, **kwargs)
            raise NotImplementedError(
                "LibreDINOv2 task='embed' export currently supports ONNX, "
                "TorchScript, ExecuTorch, TensorRT, OpenVINO, and TFLite only."
            )
        if self.task == "classify" and format.lower() in {
            "onnx",
            "torchscript",
            "executorch",
            "tensorrt",
            "openvino",
            "coreai",
        }:
            return super().export(format=format, opset=opset, **kwargs)
        if self.task == "semantic":
            return super().export(format=format, opset=opset, **kwargs)
        raise NotImplementedError(
            "LibreDINOv2 classify export currently supports ONNX, TorchScript, "
            "ExecuTorch, TensorRT, OpenVINO, and Core AI only."
        )
