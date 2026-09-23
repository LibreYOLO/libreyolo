"""LibreSigLIP2 - zero-shot classification and image/text embedding.

LibreSigLIP2 is a drop-in upgrade tier next to LibreCLIP: a classifier that
needs **no training and no fixed label set**::

    from libreyolo import LibreYOLO
    model = LibreYOLO("LibreSigLIP2b16-cls.pt")
    model.set_classes(["a forklift", "an empty aisle", "a spill"])
    r = model.predict("warehouse.jpg")[0]
    print(model.names[r.probs.top1], float(r.probs.top1conf))

It reuses LibreYOLO's ``classify`` plumbing (``Results.probs`` / ``top1`` /
``top5``) and the one open-vocabulary primitive, :meth:`set_classes`. Two things
differ from LibreCLIP:

* **Scoring is sigmoid-based.** SigLIP scores each class independently as
  ``logit = scale * (img . txt) + bias``. For drop-in classify UX the default is
  a softmax over the user's class set (matches LibreCLIP top-1/top-5); passing
  ``multi_label=True`` returns independent sigmoid probabilities - a capability
  CLIP cannot offer. ``logit_bias`` is loaded and applied; it cancels under
  softmax but is essential for calibrated sigmoid probabilities.
* **Multilingual tokenizer.** A SentencePiece (Gemma vocab) tokenizer behind the
  optional ``libreyolo[siglip2]`` extra, so class names in many languages work.

The towers are a native ``torch`` re-implementation (see :mod:`.nn`); no
``transformers`` dependency at runtime. Weights are the Apache-2.0
``google/siglip2-*`` fixed-resolution checkpoints (``model_type: "siglip"``),
converted with ``weights/convert_siglip2_weights.py``.

Zero-shot only: ``train()`` raises. ONNX export bakes the *current* label set
into a fixed ``[B, K]`` classifier graph (see :meth:`export`).

With ``task="embed"``, image prediction returns one normalized vector and
``embed_text`` returns normalized text rows in the same space. The default task
remains ``classify``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...tasks import normalize_task
from ...utils.serialization import load_trusted_torch_file
from ..base.classify_preprocess import ClassifyPreprocessMixin
from ..base.model import BaseModel
from .labels import (
    DEFAULT_TEMPLATES,
    humanize_labels,
    imagenet1k_classnames,
    openai_imagenet_templates,
)
from .nn import SIGLIP2_CONFIGS, build_siglip2_model

logger = logging.getLogger(__name__)

# SigLIP preprocessing: symmetric [-1, 1] normalization + square bilinear resize
# (NOT ImageNet or CLIP stats).
SIGLIP_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
SIGLIP_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)


def siglip2_logits(
    image_features: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    """SigLIP scores: ``scale.exp() * (img_norm . txt_norm) + bias``.

    ``image_features`` are un-normalized ``[B, D]``; ``text_embeds`` are the
    L2-normalized class matrix ``[K, D]``. Returns ``[B, K]`` logits.
    """
    image_features = F.normalize(image_features, dim=-1)
    scale = logit_scale.exp().to(
        device=image_features.device, dtype=image_features.dtype
    )
    bias = logit_bias.to(device=image_features.device, dtype=image_features.dtype)
    text_embeds = text_embeds.to(
        device=image_features.device, dtype=image_features.dtype
    )
    return scale * (image_features @ text_embeds.t()) + bias


def siglip2_probs(logits: torch.Tensor, multi_label: bool) -> torch.Tensor:
    """Convert SigLIP logits to probabilities.

    ``multi_label=False`` (default): softmax over the class set - a normalized
    single-label distribution matching LibreCLIP. The ``logit_bias`` is a single
    scalar and cancels here, so softmax hides a missing bias.

    ``multi_label=True``: independent per-class sigmoid probabilities. These
    depend on ``logit_bias`` (SigLIP's calibrated operating point), so this mode
    is what exercises correct bias handling.
    """
    logits = logits.float()
    if multi_label:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=-1)


class LibreSigLIP2(ClassifyPreprocessMixin, BaseModel):
    """Dual-tower zero-shot classifier and image/text embedder."""

    FAMILY: ClassVar[str] = "siglip2"
    FILENAME_PREFIX: ClassVar[str] = "LibreSigLIP2"
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    WEIGHT_EXT: ClassVar[str] = ".pt"

    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        size: cfg.image_size for size, cfg in SIGLIP2_CONFIGS.items()
    }
    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("classify", "embed")
    WEIGHT_TASKS: ClassVar[Tuple[str, ...]] = ("classify",)
    DEFAULT_TASK: ClassVar[str] = "classify"
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = True
    TRAIN_CONFIG = None

    # Attention pooling + fixed square resize make multi-scale TTA meaningless.
    TTA_ENABLED: ClassVar[bool] = False

    EVAL_MEAN = SIGLIP_MEAN
    EVAL_STD = SIGLIP_STD
    EVAL_SQUARE_RESIZE = True
    crop_pct = 1.0
    interpolation = "bilinear"
    validator_class: ClassVar[Optional[type]] = (
        None  # set lazily (see _resolve_validator)
    )

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """SigLIP checkpoints carry a learned ``logit_bias`` (CLIP has only
        ``logit_scale``) alongside the ``vision_model.embeddings.patch_embedding``
        conv and the ``text_model.head`` projection - a signature no other family
        shares."""
        return (
            "logit_bias" in weights_dict
            and "logit_scale" in weights_dict
            and "vision_model.embeddings.patch_embedding.weight" in weights_dict
            and "text_model.head.weight" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        w = weights_dict.get("vision_model.embeddings.patch_embedding.weight")
        if w is None:
            return None
        width = int(w.shape[0])
        if width == 768:
            return "b16"
        if width == 1152:
            return "so400m"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # Open-vocabulary: no fixed head; class count is set by set_classes()
        # (defaults to ImageNet-1k on construction).
        return None

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        self,
        model_path: str | dict | None = None,
        size: str | None = None,
        nb_classes: int | None = None,
        device: str = "auto",
        task: str | None = None,
        templates: Optional[Sequence[str]] = None,
        classes: Optional[Sequence[str]] = None,
        multi_label: bool = False,
        **kwargs,
    ) -> None:
        resolved_task = normalize_task(task) if task is not None else "classify"
        if resolved_task not in self.SUPPORTED_TASKS:
            raise ValueError(
                "LibreSigLIP2 supports task in ('classify', 'embed'); "
                f"got {task!r}."
            )

        if isinstance(model_path, dict):
            weight_source: str | dict = model_path
            if size is None:
                size = self.detect_size(self._extract_state(model_path))
        elif isinstance(model_path, str):
            weight_source = self._resolve_weights_path(model_path)
            if size is None:
                size = self.detect_size_from_filename(model_path)
        else:
            size = size or "b16"
            weight_source = self._resolve_weights_path(
                f"{self.FILENAME_PREFIX}{size}-cls.pt"
            )
        size = size or "b16"

        self._default_templates = (
            list(templates) if templates else list(DEFAULT_TEMPLATES)
        )
        self._multi_label = bool(multi_label)
        self._text_embeds: Optional[torch.Tensor] = None
        self.tokenizer = None  # built after super().__init__

        super().__init__(
            model_path=None,
            size=size,
            nb_classes=1000,
            device=device,
            task=resolved_task,
            **kwargs,
        )

        self._load_weights(weight_source)
        if isinstance(weight_source, str) and Path(weight_source).is_file():
            self.model_path = str(weight_source)
        self.model.eval()

        from .tokenizer import SigLIP2Tokenizer

        self.tokenizer = SigLIP2Tokenizer(context_length=self.model.context_length)

        if self.task == "classify":
            if classes is not None:
                self.set_classes(list(classes), templates=self._default_templates)
            else:
                self.set_classes(
                    imagenet1k_classnames(), templates=self._default_templates
                )
        else:
            self.names = {}

    @staticmethod
    def _extract_state(ckpt: dict) -> dict:
        for key in ("model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt

    # =========================================================================
    # Open-vocabulary head - the headline primitive
    # =========================================================================

    @torch.no_grad()
    def _encode_texts(self, texts: List[str], chunk: int = 256) -> torch.Tensor:
        """Run the text tower over many prompts, returning L2-normalized [N, D]."""
        out: List[torch.Tensor] = []
        for start in range(0, len(texts), chunk):
            batch = texts[start : start + chunk]
            tokens = self.tokenizer(batch).to(self.device)
            feats = self.model.encode_text(tokens)
            out.append(F.normalize(feats, dim=-1))
        return torch.cat(out, dim=0)

    def embed_text(self, texts: str | Sequence[str]) -> torch.Tensor:
        """Embed text rows in the same vector space as image embeddings."""
        items = [texts] if isinstance(texts, str) else list(texts)
        if any(not isinstance(text, str) for text in items):
            raise TypeError("embed_text() expects a string or a sequence of strings.")
        if not items:
            return torch.empty(
                (0, self.model.config.projection_size), dtype=torch.float32
            )
        return self._encode_texts(items).float().cpu()

    def set_classes(
        self,
        labels: Sequence[str],
        templates: Optional[Sequence[str]] = None,
        multi_label: Optional[bool] = None,
    ) -> "LibreSigLIP2":
        """Define the (open) class set for zero-shot classification.

        For each label, every prompt template is rendered, encoded by the text
        tower, L2-normalized, averaged across templates, then re-normalized. The
        resulting ``[K, D]`` text-embedding matrix is cached - it is the
        classifier head and is **not** recomputed per image.

        Args:
            labels: Class label strings (e.g. ``["a forklift", "a spill"]``).
            templates: Optional ``{}``-format prompt templates; defaults to the
                templates set at construction (``["a photo of a {}."]``). Pass
                ``LibreSigLIP2.imagenet_ensemble()`` for the 80-prompt ensemble.
            multi_label: If given, switch scoring mode for subsequent predicts.
                ``False`` (default) softmaxes over the class set; ``True`` returns
                independent sigmoid probabilities (SigLIP's native calibration).
        """
        labels = [str(lbl) for lbl in labels]
        if not labels:
            raise ValueError("set_classes() requires at least one label.")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized; weights must load first.")
        templates = list(templates) if templates else list(self._default_templates)
        if multi_label is not None:
            self._multi_label = bool(multi_label)

        prompts = [tmpl.format(label) for label in labels for tmpl in templates]
        feats = self._encode_texts(prompts)  # [K*T, D], normalized
        feats = feats.view(len(labels), len(templates), -1).mean(dim=1)  # [K, D]
        feats = F.normalize(feats, dim=-1)

        self._text_embeds = feats.to(self.device)
        self.nb_classes = len(labels)
        self.names = {i: label for i, label in enumerate(labels)}
        return self

    @property
    def multi_label(self) -> bool:
        return self._multi_label

    @multi_label.setter
    def multi_label(self, value: bool) -> None:
        self._multi_label = bool(value)

    @staticmethod
    def imagenet_ensemble() -> List[str]:
        """The 80-prompt OpenAI ImageNet template ensemble (a few extra points)."""
        return openai_imagenet_templates()

    # =========================================================================
    # BaseModel abstract surface
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return build_siglip2_model(self.size)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "image_tower": self.model.vision_model,
            "text_tower": self.model.text_model,
        }

    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        image_features = self.model.encode_image(input_tensor.to(self.device))
        if self.task == "embed":
            return F.normalize(image_features.float(), dim=-1)
        if self._text_embeds is None:
            raise RuntimeError("No classes set; call set_classes() first.")
        return siglip2_logits(
            image_features,
            self._text_embeds,
            self.model.logit_scale,
            self.model.logit_bias,
        )

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
        logits = output[0] if isinstance(output, (list, tuple)) else output
        probs = siglip2_probs(logits, self._multi_label)[0]
        return {"probs": probs.cpu()}

    # =========================================================================
    # Weights I/O
    # =========================================================================

    def _strict_loading(self) -> bool:
        return True

    def _load_weights(self, model_path: str | dict) -> None:
        if isinstance(model_path, dict):
            loaded = model_path
        else:
            path = Path(model_path)
            if not path.exists():
                from ...utils.download import download_weights

                download_weights(str(path), self.size)
            loaded = load_trusted_torch_file(
                str(model_path), map_location="cpu", context="LibreSigLIP2 weights"
            )

        if not isinstance(loaded, dict):
            raise TypeError("LibreSigLIP2 checkpoints must be dictionaries.")

        ckpt_family = loaded.get("model_family", "")
        if ckpt_family and ckpt_family != self.FAMILY:
            raise RuntimeError(
                f"Checkpoint was trained with model_family='{ckpt_family}' but is "
                f"being loaded into '{self.FAMILY}'."
            )
        ckpt_task = loaded.get("task")
        if (
            isinstance(ckpt_task, str)
            and normalize_task(ckpt_task) not in ("classify", "embed")
        ):
            raise RuntimeError(
                f"Checkpoint task={normalize_task(ckpt_task)!r} is not compatible "
                "with LibreSigLIP2."
            )

        state = self._extract_state(loaded)
        if (
            "logit_bias" not in state
            or "vision_model.embeddings.patch_embedding.weight" not in state
        ):
            raise RuntimeError(
                "Checkpoint does not look like a LibreSigLIP2 model (missing "
                "'logit_bias'/'vision_model.embeddings.patch_embedding.weight')."
            )
        self.model.load_state_dict(state, strict=self._strict_loading())
        self.model.to(self.device).eval()

    # =========================================================================
    # Validation - reuse ClassifyValidator with SigLIP preprocessing + open vocab
    # =========================================================================

    def _resolve_validator(self):
        from ...validation.siglip2_validator import SigLIP2ClassifyValidator

        return SigLIP2ClassifyValidator

    def val(self, data: str | None = None, **kwargs) -> Dict:
        """Zero-shot top-1/top-5 on an ImageFolder split.

        Reads the train-split folder names, humanizes wnid folders to readable
        labels, calls :meth:`set_classes`, then runs the SigLIP-preprocessing
        validator. Zero-shot accuracy depends on the label *wording*.
        """
        if self.task != "classify":
            raise NotImplementedError(
                "LibreSigLIP2 retrieval validation is not implemented; load "
                "task='classify' for zero-shot classification validation."
            )
        from ...data.classify_dataset import get_class_names, resolve_classify_data

        if data is None:
            raise ValueError("LibreSigLIP2.val() requires data= (an ImageFolder root).")
        root = resolve_classify_data(data)
        folder_names = get_class_names(root, split="train")
        self.set_classes(humanize_labels(folder_names))
        self.validator_class = self._resolve_validator()
        return super().val(data=data, **kwargs)

    # =========================================================================
    # Training is out of scope (zero-shot)
    # =========================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreSigLIP2 is zero-shot / inference-only; there is nothing to train. "
            "Use set_classes([...]) to define classes, then predict()/val(). "
            "SigLIP fine-tuning (linear-probe / full) is a separate future feature."
        )

    # =========================================================================
    # Export - frozen-class ONNX
    # =========================================================================

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export image embeddings or a frozen-class ONNX classifier.

        ``task='embed'`` traces the normalized image tower through the shared
        exporters. For ``task='classify'``, the current ``set_classes`` text
        embeddings are baked into a final linear projection with the learned
        scale and bias, giving a standard ``[B, K]`` classifier graph without
        the text tower or tokenizer.
        """
        if self.task == "embed":
            if format.lower() in {
                "onnx",
                "torchscript",
                "executorch",
                "tensorrt",
                "openvino",
                "tflite",
            }:
                kwargs.setdefault("opset", 17)
                return super().export(format=format, **kwargs)
            raise NotImplementedError(
                "LibreSigLIP2 task='embed' export currently supports ONNX, "
                "TorchScript, ExecuTorch, TensorRT, OpenVINO, and TFLite only."
            )
        if format.lower() in {
            "torchscript",
            "executorch",
            "tensorrt",
            "openvino",
            "tflite",
        }:
            if self._text_embeds is None:
                raise RuntimeError(
                    "No classes set; call set_classes() before export()."
                )
            if self._multi_label:
                raise NotImplementedError(
                    "LibreSigLIP2 multi-label exported-backend prediction is "
                    "not implemented; set multi_label=False before export()."
                )
            kwargs.setdefault("opset", 17)
            return super().export(format=format, **kwargs)
        if format.lower() not in {"onnx", "coreai"}:
            raise NotImplementedError(
                f"LibreSigLIP2 export to {format!r} is not implemented. "
                "Open-vocabulary export (two towers + tokenizer) is out of "
                "scope for v1."
            )
        if self._text_embeds is None:
            raise RuntimeError("No classes set; call set_classes() before export().")

        if format.lower() == "coreai":
            # LibreSigLIP2 is a two-tower module with no single forward(x),
            # which is why the ONNX path builds its graph by hand. Reuse that
            # same frozen-class module rather than duplicating it, then hand
            # it to the shared Core AI converter. The logit_bias matters here:
            # without it the exported logits do not match native.
            import torch as _torch

            from ...export.coreai import (
                export_coreai,
                prepare_frozen_classifier_export,
            )
            from .export import _FrozenSigLIP2Classifier

            size, output_path, metadata = prepare_frozen_classifier_export(
                self, kwargs, default_output="siglip2_coreai"
            )
            scale = float(self.model.logit_scale.exp().detach().cpu())
            weight = (scale * self._text_embeds).detach().cpu()
            bias = self.model.logit_bias.detach().to("cpu", _torch.float32).reshape(())
            device = next(self.model.vision_model.parameters()).device
            was_training = self.model.vision_model.training
            vision = self.model.vision_model.to("cpu").eval()
            try:
                frozen = _FrozenSigLIP2Classifier(vision, weight, bias).eval()
                dummy = _torch.randn(1, 3, size, size)
                return export_coreai(
                    frozen,
                    dummy,
                    output_path=output_path,
                    metadata=metadata,
                    model_family="siglip2",
                )
            finally:
                self.model.vision_model.to(device).train(was_training)

        from .export import export_frozen_onnx

        return export_frozen_onnx(self, **kwargs)
