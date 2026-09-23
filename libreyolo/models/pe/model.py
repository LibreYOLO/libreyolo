"""LibrePE - Perception Encoder Core zero-shot classification and embedding.

PE Core is Meta's dual-tower vision-language encoder
(https://arxiv.org/abs/2504.13181). It maps images, finite videos, and text
into one normalized embedding space::

    from libreyolo import LibreYOLO

    model = LibreYOLO("LibrePEb16-cls.pt")
    model.set_classes(["a forklift", "an empty aisle", "a spill"])
    r = model.predict("warehouse.jpg")[0]
    print(model.names[r.probs.top1], float(r.probs.top1conf))

With ``task="embed"`` image prediction returns one normalized ``(1, D)`` row and
:meth:`embed_text` returns normalized text rows in the same space::

    embedder = LibreYOLO("LibrePEb16-cls.pt", task="embed")
    image_rows = embedder.predict("photo.jpg")[0].embeddings
    text_rows = embedder.embed_text(["a dog", "a cat"])

Unlike every other embedding family, PE also defines a **whole-clip** video
embedding: frames are sampled uniformly over a finite video, encoded
independently by the image tower, averaged, then L2-normalized exactly once::

    clip_row = embedder.predict("clip.mp4", clip_frames=8)[0].embeddings

That behavior is opt-in at the family level (``VIDEO_EMBED_MODE = "clip"``);
every other family keeps its existing frame-by-frame video path. Image *lists*
remain image batches and are never guessed to be a temporal clip. Live and
otherwise unbounded sources are rejected rather than buffered.

The towers are a native ``torch`` re-implementation (see :mod:`.nn`); neither
``timm`` nor ``open_clip`` is imported at runtime. Weights are the Apache-2.0
OpenCLIP-compatible ``timm/PE-Core-*`` conversions, converted with
``weights/convert_pe_weights.py``.

Zero-shot only: :meth:`train` raises. See the class docstring of
:class:`LibrePE` for why PE Core has no supervised training path.
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
from ..clip.labels import (
    DEFAULT_TEMPLATES,
    imagenet1k_classnames,
    openai_imagenet_templates,
)
from .nn import PE_CONFIGS, PE_MEAN, PE_STD, build_pe_model

logger = logging.getLogger(__name__)

# LibreYOLO operational default for uniform frame sampling over a finite video.
# This is a LibreYOLO choice, not a claim about a unique upstream clip length.
DEFAULT_CLIP_FRAMES = 8


class LibrePE(ClassifyPreprocessMixin, BaseModel):
    """Perception Encoder Core: zero-shot classifier and image/video/text embedder.

    Inference-only by design, not by omission. PE Core exposes zero-shot
    classification and foundation embeddings; it has no closed-set supervised
    head, ``embed`` has no label or task loss in LibreYOLO, and the pinned
    upstream provides no practical single-GPU PE Core fine-tuning recipe.
    :meth:`train` therefore rejects immediately rather than silently routing to
    the generic classification trainer.
    """

    FAMILY: ClassVar[str] = "pe"
    FILENAME_PREFIX: ClassVar[str] = "LibrePE"
    WEIGHT_EXT: ClassVar[str] = ".pt"

    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        size: cfg.image_size for size, cfg in PE_CONFIGS.items()
    }
    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("classify", "embed")
    WEIGHT_TASKS: ClassVar[Tuple[str, ...]] = ("classify",)
    DEFAULT_TASK: ClassVar[str] = "classify"
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = True
    TRAIN_CONFIG = None

    # PE pools with a single attention latent over a fixed square resize, so
    # multi-scale TTA is meaningless.
    TTA_ENABLED: ClassVar[bool] = False

    # Opt in to the shared finite-video clip-embedding path. Every other family
    # keeps the default "frames" behavior.
    VIDEO_EMBED_MODE: ClassVar[str] = "clip"

    EVAL_MEAN = PE_MEAN
    EVAL_STD = PE_STD
    EVAL_SQUARE_RESIZE = True
    crop_pct = 1.0
    interpolation = "bilinear"
    validator_class: ClassVar[Optional[type]] = None

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """PE checkpoints pair an EVA-style RoPE trunk with attention pooling
        and an OpenCLIP text tower.

        ``visual.trunk.attn_pool.latent`` plus the bias-free
        ``visual.trunk.patch_embed.proj.weight`` is a signature no other
        LibreYOLO family carries: CLIP uses ``visual.transformer.*``, SigLIP2
        uses ``vision_model.embeddings.*``, and DINOv2 has no text tower.
        """
        return (
            "visual.trunk.attn_pool.latent" in weights_dict
            and "visual.trunk.patch_embed.proj.weight" in weights_dict
            and "text.text_projection" in weights_dict
            and "logit_scale" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Infer size from (trunk width, patch size, context length).

        Width alone is ambiguous-free across the released series, but patch
        size and context length are checked too so a mismatched or hand-edited
        checkpoint fails rather than loading as the wrong variant.
        """
        patch = weights_dict.get("visual.trunk.patch_embed.proj.weight")
        pos = weights_dict.get("text.positional_embedding")
        if patch is None or pos is None:
            return None
        width, patch_size, context = (
            int(patch.shape[0]),
            int(patch.shape[-1]),
            int(pos.shape[0]),
        )
        for size, cfg in PE_CONFIGS.items():
            if (
                cfg.embed_dim == width
                and cfg.patch_size == patch_size
                and cfg.context_length == context
            ):
                return size
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # Open-vocabulary: no fixed head. The class count is whatever
        # set_classes() defines (ImageNet-1k on construction).
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
        clip_frames: int = DEFAULT_CLIP_FRAMES,
        **kwargs,
    ) -> None:
        resolved_task = normalize_task(task) if task is not None else "classify"
        if resolved_task not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"LibrePE supports task in {self.SUPPORTED_TASKS}; got {task!r}."
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
        self._text_embeds: Optional[torch.Tensor] = None
        self.clip_frames = self._validate_clip_frames(clip_frames)
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

        from ..clip.tokenizer import SimpleTokenizer

        self.tokenizer = SimpleTokenizer(context_length=self.model.context_length)

        if self.task == "classify":
            self.set_classes(
                list(classes) if classes is not None else imagenet1k_classnames(),
                templates=self._default_templates,
            )
        else:
            self.names = {}

    @staticmethod
    def _validate_clip_frames(clip_frames: int) -> int:
        frames = int(clip_frames)
        if frames < 1:
            raise ValueError(f"clip_frames must be positive; got {clip_frames!r}.")
        return frames

    @staticmethod
    def _extract_state(ckpt: dict) -> dict:
        for key in ("model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt

    # =========================================================================
    # Open-vocabulary head
    # =========================================================================

    @torch.no_grad()
    def _encode_texts(self, texts: List[str], chunk: int = 256) -> torch.Tensor:
        out: List[torch.Tensor] = []
        for start in range(0, len(texts), chunk):
            tokens = self.tokenizer(texts[start : start + chunk]).to(self.device)
            out.append(F.normalize(self.model.encode_text(tokens), dim=-1))
        return torch.cat(out, dim=0)

    def embed_text(self, texts: str | Sequence[str]) -> torch.Tensor:
        """Embed text rows into the same space as image and video embeddings."""
        items = [texts] if isinstance(texts, str) else list(texts)
        if any(not isinstance(text, str) for text in items):
            raise TypeError("embed_text() expects a string or a sequence of strings.")
        if not items:
            return torch.empty((0, self.model.embedding_dim), dtype=torch.float32)
        return self._encode_texts(items).float().cpu()

    def set_classes(
        self,
        labels: Sequence[str],
        templates: Optional[Sequence[str]] = None,
    ) -> "LibrePE":
        """Define the (open) class set for zero-shot classification.

        Each label is rendered through every template, encoded, L2-normalized,
        averaged across templates, then re-normalized. The resulting ``[K, D]``
        matrix *is* the classifier head and is cached, not recomputed per image.
        """
        labels = [str(label) for label in labels]
        if not labels:
            raise ValueError("set_classes() requires at least one label.")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized; weights must load first.")
        templates = list(templates) if templates else list(self._default_templates)

        prompts = [tmpl.format(label) for label in labels for tmpl in templates]
        feats = self._encode_texts(prompts)
        feats = feats.view(len(labels), len(templates), -1).mean(dim=1)
        self._text_embeds = F.normalize(feats, dim=-1).to(self.device)
        self.nb_classes = len(labels)
        self.names = {i: label for i, label in enumerate(labels)}
        return self

    @staticmethod
    def imagenet_ensemble() -> List[str]:
        """The 80-prompt OpenAI ImageNet template ensemble."""
        return openai_imagenet_templates()

    # =========================================================================
    # BaseModel abstract surface
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return build_pe_model(self.size)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "image_tower": self.model.visual,
            "text_tower": self.model.text,
        }

    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        tensor = input_tensor.to(self.device)
        if tensor.ndim == 5:
            # Whole-clip path: (B, F, C, H, W) -> one normalized row per clip.
            return self.model.encode_video(tensor).float()
        image_features = self.model.encode_image(tensor)
        if self.task == "embed":
            return F.normalize(image_features.float(), dim=-1)
        if self._text_embeds is None:
            raise RuntimeError("No classes set; call set_classes() first.")
        scale = self.model.logit_scale.exp().to(
            device=image_features.device, dtype=image_features.dtype
        )
        text_embeds = self._text_embeds.to(
            device=image_features.device, dtype=image_features.dtype
        )
        return scale * (F.normalize(image_features, dim=-1) @ text_embeds.t())

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
        return {"probs": torch.softmax(logits.float(), dim=-1)[0].cpu()}

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
                str(model_path), map_location="cpu", context="LibrePE weights"
            )

        if not isinstance(loaded, dict):
            raise TypeError("LibrePE checkpoints must be dictionaries.")

        ckpt_family = loaded.get("model_family", "")
        if ckpt_family and ckpt_family != self.FAMILY:
            raise RuntimeError(
                f"Checkpoint was trained with model_family='{ckpt_family}' but is "
                f"being loaded into '{self.FAMILY}'."
            )
        ckpt_task = loaded.get("task")
        if isinstance(ckpt_task, str) and normalize_task(ckpt_task) not in (
            "classify",
            "embed",
        ):
            raise RuntimeError(
                f"Checkpoint task={normalize_task(ckpt_task)!r} is not compatible "
                "with LibrePE."
            )

        state = self._extract_state(loaded)
        if not self.can_load(state):
            raise RuntimeError(
                "Checkpoint does not look like a LibrePE model (missing the "
                "'visual.trunk.attn_pool.latent' / 'text.text_projection' "
                "signature)."
            )
        self.model.load_state_dict(state, strict=self._strict_loading())
        self.model.to(self.device).eval()

    # =========================================================================
    # Training is out of scope (zero-shot foundation encoder)
    # =========================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "LibrePE is inference-only: PE Core exposes zero-shot classification "
            "and foundation embeddings, not a closed-set supervised head, so "
            "there is no LibreYOLO task loss to optimize (task='embed' has no "
            "labels; task='classify' has no learned head). The pinned upstream "
            "also ships no practical single-GPU PE Core fine-tuning recipe. "
            "Use set_classes([...]) then predict()/val() for zero-shot "
            "classification, or task='embed' for retrieval."
        )

    # =========================================================================
    # Export
    # =========================================================================

    def export(
        self,
        format: str = "onnx",
        video: bool = False,
        clip_frames: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Export an image-embedding, frozen-class, or fixed-frame video graph.

        The graph is chosen from the loaded task plus ``video=``:

        * ``task='embed'``           -> class-independent ``[B, D]`` unit rows.
        * ``task='classify'``        -> ``[B, K]`` logits with the current
          ``set_classes`` matrix baked in (no text tower, no tokenizer).
        * ``video=True``             -> fixed-``F`` 5D clip graph, ``[B, D]``.

        Only ONNX and TorchScript are implemented; both were parity tested.
        Other formats are rejected rather than silently advertised.
        """
        from .export import export_onnx, export_torchscript

        if video and self.task != "embed":
            raise ValueError(
                "Video export produces embeddings; load the model with "
                "task='embed' to export a clip graph."
            )
        kind = "video" if video else ("embed" if self.task == "embed" else "classify")
        frames = self._validate_clip_frames(
            clip_frames if clip_frames is not None else self.clip_frames
        )

        fmt = format.lower()
        if fmt == "onnx":
            kwargs.setdefault("opset", 17)
            return export_onnx(self, kind, frames=frames, **kwargs)
        if fmt == "torchscript":
            kwargs.pop("opset", None)
            kwargs.pop("dynamic_batch", None)
            return export_torchscript(self, kind, frames=frames, **kwargs)
        raise NotImplementedError(
            f"LibrePE export to {format!r} is not implemented. PE ships ONNX and "
            "TorchScript, the two formats its graphs were parity tested in."
        )
