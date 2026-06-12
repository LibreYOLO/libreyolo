"""Base class for the ``LibreVLM`` tier: generic vision-language models used as
open-vocabulary object detectors.

A VLM here is a multi-file Hugging Face repo: an autoregressive model that takes
an image plus a text prompt and generates text back. When that text is a list of
boxes, this base parses it into the standard ``Results``. There is no detection
head and no fixed class set; the vocabulary is the list of words you provide.

It exposes two layers: ``chat()`` (raw image-plus-text generation) and the
detection convenience (``set_classes()`` + ``predict()``/``track()``). It does
NOT define ``can_load``, which keeps VLM families out of the state-dict
``_registry`` and away from the ``LibreYOLO`` factory.

Subclasses declare a small adapter (HF_REPOS, INPUT_SIZES, the coordinate
convention, an optional license notice). See ``docs/librevlm_design.md`` for the
design decisions and the "add a new model" checklist, and
``docs/adr/0002-librevlm-contract.md`` for the contract.
"""

from __future__ import annotations

import json
import logging
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...utils.image_loader import ImageInput, ImageLoader
from ..base.model import BaseModel
from .parsing import build_detection_dict, extract_detections

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "LibreVLM models require the 'vlm' extra. Install with:\n"
    "    pip install 'libreyolo[vlm]'"
)
_SNAPSHOT_COMPLETE_MARKER = ".libreyolo_snapshot_complete"


class LibreVLMModel(BaseModel):
    """Generative VLM repurposed as a closed-set object detector."""

    # Subclasses override these.
    FAMILY: ClassVar[str] = ""
    FILENAME_PREFIX: ClassVar[str] = ""
    HF_REPOS: ClassVar[Dict[str, str]] = {}
    INPUT_SIZES: ClassVar[Dict[str, int]] = {}
    SUPPORTED_TASKS: ClassVar[tuple] = ("detect",)
    DEFAULT_TASK: ClassVar[str] = "detect"

    # Generative output has no calibrated per-box confidence. v1 assigns a
    # constant placeholder so predict/draw/track behave; ``conf=`` filtering and
    # mAP are therefore soft. Override ``_score_detections`` for a real signal.
    DEFAULT_SCORE: ClassVar[float] = 1.0
    MAX_NEW_TOKENS: ClassVar[int] = 1024
    # Output coordinate convention. LFM2-VL emits ``bbox`` normalized to [0, 1];
    # Qwen-style models emit ``bbox_2d`` on a 0-1000 scale. Families override.
    BBOX_KEY: ClassVar[str] = "bbox"
    COORD_DIVISOR: ClassVar[float] = 1.0
    # Box layout the model emits: "xyxy" (corners, default), "xywh" (top-left +
    # size), or "cxcywh" (center + size). Set by families whose output differs.
    BOX_FORMAT: ClassVar[str] = "xyxy"
    # Greedy decoding on a small VLM can fall into a repetition loop, emitting
    # the same box until the token budget is exhausted. A mild penalty breaks
    # the loop (and makes generation much faster) with negligible effect on the
    # numeric coordinates. Families may override the class attribute.
    REPETITION_PENALTY: ClassVar[float] = 1.1

    # Multi-scale TTA / tiling are meaningless for a fixed-resolution generator.
    TTA_ENABLED: ClassVar[bool] = False

    # ``_preprocess`` returns an HF BatchEncoding (not a stackable image
    # tensor) and ``_forward`` runs autoregressive generation, so the stacked
    # single-forward predict path does not apply.
    SUPPORTS_BATCHED_PREDICT: ClassVar[bool] = False

    # Family-specific weight license, printed once before the first download.
    _LICENSE_NOTICE: ClassVar[str] = ""
    _LICENSE_NOTICE_SHOWN: ClassVar[bool] = False

    # Off by default: all shipped families load through native transformers
    # classes, so we never execute third-party repo code. A family that genuinely
    # needs it must opt in explicitly (and pin a revision).
    TRUST_REMOTE_CODE: ClassVar[bool] = False
    # Current shipped VLM repos all provide safetensors. Avoid pulling duplicate
    # PyTorch binaries and export artifacts when downloading the local snapshot.
    SNAPSHOT_IGNORE_PATTERNS: ClassVar[tuple[str, ...]] = (
        "*.bin",
        "*.bin.index.json",
        "*.h5",
        "*.msgpack",
        "*.onnx",
        "*.ot",
        "*.tflite",
        "onnx/*",
    )
    UNSUPPORTED_GENERATE_INPUTS: ClassVar[tuple[str, ...]] = ("token_type_ids",)

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        self,
        size: str,
        *,
        nb_classes: int = 80,
        names: Optional[list] = None,
        device: str = "auto",
        task: str | None = None,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ):
        if size not in self.HF_REPOS:
            raise ValueError(
                f"Invalid size {size!r} for {type(self).__name__}. "
                f"Must be one of: {', '.join(self.HF_REPOS)}"
            )
        self._custom_prompt = prompt
        if max_new_tokens is not None:
            self.MAX_NEW_TOKENS = max_new_tokens

        # BaseModel.__init__ sets size/device/input_size/names, then calls
        # _init_model() (which downloads + loads the HF model and processor).
        # Passing the repo id as model_path keeps BaseModel on its non-dict,
        # non-None branch: it sets eval() and skips load_state_dict.
        super().__init__(
            model_path=self.HF_REPOS[size],
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=task,
            **kwargs,
        )

        if names is not None:
            self.set_classes(names)
        else:
            self._name_to_id = {v.strip().lower(): k for k, v in self.names.items()}
        self.model.eval()

    # =========================================================================
    # Open-vocabulary API
    # =========================================================================

    def set_classes(self, classes: list) -> "LibreVLMModel":
        """Set the open-vocabulary class list to detect.

        Sticky: call once after loading and the vocabulary persists across every
        later ``predict()`` / ``track()`` call until set again. ``classes`` is a
        plain list of label strings, e.g. ``["pink car", "wheel"]``; any words
        work, since the model is prompted with them rather than constrained to a
        fixed head. Returns ``self`` so calls can chain.
        """
        # Reject a bare string (it would enumerate into one-character classes)
        # and other non-sequence inputs.
        if isinstance(classes, str) or not isinstance(classes, (list, tuple)):
            raise TypeError(
                "set_classes() expects a list/tuple of label strings, "
                f"e.g. [\"boat\"], not {type(classes).__name__}."
            )
        if not classes:
            raise ValueError("set_classes() requires a non-empty list of labels.")
        names = [str(c) for c in classes]
        keys = [name.strip().lower() for name in names]
        if len(keys) != len(set(keys)):
            raise ValueError("set_classes() labels must be unique case-insensitively.")
        self.names = {i: name for i, name in enumerate(names)}
        self.nb_classes = len(self.names)
        self._name_to_id = {v.strip().lower(): k for k, v in self.names.items()}
        return self

    def chat(
        self,
        image: ImageInput,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        color_format: str = "auto",
    ) -> str:
        """Raw multimodal generation: image + prompt in, generated text out.

        The escape hatch beneath the detection convenience. Use it for free-form
        questions, custom output formats, counting, or any prompt the detection
        wrapper does not cover. Returns the model's decoded text verbatim.
        """
        img = ImageLoader.load(image, color_format=color_format)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": str(prompt)},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.device)
        inputs = self._prepare_generation_inputs(inputs)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=self.REPETITION_PENALTY,
            )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    # =========================================================================
    # Weight acquisition (autodownload via Hugging Face, license-gated)
    # =========================================================================

    @staticmethod
    def _snapshot_complete(local_dir: Path) -> bool:
        """True only if config.json AND every weight file are present.

        config.json alone is not proof: an interrupted download can leave it
        without the weight shards. For a sharded checkpoint, every shard named in
        the index must exist, not just one, so a multi-shard download interrupted
        between shards is correctly seen as incomplete and resumed.
        """
        if not (local_dir / _SNAPSHOT_COMPLETE_MARKER).exists():
            return False
        if not (local_dir / "config.json").exists():
            return False
        for index_name in (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ):
            index = local_dir / index_name
            if index.exists():
                try:
                    weight_map = json.loads(
                        index.read_text(encoding="utf-8")
                    ).get("weight_map", {})
                except (ValueError, OSError):
                    return False
                shards = set(weight_map.values())
                return bool(shards) and all((local_dir / s).exists() for s in shards)
        # Single-file checkpoint: any weight file is enough.
        return any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))

    @classmethod
    def _notify_license_once(cls) -> None:
        # Once per process per family. Routed through the logger (not print) so it
        # survives stdout capture and lands in application logs.
        if cls._LICENSE_NOTICE_SHOWN or not cls._LICENSE_NOTICE:
            return
        cls._LICENSE_NOTICE_SHOWN = True
        logger.warning(cls._LICENSE_NOTICE)

    def _ensure_weights(self) -> str:
        """Return a local weights dir for this size, downloading if needed.

        Downloads into ``weights/<FILENAME_PREFIX><size>/`` via ``local_dir`` so
        files are placed directly (copies, no symlinks). This matches LibreYOLO's
        ``weights/`` convention and avoids the symlinked HF cache, which needs
        admin/Developer Mode on Windows.
        """
        repo = self.HF_REPOS[self.size]
        local_dir = Path("weights") / f"{self.FILENAME_PREFIX}{self.size}"
        # Only short-circuit on a *complete* snapshot; otherwise (re)download.
        # snapshot_download skips complete files and resumes partial ones, so
        # re-calling it is safe and cheap.
        if self._snapshot_complete(local_dir):
            return str(local_dir)
        try:
            from huggingface_hub import snapshot_download
            import transformers
        except ImportError as exc:  # ships with transformers
            raise ImportError(_INSTALL_HINT) from exc
        _ = transformers
        self._notify_license_once()
        logger.info("Downloading %s weights from %s -> %s ...", self.FAMILY, repo, local_dir)
        snapshot_download(
            repo,
            local_dir=str(local_dir),
            ignore_patterns=self.SNAPSHOT_IGNORE_PATTERNS,
        )
        (local_dir / _SNAPSHOT_COMPLETE_MARKER).write_text(
            json.dumps({"repo": repo}) + "\n", encoding="utf-8"
        )
        if not self._snapshot_complete(local_dir):
            (local_dir / _SNAPSHOT_COMPLETE_MARKER).unlink(missing_ok=True)
            raise FileNotFoundError(
                f"Downloaded snapshot for {repo} is missing config or safetensors files "
                f"in {local_dir}."
            )
        return str(local_dir)

    def _resolve_dtype(self) -> "torch.dtype":
        """bf16 on CUDA only when the device supports it, else fp16; fp32 on CPU."""
        if self.device.type != "cuda":
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _load_pretrained(self, snapshot_dir: str):
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        model = AutoModelForImageTextToText.from_pretrained(
            snapshot_dir, dtype=self._resolve_dtype(), trust_remote_code=self.TRUST_REMOTE_CODE
        )
        processor = AutoProcessor.from_pretrained(
            snapshot_dir, trust_remote_code=self.TRUST_REMOTE_CODE
        )
        return model, processor

    def _init_model(self) -> nn.Module:
        snapshot_dir = self._ensure_weights()
        model, processor = self._load_pretrained(snapshot_dir)
        self.processor = processor
        # The actual loaded weight dtype (bf16/fp16 on CUDA, fp32 on CPU). Used to
        # cast the processor's float tensors so a half-precision model never gets
        # fp32 pixel_values on a vision tower that does not self-cast.
        self._model_dtype = next(model.parameters()).dtype
        return model

    def _prepare_generation_inputs(self, inputs: Any) -> Any:
        """Cast inputs and drop processor keys unsupported by ``generate``."""
        inputs = self._cast_inputs(inputs)
        if isinstance(inputs, MutableMapping):
            for key in self.UNSUPPORTED_GENERATE_INPUTS:
                inputs.pop(key, None)
        return inputs

    def _cast_inputs(self, inputs: Any) -> Any:
        """Cast the processor's float tensors (e.g. ``pixel_values``) to the model
        dtype. Integer ``input_ids`` are left intact. A no-op on CPU (model is fp32)."""
        dtype = getattr(self, "_model_dtype", None)
        if dtype is None:
            return inputs
        def cast(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(dtype=dtype) if value.is_floating_point() else value
            if isinstance(value, MutableMapping):
                for key, nested in value.items():
                    value[key] = cast(nested)
                return value
            if isinstance(value, tuple):
                return tuple(cast(nested) for nested in value)
            if isinstance(value, list):
                return [cast(nested) for nested in value]
            return value

        return cast(inputs)

    # =========================================================================
    # Prompt
    # =========================================================================

    def _detection_prompt(self) -> str:
        """Build the detection prompt for the current vocabulary.

        Handles the custom-prompt override and the label join; families supply
        only the format-specific body via ``_format_detection_prompt``.
        """
        if self._custom_prompt:
            return self._custom_prompt
        labels = ", ".join(self.names[i] for i in range(len(self.names)))
        return self._format_detection_prompt(labels)

    def _format_detection_prompt(self, labels: str) -> str:
        """Format-specific detection ask. Default uses a ``bbox`` key on a [0,1]
        scale; families whose output differs override this."""
        return (
            f"Detect all instances of: {labels}. "
            'Response must be a JSON array: '
            '[{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]. '
            "Coordinates are normalized to [0,1]. "
            "Only include objects that are actually visible; if there are none, "
            "respond with an empty array []."
        )

    # =========================================================================
    # InferenceRunner hooks: the whole predict/track surface
    # =========================================================================

    def _get_input_size(self) -> int:
        return self.input_size

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {name: module for name, module in self.model.named_modules() if name}

    @staticmethod
    def _get_preprocess_numpy():
        raise NotImplementedError(
            "VLM families preprocess through the HF processor, not a numpy hook."
        )

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": self._detection_prompt()},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        # original_size is (W, H); ratio is unused because boxes come back
        # normalized to the image, so no letterbox/unpad bookkeeping is needed.
        return inputs, img, img.size, 1.0

    def _forward(self, inputs: Any) -> torch.Tensor:
        inputs = self._prepare_generation_inputs(inputs)
        input_len = inputs["input_ids"].shape[1]
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=self.REPETITION_PENALTY,
        )
        # Strip the prompt tokens; keep only what the model generated.
        return generated[:, input_len:]

    def _score_detections(self, items: list) -> float:
        """Per-call confidence for parsed detections (placeholder in v1)."""
        return self.DEFAULT_SCORE

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
        text = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        items = extract_detections(text)
        return build_detection_dict(
            items,
            self._name_to_id,
            original_size,
            conf_thres=conf_thres,
            max_det=max_det,
            classes=kwargs.get("classes"),
            default_score=self._score_detections(items),
            bbox_key=self.BBOX_KEY,
            coord_divisor=self.COORD_DIVISOR,
            box_format=self.BOX_FORMAT,
        )

    # =========================================================================
    # Out of scope for the inference-first VLM tier
    # =========================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            f"Training is out of scope for {type(self).__name__} in LibreYOLO. "
            "Fine-tune the VLM upstream and load the resulting weights."
        )

    def val(self, *args, **kwargs):
        raise NotImplementedError(
            f"Dataset validation is not supported for {type(self).__name__}: "
            "generated boxes carry only a placeholder confidence, so COCO mAP "
            "would be misleading. Evaluate qualitatively via predict()."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} is a generative VLM and does not export to "
            f"{format!r}. Run it through predict()/track() instead."
        )
