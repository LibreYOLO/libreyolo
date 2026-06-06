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

import logging
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
    # Greedy decoding on a small VLM can fall into a repetition loop, emitting
    # the same box until the token budget is exhausted. A mild penalty breaks
    # the loop (and makes generation much faster) with negligible effect on the
    # numeric coordinates. Families may override the class attribute.
    REPETITION_PENALTY: ClassVar[float] = 1.1

    # Multi-scale TTA / tiling are meaningless for a fixed-resolution generator.
    TTA_ENABLED: ClassVar[bool] = False

    # Family-specific weight license, printed once before the first download.
    _LICENSE_NOTICE: ClassVar[str] = ""
    _LICENSE_NOTICE_SHOWN: ClassVar[bool] = False

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
            self._name_to_id = {v.lower(): k for k, v in self.names.items()}
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
        if not classes:
            raise ValueError("set_classes() requires a non-empty list of labels.")
        self.names = {i: str(c) for i, c in enumerate(classes)}
        self.nb_classes = len(self.names)
        self._name_to_id = {v.lower(): k for k, v in self.names.items()}
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

    @classmethod
    def _notify_license_once(cls) -> None:
        if cls._LICENSE_NOTICE_SHOWN or not cls._LICENSE_NOTICE:
            return
        cls._LICENSE_NOTICE_SHOWN = True
        print(cls._LICENSE_NOTICE)

    def _ensure_weights(self) -> str:
        """Return a local weights dir for this size, downloading if needed.

        Downloads into ``weights/<FILENAME_PREFIX><size>/`` via ``local_dir`` so
        files are placed directly (copies, no symlinks). This matches LibreYOLO's
        ``weights/`` convention and avoids the symlinked HF cache, which needs
        admin/Developer Mode on Windows.
        """
        repo = self.HF_REPOS[self.size]
        local_dir = Path("weights") / f"{self.FILENAME_PREFIX}{self.size}"
        if (local_dir / "config.json").exists():
            return str(local_dir)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # ships with transformers
            raise ImportError(_INSTALL_HINT) from exc
        self._notify_license_once()
        logger.info("Downloading %s weights from %s -> %s ...", self.FAMILY, repo, local_dir)
        snapshot_download(repo, local_dir=str(local_dir))
        return str(local_dir)

    def _load_pretrained(self, snapshot_dir: str):
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(
            snapshot_dir, dtype=dtype, trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(snapshot_dir, trust_remote_code=True)
        return model, processor

    def _init_model(self) -> nn.Module:
        snapshot_dir = self._ensure_weights()
        model, processor = self._load_pretrained(snapshot_dir)
        self.processor = processor
        return model

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
            default_score=self._score_detections(items),
            bbox_key=self.BBOX_KEY,
            coord_divisor=self.COORD_DIVISOR,
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
