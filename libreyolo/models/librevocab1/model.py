"""LibreVocab1 — BaseModel-style wrapper around SAM3 + CRADIOv4.

Path A architecture: a thin ``BaseModel`` shell whose ``_init_model`` builds
the SAM3 image stack with CRADIOv4 swapped in as the vision trunk, via the
``build_librevocab1`` orchestrator.

This class does **not** load LibreYOLO-format checkpoints. There is no
LibreVocab1 ``.pt`` file; the model is composed at runtime from upstream
SAM3 + CRADIOv4 weights. Therefore:

    - ``can_load`` always returns ``False`` (the factory cannot route here).
    - ``LibreVocab1`` is meant to be constructed directly:
        ``model = LibreVocab1(size="h", device="cuda")``.
    - The family is **not** auto-registered in ``libreyolo/models/__init__.py``
      until inference parity vs ``mranzinger/sam3-radio`` is verified.

Sizes mirror CRADIOv4 variants:
    - ``s`` -> ``c-radio_v4-so400m`` (412M backbone, 1152-d features)
    - ``h`` -> ``c-radio_v4-h`` (631M backbone, 1280-d features)

Both run at SAM3's input resolution of 1152x1152, where CRADIOv4's 16-pixel
patch produces a 72x72 token grid (same effective grid as SAM3's 14-pixel
patch at 1008x1008).
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

import torch
import torch.nn as nn

from ..base import BaseModel


_RADIO_VERSION_BY_SIZE = {
    "s": "c-radio_v4-so400m",
    "h": "c-radio_v4-h",
}


class LibreVocab1(BaseModel):
    """Open-vocabulary detector: SAM3 stack with CRADIOv4 vision encoder.

    Args (constructor):
        size: ``"s"`` (CRADIOv4-SO400M) or ``"h"`` (CRADIOv4-H).
        device: torch device string.
        sam3_checkpoint_path: optional local path to a SAM3 image checkpoint.
            If not provided, downloads from the SAM3 HuggingFace repo.
        confidence_threshold: score threshold passed to ``Sam3Processor``.
        vitdet_window_size: optional ViTDet window size for CRADIOv4
            (e.g. ``8`` or ``16``). Reduces high-resolution latency.
        nb_classes: ignored. Open-vocab is class-list-free; classes come from
            text prompts at inference. Kept for ``BaseModel`` signature
            compatibility.
    """

    FAMILY = "librevocab1"
    FILENAME_PREFIX = "LibreVocab1"
    INPUT_SIZES = {"s": 1152, "h": 1152}
    TRAIN_CONFIG = None  # fine-tuning surface deferred to a later iteration
    SUPPORTS_SEG = True  # SAM3 emits masks; we expose them via _postprocess

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # No LibreVocab1 checkpoint format exists — runtime-composed model.
        return False

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # Open-vocab: nb_classes is set per-call by the prompt list length.
        return None

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        m = re.search(r"librevocab1([sh])(?:_|\.|$)", filename.lower())
        return m.group(1) if m else None

    # ------------------------------------------------------------------
    # Constructor & build
    # ------------------------------------------------------------------

    def __init__(
        self,
        size: str = "h",
        device: str = "cuda",
        sam3_checkpoint_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        vitdet_window_size: Optional[int] = None,
        nb_classes: Optional[int] = None,  # accepted for BaseModel parity, unused
        **kwargs,
    ) -> None:
        if size not in _RADIO_VERSION_BY_SIZE:
            raise ValueError(
                f"LibreVocab1 size must be one of {sorted(_RADIO_VERSION_BY_SIZE)}; got {size!r}"
            )
        # Stash builder args before super().__init__() because BaseModel.__init__
        # will call self._init_model() and we need them readable from there.
        self._radio_version = _RADIO_VERSION_BY_SIZE[size]
        self._sam3_checkpoint_path = sam3_checkpoint_path
        self._confidence_threshold = confidence_threshold
        self._vitdet_window_size = vitdet_window_size
        # Route through BaseModel: it does not load any LibreYOLO checkpoint
        # when model_path is None, but it does set self.size / self.device /
        # self.nb_classes / self.input_size and call self._init_model().
        super().__init__(
            model_path=None,
            size=size,
            nb_classes=nb_classes if nb_classes is not None else 80,
            device=device,
        )
        # BaseModel sets train() when model_path is None; we want eval for
        # inference parity with sam3-radio's demo.
        self.model.eval()

    def _init_model(self) -> nn.Module:
        from .builder import build_librevocab1
        sam3_model, processor = build_librevocab1(
            radio_model_version=self._radio_version,
            sam3_checkpoint_path=self._sam3_checkpoint_path,
            load_sam3_from_hf=self._sam3_checkpoint_path is None,
            confidence_threshold=self._confidence_threshold,
            vitdet_window_size=self._vitdet_window_size,
            device=str(self.device),
        )
        self.processor = processor
        return sam3_model

    # ------------------------------------------------------------------
    # BaseModel surface (placeholders until inference parity is verified)
    # ------------------------------------------------------------------

    def _get_available_layers(self) -> List[str]:
        return []

    def _preprocess(self, image_input: Any, **kwargs):
        # Sam3Processor handles the canvas + normalization. We let it own
        # preprocessing rather than reimplementing it under the LibreYOLO
        # contract — that contract was designed for fixed-class detectors,
        # which LibreVocab1 isn't.
        raise NotImplementedError(
            "LibreVocab1 uses Sam3Processor for preprocessing; call "
            "model.predict(image, prompts) instead of the BaseModel "
            "predict() entry point."
        )

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        raise NotImplementedError(
            "LibreVocab1 forward goes through Sam3Processor + Sam3Image; "
            "use predict(image, prompts)."
        )

    def _postprocess(self, predictions: Any, metadata: Any, **kwargs) -> Any:
        raise NotImplementedError(
            "LibreVocab1 postprocess is handled inside Sam3Processor."
        )

    def _get_preprocess_numpy(self):
        raise NotImplementedError("Sam3Processor handles preprocessing.")

    # ------------------------------------------------------------------
    # User-facing API (open-vocab specific)
    # ------------------------------------------------------------------

    def predict(self, image, prompts: List[str]):
        """Run text-conditioned detection + segmentation.

        Args:
            image: PIL.Image, numpy array, or path to image file.
            prompts: list of free-form text prompts. Each prompt is applied
                independently and produces its own (boxes, masks, scores).

        Returns:
            List of per-prompt result dicts:
                [{"prompt": str,
                  "boxes": Tensor (N, 4) in xyxy pixel coords,
                  "masks": BoolTensor (N, H, W),
                  "scores": Tensor (N,)},
                 ...]
        """
        from PIL import Image as _Image
        if isinstance(image, str):
            image = _Image.open(image).convert("RGB")
        img_w, img_h = image.size
        state = self.processor.set_image(image)
        results = []
        for prompt in prompts:
            # set_text_prompt erases any previous text prompt and runs
            # inference. The returned state has boxes/masks/scores set.
            state = self.processor.set_text_prompt(prompt, state)
            boxes = state.get("boxes")
            if boxes is not None and boxes.numel() > 0:
                # Sam3Processor unscales sigmoid outputs by image size without a
                # final clip; slight overshoots (≈1 px) are common at edges.
                # Clamp to the image bounds so downstream consumers can trust
                # the box rectangle is inside the image.
                boxes = boxes.clone()
                boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=img_w)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=img_h)
            results.append({
                "prompt": prompt,
                "boxes": boxes,
                "masks": state.get("masks"),
                "scores": state.get("scores"),
            })
        return results
