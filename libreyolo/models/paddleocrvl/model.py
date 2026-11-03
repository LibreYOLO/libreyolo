"""LibreYOLO wrapper for PaddleOCR-VL, a two-stage document parser.

PaddleOCR-VL (PaddlePaddle, Apache-2.0 code and weights) pairs the
PP-DocLayoutV2 layout detector (RT-DETR-style boxes for 25 layout classes
plus a pointer network for reading order) with PaddleOCR-VL-0.9B, a
NaViT-style vision encoder and the ERNIE-4.5-0.3B decoder that reads one
region at a time from a task prompt: text, table (OTSL, returned as HTML),
formula (LaTeX) or chart.

Both networks run through the native transformers classes
(``PaddleOCRVLForConditionalGeneration`` and
``PPDocLayoutV2ForObjectDetection``, transformers >= 5.3); no remote code is
executed. The pipeline around them (layout thresholds, NMS and merges, the
overlap filter, prompt routing, pixel budgets, content clean-up and page
markdown) is a port of the upstream PaddleX pipeline, see ``NOTICE``.

    from libreyolo import LibreVLM
    model = LibreVLM("paddleocr-vl")
    result = model.predict("page.png")
    result.ocr.texts        # one transcript per region, in reading order
    result.ocr.labels       # layout class per region ("text", "table", ...)
    result.ocr.markdown     # the whole page as markdown
    LibreVLM("paddleocr-vl", use_layout_detection=False, prompt_label="table")

Confidence: ``Results.ocr.conf`` is the geometric mean of the generated
tokens' probabilities, ``Results.ocr.det_conf`` the layout score. Regions are
axis-aligned boxes written as 4-point quads.

Devices: CUDA runs the recognizer in bf16 (fp16 without bf16 support) and
the layout detector in fp32; CPU and MPS run both in fp32. On MPS some
transformers attention paths may fall back to CPU ops; set
``PYTORCH_ENABLE_MPS_FALLBACK=1`` if an operator is reported missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..base.model import BaseModel
from ..vlm.base import _INSTALL_HINT, LibreVLMModel
from . import blocks, layout
from .markdown import build_markdown

logger = logging.getLogger(__name__)

LAYOUT_SUBDIR = "layout"
_MIN_TRANSFORMERS = (5, 3)
_PROMPT_LABELS = tuple(blocks.PROMPTS)


def _check_transformers() -> None:
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    parts = []
    for piece in transformers.__version__.split(".")[:2]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits or 0))
    if tuple(parts) < _MIN_TRANSFORMERS:
        raise ImportError(
            "PaddleOCR-VL needs transformers >= 5.3 (native PaddleOCR-VL and "
            f"PP-DocLayoutV2 classes); found {transformers.__version__}. "
            "Upgrade with: pip install -U 'transformers>=5.3'"
        )


class _PageRequest:
    """One predict call: the page image. The runner calls ``to``."""

    def __init__(self, image: Image.Image):
        self.image = image

    def to(self, _device):
        return self


class LibrePaddleOCRVL(LibreVLMModel):
    """PaddleOCR-VL document parsing as a LibreVLM ``ocr`` family."""

    FAMILY = "paddleocrvl"
    FILENAME_PREFIX = "LibrePaddleOCRVL"
    HF_REPOS: ClassVar[Dict[str, str]] = {"0.9b": "LibreYOLO/LibrePaddleOCRVL0.9b"}
    # Pinned mirror commit: the VL snapshot at the root, PP-DocLayoutV2 in layout/.
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "0.9b": "9fe2730f6d29e4a3805175312114ef70d1e4c9d6"
    }
    # Nominal: the layout stage runs at 800x800 and the recognizer resizes
    # every region within its own pixel budget.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"0.9b": layout.LAYOUT_INPUT_SIZE}
    SUPPORTED_TASKS = ("ocr",)
    DEFAULT_TASK = "ocr"
    # Upstream local inference: greedy, no repetition penalty, 4096 tokens.
    MAX_NEW_TOKENS = 4096
    REPETITION_PENALTY = 1.0
    DEFAULT_SCORE = 1.0
    SNAPSHOT_IGNORE_PATTERNS: ClassVar[tuple[str, ...]] = (
        *LibreVLMModel.SNAPSHOT_IGNORE_PATTERNS,
        "*.py",
        "*.pdiparams",
        "*.pdmodel",
    )
    TRAIN_UNSUPPORTED_REASON = (
        "PaddleOCR-VL is inference-only in LibreYOLO: the upstream fine-tuning "
        "recipe lives in the Paddle training stack and has no PyTorch port."
    )

    def __init__(
        self,
        size: str = "0.9b",
        *,
        use_layout_detection: bool = True,
        prompt_label: str = "ocr",
        use_chart_recognition: bool = False,
        layout_threshold: Optional[float | Dict[int, float]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            size: ``"0.9b"`` (the only released size).
            use_layout_detection: Run PP-DocLayoutV2 and recognize each region
                (default). ``False`` treats the whole image as one region of
                kind ``prompt_label``, for pre-cropped elements.
            prompt_label: Region kind without layout detection: ``"ocr"``,
                ``"table"``, ``"formula"`` or ``"chart"``.
            use_chart_recognition: Recognize ``chart`` regions into tables
                (upstream default: charts are skipped like pictures).
            layout_threshold: Override the upstream per-class layout score
                thresholds with one float or a ``{class_id: float}`` dict.
            min_pixels / max_pixels: Per-region pixel budget of the vision
                encoder (upstream defaults ``112896`` / ``1003520``).
        """
        prompt_label = str(prompt_label).strip().lower()
        if prompt_label not in _PROMPT_LABELS:
            raise ValueError(
                f"prompt_label must be one of {list(_PROMPT_LABELS)}, got {prompt_label!r}"
            )
        self.use_layout_detection = bool(use_layout_detection)
        self.prompt_label = prompt_label
        self.use_chart_recognition = bool(use_chart_recognition)
        self.layout_threshold = layout_threshold
        self.min_pixels = int(min_pixels) if min_pixels else blocks.DEFAULT_MIN_PIXELS
        self.max_pixels = int(max_pixels) if max_pixels else blocks.DEFAULT_MAX_PIXELS
        self.layout_model: Optional[nn.Module] = None
        kwargs.setdefault("names", list(layout.LAYOUT_LABELS))
        super().__init__(size=size, **kwargs)

    # =========================================================================
    # Loading
    # =========================================================================

    def _load_pretrained(self, snapshot_dir: str):
        _check_transformers()
        from transformers import (
            AutoModelForImageTextToText,
            AutoModelForObjectDetection,
            AutoProcessor,
        )

        model = AutoModelForImageTextToText.from_pretrained(
            snapshot_dir, dtype=self._resolve_dtype()
        )
        # The PIL image backend reproduces upstream's resize bit for bit; the
        # default torchvision backend shifts pixels by up to 0.1 and changes
        # greedy outputs on some crops.
        try:
            processor = AutoProcessor.from_pretrained(snapshot_dir, backend="pil")
        except (TypeError, ValueError):
            processor = AutoProcessor.from_pretrained(snapshot_dir)
        layout_dir = Path(snapshot_dir) / LAYOUT_SUBDIR
        if not (layout_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"PP-DocLayoutV2 weights are missing: {layout_dir}. Delete the "
                "snapshot directory to re-download it."
            )
        # The DETR head stays fp32 on every device: reduced precision moves
        # boxes and can flip the per-class thresholds.
        self.layout_model = AutoModelForObjectDetection.from_pretrained(
            str(layout_dir), dtype=torch.float32
        ).eval()
        return model, processor

    def set_classes(self, classes: list) -> "LibrePaddleOCRVL":
        if [str(c) for c in classes] != list(layout.LAYOUT_LABELS):
            raise ValueError(
                "PaddleOCR-VL reads every layout region; it has no vocabulary "
                "to set. Its names are the 25 PP-DocLayoutV2 classes."
            )
        return super().set_classes(classes)

    # =========================================================================
    # Recognition
    # =========================================================================

    def _encode(
        self,
        image: Image.Image,
        prompt: str,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        size = {
            "shortest_edge": int(min_pixels or self.min_pixels),
            "longest_edge": int(max_pixels or self.max_pixels),
        }
        inputs = self.processor(
            text=[text], images=[image], size=size, return_tensors="pt"
        ).to(self.device)
        return self._prepare_generation_inputs(inputs)

    def recognize(
        self,
        image: Image.Image,
        prompt: str,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Greedy-decode one region crop.

        Returns ``{"text", "confidence", "input_ids", "token_ids"}``; the ids
        are the exact prompt and generated sequences (used by the parity
        check).
        """
        inputs = self._encode(image, prompt, min_pixels, max_pixels)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        prompt_len = int(inputs["input_ids"].shape[1])
        new_tokens = out.sequences[0, prompt_len:]
        confidence = self.DEFAULT_SCORE
        if out.scores:
            logprobs = [
                torch.log_softmax(step[0].float(), dim=-1)[tok]
                for step, tok in zip(out.scores, new_tokens)
            ]
            if logprobs:
                confidence = float(torch.stack(logprobs).mean().exp())
        text = self.processor.batch_decode(
            new_tokens[None],
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
        )[0]
        return {
            "text": text,
            "confidence": confidence,
            "input_ids": inputs["input_ids"][0].tolist(),
            "token_ids": new_tokens.tolist(),
        }

    # =========================================================================
    # Layout
    # =========================================================================

    def detect_layout(self, image: ImageInput, color_format: str = "auto") -> List[dict]:
        """PP-DocLayoutV2 regions in reading order, before the overlap filter.

        Each region is ``{"cls_id", "label", "score", "coordinate"}`` with
        ``xyxy`` pixel coordinates on the original image.
        """
        img = ImageLoader.load(image, color_format=color_format)
        rgb = np.asarray(img.convert("RGB"))
        return self._detect_layout(rgb)

    def _detect_layout(self, rgb: np.ndarray) -> List[dict]:
        detector = self.layout_model.to(self.device)
        pixel_values = layout.preprocess_layout(rgb).to(self.device)
        with torch.no_grad():
            out = detector(pixel_values=pixel_values)
        height, width = rgb.shape[:2]
        rows = layout.raw_layout_boxes(
            out.logits, out.pred_boxes, out.order_logits, (width, height)
        )
        return layout.postprocess_layout(
            rows, (width, height), threshold=self.layout_threshold
        )

    # =========================================================================
    # InferenceRunner hooks
    # =========================================================================

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        return _PageRequest(img), img, img.size, 1.0

    def _forward(self, request: _PageRequest) -> List[dict]:
        rgb = np.asarray(request.image.convert("RGB"))
        height, width = rgb.shape[:2]
        if self.use_layout_detection:
            regions = layout.filter_overlap_boxes(self._detect_layout(rgb))
            chart = self.use_chart_recognition
        else:
            label = {"formula": "display_formula", "ocr": "text"}.get(
                self.prompt_label, self.prompt_label
            )
            regions = [
                {
                    "cls_id": -1,
                    "label": label,
                    "score": 1.0,
                    "coordinate": [0.0, 0.0, float(width), float(height)],
                }
            ]
            chart = self.use_chart_recognition or self.prompt_label == "chart"
        records = []
        for region in regions:
            prepared = blocks.prepare_region(rgb, region, use_chart_recognition=chart)
            if prepared is None:
                continue
            crop, prompt = prepared
            rec = self.recognize(Image.fromarray(crop), prompt)
            records.append(
                {
                    **region,
                    "text": blocks.clean_content(rec["text"], region["label"]),
                    "confidence": rec["confidence"],
                }
            )
        return records

    def _postprocess(
        self,
        output: List[dict],
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        # ``conf`` does not gate regions: the layout stage applies the upstream
        # per-class thresholds (override with ``layout_threshold=``).
        records = list(output)[: max(0, int(max_det))]
        polygons, texts, confs, det_confs, labels = [], [], [], [], []
        for record in records:
            x1, y1, x2, y2 = record["coordinate"]
            polygons.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            texts.append(record["text"])
            confs.append(record["confidence"])
            det_confs.append(record["score"])
            labels.append(record["label"])
        return {
            "ocr": {
                "polygons": polygons,
                "texts": texts,
                "confidences": confs,
                "det_confidences": det_confs,
                "labels": labels,
                "markdown": build_markdown(
                    {"label": label, "text": text} for label, text in zip(labels, texts)
                ),
            }
        }

    # =========================================================================
    # Validation
    # =========================================================================

    def val(self, *args, **kwargs):
        """Validate on an OCR dataset (``docs/dataset_schema.md``, ``ocr``).

        Reports detection hmean, end-to-end F1 and 1-NED through the shared
        OCR validator, which drives this model's own ``predict``.
        """
        return BaseModel.val(self, *args, **kwargs)


__all__ = ["LibrePaddleOCRVL"]
