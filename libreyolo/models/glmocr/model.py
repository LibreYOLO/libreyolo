"""LibreYOLO wrapper for GLM-OCR, a two-stage document parser.

GLM-OCR (Z.ai, MIT weights, Apache-2.0 SDK) pairs the PP-DocLayoutV3 layout
detector (25 layout classes plus reading order) with GLM-OCR-0.9B, a CogViT
vision encoder and a GLM-0.5B decoder that reads one region at a time from a
task prompt: ``Text Recognition:``, ``Table Recognition:`` (HTML) or
``Formula Recognition:`` (LaTeX).

Both networks run through the native transformers classes
(``GlmOcrForConditionalGeneration`` and ``PPDocLayoutV3ForObjectDetection``,
transformers >= 5.3); no remote code is executed. The pipeline around them
(layout NMS and merges, label routing, crops, the resize and JPEG round trip,
decoding settings, content clean-up, block merges and page markdown) is a port
of the official GLM-OCR SDK, see ``NOTICE``.

    from libreyolo import LibreVLM
    model = LibreVLM("glm-ocr")
    result = model.predict("page.png")
    result.ocr.texts        # one transcript per block, in reading order
    result.ocr.labels       # layout class per block ("text", "table", ...)
    result.ocr.markdown     # the whole page as markdown
    LibreVLM("glm-ocr", use_layout_detection=False, prompt_label="table")
    model.chat("id_card.png", 'Extract as JSON: {"name": "", "date": ""}')

Confidence: ``Results.ocr.conf`` is the geometric mean of the generated
tokens' probabilities (raw logits, before the repetition penalty),
``Results.ocr.det_conf`` the layout score. Regions are axis-aligned boxes
written as 4-point quads.

Devices: CUDA runs the recognizer in bf16 (fp16 without bf16 support) and
the layout detector in fp32; CPU and MPS run both in fp32.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..base.model import BaseModel
from ..vlm.base import _INSTALL_HINT, LibreVLMModel
from . import formatter, layout, regions

logger = logging.getLogger(__name__)

LAYOUT_SUBDIR = "layout"
_MIN_TRANSFORMERS = (5, 3)
_PROMPT_LABELS = tuple(regions.TASK_PROMPTS)


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
            "GLM-OCR needs transformers >= 5.3 (native GLM-OCR and "
            f"PP-DocLayoutV3 classes); found {transformers.__version__}. "
            "Upgrade with: pip install -U 'transformers>=5.3'"
        )


class _PageRequest:
    """One predict call: the page image. The runner calls ``to``."""

    def __init__(self, image: Image.Image):
        self.image = image

    def to(self, _device):
        return self


class LibreGLMOCR(LibreVLMModel):
    """GLM-OCR document parsing as a LibreVLM ``ocr`` family."""

    FAMILY = "glmocr"
    FILENAME_PREFIX = "LibreGLMOCR"
    HF_REPOS: ClassVar[Dict[str, str]] = {"0.9b": "LibreYOLO/LibreGLMOCR0.9b"}
    # Pinned mirror commit: GLM-OCR at the root, PP-DocLayoutV3 in layout/.
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "0.9b": "196bf843d3b7e8b8e7bc55edd1580c2455a7fb02"
    }
    # Nominal: the layout stage runs at 800x800 and the recognizer resizes
    # every region within its own pixel budget.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"0.9b": 800}
    SUPPORTED_TASKS = ("ocr",)
    DEFAULT_TASK = "ocr"
    # SDK request defaults: greedy (top_k=1), repetition penalty 1.1,
    # max_tokens 8192.
    MAX_NEW_TOKENS = 8192
    REPETITION_PENALTY = 1.1
    DEFAULT_SCORE = 1.0
    SNAPSHOT_IGNORE_PATTERNS: ClassVar[tuple[str, ...]] = (
        *LibreVLMModel.SNAPSHOT_IGNORE_PATTERNS,
        "*.py",
        "*.yml",
    )
    TRAIN_UNSUPPORTED_REASON = (
        "GLM-OCR is inference-only in LibreYOLO: no fine-tuning recipe is "
        "released upstream."
    )

    def __init__(
        self,
        size: str = "0.9b",
        *,
        use_layout_detection: bool = True,
        prompt_label: str = "text",
        layout_threshold: Optional[float | Dict[int, float]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        image_format: Optional[str] = regions.IMAGE_FORMAT,
        **kwargs,
    ):
        """
        Args:
            size: ``"0.9b"`` (the only released size).
            use_layout_detection: Run PP-DocLayoutV3 and recognize each region
                (default). ``False`` reads the whole image as one region of
                kind ``prompt_label``, for pre-cropped elements.
            prompt_label: Region kind without layout detection: ``"text"``,
                ``"table"`` or ``"formula"``.
            layout_threshold: Layout score threshold, one float (upstream
                ``0.3``) or a ``{class_id: float}`` dict; classes missing from
                the dict use ``0.3``.
            min_pixels / max_pixels: Per-region resize budget before the
                recognizer (upstream ``12544`` / ``71372800``).
            image_format: Format of the region round trip the upstream SDK
                applies before recognition (``"JPEG"``); ``None`` skips it.
            prompt: Optional override sent to the recognizer for every region
                instead of the routed task prompt.
        """
        prompt_label = str(prompt_label).strip().lower()
        if prompt_label not in _PROMPT_LABELS:
            raise ValueError(
                f"prompt_label must be one of {list(_PROMPT_LABELS)}, got {prompt_label!r}"
            )
        self.use_layout_detection = bool(use_layout_detection)
        self.prompt_label = prompt_label
        self.layout_threshold = layout_threshold
        self.min_pixels = int(min_pixels) if min_pixels else regions.MIN_PIXELS
        self.max_pixels = int(max_pixels) if max_pixels else regions.MAX_PIXELS
        self.image_format = image_format or None
        self.layout_model: Optional[nn.Module] = None
        self.layout_processor = None
        kwargs.setdefault("names", list(layout.LAYOUT_LABELS))
        super().__init__(size=size, **kwargs)

    # =========================================================================
    # Loading
    # =========================================================================

    def _load_pretrained(self, snapshot_dir: str):
        _check_transformers()
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            PPDocLayoutV3ForObjectDetection,
        )

        try:
            from transformers import PPDocLayoutV3ImageProcessor
        except ImportError:
            # Before the image-processor backends were unified the same
            # torchvision implementation carried the ``Fast`` suffix.
            from transformers import (
                PPDocLayoutV3ImageProcessorFast as PPDocLayoutV3ImageProcessor,
            )

        model = AutoModelForImageTextToText.from_pretrained(
            snapshot_dir, dtype=self._resolve_dtype()
        )
        processor = AutoProcessor.from_pretrained(snapshot_dir)
        layout_dir = Path(snapshot_dir) / LAYOUT_SUBDIR
        if not (layout_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"PP-DocLayoutV3 weights are missing: {layout_dir}. Delete the "
                "snapshot directory to re-download it."
            )
        self.layout_processor = PPDocLayoutV3ImageProcessor.from_pretrained(
            str(layout_dir)
        )
        self.layout_model = PPDocLayoutV3ForObjectDetection.from_pretrained(
            str(layout_dir), dtype=torch.float32
        ).eval()
        return model, processor

    def set_classes(self, classes: list) -> "LibreGLMOCR":
        if [str(c) for c in classes] != list(layout.LAYOUT_LABELS):
            raise ValueError(
                "GLM-OCR reads every layout region; it has no vocabulary to "
                "set. Its names are the 25 PP-DocLayoutV3 classes."
            )
        return super().set_classes(classes)

    # =========================================================================
    # Recognition
    # =========================================================================

    def _encode(self, image: Image.Image, prompt: str):
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
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(
            self.device
        )
        return self._prepare_generation_inputs(inputs)

    def recognize(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Decode one already prepared region image (see ``prepare_region``).

        Returns ``{"text", "confidence", "input_ids", "token_ids"}``; the ids
        are the exact prompt and generated sequences (used by the parity
        check).
        """
        inputs = self._encode(image, prompt)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=self.REPETITION_PENALTY,
                use_cache=True,
                output_logits=True,
                return_dict_in_generate=True,
            )
        prompt_len = int(inputs["input_ids"].shape[1])
        new_tokens = out.sequences[0, prompt_len:]
        confidence = self.DEFAULT_SCORE
        logits = getattr(out, "logits", None)
        if logits:
            logprobs = [
                torch.log_softmax(step[0].float(), dim=-1)[tok]
                for step, tok in zip(logits, new_tokens)
            ]
            if logprobs:
                confidence = float(torch.stack(logprobs).mean().exp())
        text = self.processor.batch_decode(new_tokens[None], skip_special_tokens=True)[
            0
        ]
        return {
            "text": text.strip(),
            "confidence": confidence,
            "input_ids": inputs["input_ids"][0].tolist(),
            "token_ids": new_tokens.tolist(),
        }

    def prepare_region(self, crop: Image.Image) -> Optional[Image.Image]:
        """Resize and round-trip a crop as the SDK does; ``None`` if empty."""
        if crop.size[0] <= 0 or crop.size[1] <= 0:
            return None
        return regions.prepare_image(
            crop,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            image_format=self.image_format,
        )

    # =========================================================================
    # Layout
    # =========================================================================

    def _threshold_args(self) -> Tuple[float, Optional[Dict[int, float]]]:
        value = self.layout_threshold
        if value is None:
            return layout.LAYOUT_THRESHOLD, None
        if isinstance(value, dict):
            per_class = {int(k): float(v) for k, v in value.items()}
            return min([layout.LAYOUT_THRESHOLD, *per_class.values()]), per_class
        return float(value), None

    def detect_layout(
        self, image: ImageInput, color_format: str = "auto"
    ) -> List[dict]:
        """PP-DocLayoutV3 regions in reading order after the SDK post-processing.

        Each region is ``{"cls_id", "label", "score", "coordinate"}`` with
        integer ``xyxy`` pixel coordinates on the original image; headers,
        footers and other ``abandon`` classes are still included.
        """
        img = ImageLoader.load(image, color_format=color_format)
        return self._detect_layout(img.convert("RGB"))

    def _detect_layout(self, image: Image.Image) -> List[dict]:
        detector = self.layout_model.to(self.device)
        inputs = self.layout_processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = detector(**inputs)
        pre_threshold, per_class = self._threshold_args()
        target = torch.tensor([image.size[::-1]], device=self.device)
        raw = self.layout_processor.post_process_object_detection(
            outputs, threshold=pre_threshold, target_sizes=target
        )[0]
        rows = layout.raw_rows(raw)
        if per_class is not None and len(rows):
            keep = [
                row[1] >= per_class.get(int(row[0]), layout.LAYOUT_THRESHOLD)
                for row in rows
            ]
            rows = rows[keep]
        return layout.postprocess_layout(rows, image.size)

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

    def parse_page(self, image: Image.Image) -> List[dict]:
        """Run the whole pipeline on one RGB page; returns formatted blocks."""
        image = image.convert("RGB") if image.mode != "RGB" else image
        width, height = image.size
        if self.use_layout_detection:
            page_regions = layout.to_regions(self._detect_layout(image), image.size)
        else:
            page_regions = [
                {
                    "index": 0,
                    "label": {"formula": "display_formula"}.get(
                        self.prompt_label, self.prompt_label
                    ),
                    "score": 1.0,
                    "bbox_2d": [0, 0, 1000, 1000],
                    "task_type": self.prompt_label,
                }
            ]
        for region in page_regions:
            region["content"] = None
            region["confidence"] = self.DEFAULT_SCORE
            if region["task_type"] == "skip":
                continue
            if self.use_layout_detection:
                crop = regions.crop_region(image, region["bbox_2d"])
            else:
                crop = image
            prepared = self.prepare_region(crop)
            if prepared is None:
                continue
            # ``LibreVLM(..., prompt=...)`` overrides the routed task prompt.
            prompt = self._custom_prompt or regions.TASK_PROMPTS[region["task_type"]]
            rec = self.recognize(prepared, prompt)
            region["content"] = rec["text"]
            region["confidence"] = rec["confidence"]
        blocks = formatter.format_page(page_regions)
        for block in blocks:
            if self.use_layout_detection:
                block["coordinate"] = layout.bbox_to_pixels(
                    block["bbox_2d"], (width, height)
                )
            else:
                block["coordinate"] = [0, 0, width, height]
        return blocks

    def _forward(self, request: _PageRequest) -> List[dict]:
        return self.parse_page(request.image)

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
        # threshold (override with ``layout_threshold=``). Pictures carry no
        # text and are left out of the OCR regions and the markdown.
        blocks = list(output)
        markdown = formatter.page_markdown(blocks)
        blocks = [b for b in blocks if not b.get("is_image")][: max(0, int(max_det))]
        polygons, texts, confs, det_confs, labels = [], [], [], [], []
        for block in blocks:
            x1, y1, x2, y2 = (float(v) for v in block["coordinate"])
            polygons.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            texts.append(block["content"])
            confs.append(float(block.get("confidence", self.DEFAULT_SCORE)))
            det_confs.append(float(block.get("score", 1.0)))
            labels.append(block["native_label"])
        return {
            "ocr": {
                "polygons": polygons,
                "texts": texts,
                "confidences": confs,
                "det_confidences": det_confs,
                "labels": labels,
                "markdown": markdown,
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


__all__ = ["LibreGLMOCR"]
