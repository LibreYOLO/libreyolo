# SPDX-License-Identifier: Apache-2.0
# Prompts, the layout category map, the nested-box filter and the crop rules
# are adapted from Falcon OCR (Technology Innovation Institute, Apache-2.0):
# modeling_falcon_ocr.py of tiiuae/Falcon-OCR @
# 0bc28f89c90fbe8bdb063097fab144d4ce4c19b5, also published in
# https://github.com/tiiuae/Falcon-Perception (commit
# c457916c9974efbacfa91f0f6ecc2c49c6543e56). See NOTICE.
"""LibreYOLO wrapper for Falcon OCR (TII, Apache-2.0 code and weights).

Falcon OCR is a 0.27B early-fusion vision-language model for documents: one
transformer reads image patches and text tokens together (patches attend to
each other bidirectionally, text decodes causally) and writes the page as
text, with LaTeX for formulas and HTML for tables. LibreYOLO runs a PyTorch
port of the release (no remote code, no Triton, no FlexAttention), loaded
from a byte-identical LibreYOLO mirror.

    from libreyolo import LibreVLM
    model = LibreVLM("falcon-ocr")                # task "ocr", whole page
    result = model.predict("page.png")
    result.ocr.markdown                           # the page text
    LibreVLM("falcon-ocr", layout=True).predict("page.png").ocr.labels
    model.chat("crop.png", "table")               # raw generation

Two modes, as upstream:

- End to end (default, upstream ``generate``): the whole page is read in one
  pass. ``Results.ocr`` holds one region covering the image with the text,
  label ``"plain"`` (upstream's own convention for an unsplit page), and
  ``Results.ocr.markdown`` is the same text.
- ``layout=True`` (upstream ``generate_with_layout``): LibreYOLO's
  PP-DocLayoutV3 port finds regions in reading order, nested boxes are
  dropped, and each region crop is read with its category prompt. One
  ``Results.ocr`` region per crop, ``labels`` is the layout class and
  ``markdown`` joins the texts with blank lines.

Prompts (``prompt=`` or ``set_prompt()``): a category (``plain``, ``text``,
``table``, ``formula``, ``caption``, ``footnote``, ``list-item``,
``page-footer``, ``page-header``, ``section-header``, ``title``) or a custom
instruction. Upstream ``generate`` always uses the ``plain`` instruction
whatever ``category`` says; here the chosen category is applied.

Decoding is greedy (upstream default ``temperature=0``) with a budget of
``max_new_tokens`` (default 8192) rounded up to the 128-token block, as
upstream. A generative model emits no score, so ``conf`` and ``det_conf``
are 1.0 end to end; in layout mode ``det_conf`` is the layout score.

Devices: CUDA runs bf16 weights under ``torch.autocast`` (fp16 without bf16
support); CPU and MPS run float32 (about 1.1 GB of weights). Inference only:
``train()`` and ``export()`` raise; ``val()`` runs the OCR validator.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..base.model import BaseModel
from ..vlm.base import LibreVLMModel
from .network import FalconOCRConfig, FalconOCRNetwork
from .processing import (
    MAX_DIMENSION,
    MIN_DIMENSION,
    FalconOCRTokens,
    prepare_inputs,
)

logger = logging.getLogger(__name__)

IMAGE_TOKEN = "<|image|>"
END_TOKENS = ("<|end_of_query|>", "<|end_of_text|>")

CATEGORY_PROMPTS: Dict[str, str] = {
    "plain": "Extract the text content from this image.",
    "formula": "Extract the formula content from this image.",
    "table": "Extract the table content from this image.",
    "text": "Extract the text content from this image.",
    "caption": "Extract the caption content from this image.",
    "footnote": "Extract the footnote content from this image.",
    "list-item": "Extract the list-item content from this image.",
    "page-footer": "Extract the page-footer content from this image.",
    "page-header": "Extract the page-header content from this image.",
    "section-header": "Extract the section-header content from this image.",
    "title": "Extract the title content from this image.",
}

# Upstream's layout-class -> prompt-category map, verbatim. Classes missing
# here (PP-DocLayoutV3's display_formula, inline_formula, vertical_text,
# header_image, footer_image) are skipped, as upstream skips them.
LAYOUT_TO_OCR_CATEGORY: Dict[str, Optional[str]] = {
    "text": "text",
    "table": "table",
    "formula": "formula",
    "caption": "caption",
    "footnote": "footnote",
    "list-item": "list-item",
    "title": "title",
    "header": "text",
    "footer": "page-footer",
    "number": "text",
    "figure_title": "caption",
    "paragraph_title": "section-header",
    "doc_title": "title",
    "reference_content": "text",
    "reference": "text",
    "abstract": "text",
    "aside_text": "text",
    "content": "text",
    "formula_number": "text",
    "vision_footnote": "footnote",
    "algorithm": "text",
    "page-footer": "page-footer",
    "page-header": "page-header",
    "section-header": "section-header",
    "image": None,
    "picture": None,
    "figure": None,
    "chart": None,
    "seal": None,
}

LAYOUT_THRESHOLD = 0.3
CONTAINMENT_THRESHOLD = 0.8
MIN_CROP_DIM = 16


def wrap_prompt(instruction: str) -> str:
    """The release's chat-free prompt: image, instruction, OCR task token."""
    return f"{IMAGE_TOKEN}{instruction}\n<|OCR_PLAIN|>"


def resolve_prompt(prompt: Optional[str]) -> str:
    """A category name, a bare instruction, or a full prompt with ``<|image|>``."""
    if prompt is None:
        prompt = "plain"
    text = str(prompt)
    key = text.strip().lower()
    if key in CATEGORY_PROMPTS:
        return wrap_prompt(CATEGORY_PROMPTS[key])
    if IMAGE_TOKEN not in text:
        return wrap_prompt(text)
    if text.count(IMAGE_TOKEN) != 1:
        raise ValueError("Falcon OCR prompts take exactly one <|image|> placeholder.")
    return text


def clean_text(text: str) -> str:
    for token in END_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def _box_area(b) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _intersection_area(a, b) -> float:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def filter_nested(
    detections: List[dict], containment_threshold: float = CONTAINMENT_THRESHOLD
) -> List[dict]:
    """Drop any box mostly contained in a strictly larger one (upstream rule)."""
    areas = [_box_area(d["bbox"]) for d in detections]
    keep = []
    for i, det in enumerate(detections):
        nested = False
        for j, other in enumerate(detections):
            if i == j or areas[j] <= areas[i]:
                continue
            area = areas[i]
            ratio = (
                0.0
                if area <= 0
                else _intersection_area(det["bbox"], other["bbox"]) / area
            )
            if ratio > containment_threshold:
                nested = True
                break
        if not nested:
            keep.append(det)
    return keep


def layout_crops(
    image: Image.Image,
    detections: List[dict],
    max_dimension: int = MAX_DIMENSION,
) -> List[Tuple[dict, Image.Image, str]]:
    """``(detection, crop, prompt)`` per readable region, upstream's crop rules.

    An empty layout, or a lone ``image`` region, reads the whole page with the
    ``plain`` prompt (detection ``None``).
    """
    if not detections or (
        len(detections) == 1 and detections[0]["category"].strip().lower() == "image"
    ):
        return [(None, image, wrap_prompt(CATEGORY_PROMPTS["plain"]))]
    width, height = image.size
    out = []
    for det in detections:
        category = LAYOUT_TO_OCR_CATEGORY.get(det["category"].strip().lower())
        if category is None:
            continue
        x1, y1, x2, y2 = det["bbox"]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width, int(x2 + 0.5)), min(height, int(y2 + 0.5))
        cw, ch = x2 - x1, y2 - y1
        if cw < MIN_CROP_DIM or ch < MIN_CROP_DIM:
            continue
        short, long = sorted((cw, ch))
        resized_short = (
            short * (max_dimension / long) if long > max_dimension else short
        )
        if resized_short < MIN_CROP_DIM:
            continue
        crop = image.crop((x1, y1, x2, y2))
        out.append(
            (
                det,
                crop,
                wrap_prompt(CATEGORY_PROMPTS.get(category, CATEGORY_PROMPTS["plain"])),
            )
        )
    return out


def load_tokenizer(snapshot_dir: str | Path):
    """The release tokenizer through transformers' native fast class.

    ``AutoTokenizer`` would follow ``config.json``'s ``auto_map`` and ask for
    remote code; the tokenizer itself is a plain ``tokenizer.json``.
    """
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(str(snapshot_dir))


def build_network(snapshot_dir: str | Path) -> Tuple[FalconOCRNetwork, dict]:
    """Build on the meta device and load ``model.safetensors`` strictly."""
    from safetensors.torch import load_file

    snapshot_dir = Path(snapshot_dir)
    config = json.loads((snapshot_dir / "config.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        net = FalconOCRNetwork(FalconOCRConfig.from_dict(config))
    state = load_file(str(snapshot_dir / "model.safetensors"), device="cpu")
    net.load_state_dict(state, strict=True, assign=True)
    net.reset_rope_cache()
    leftover = [
        n
        for n, t in list(net.named_parameters()) + list(net.named_buffers())
        if t.is_meta
    ]
    if leftover:
        raise RuntimeError(f"Falcon OCR tensors left uninitialized: {leftover[:5]}")
    return net, config


class _FalconOCRRequest:
    """One predict call: the image. ``to`` is a no-op."""

    def __init__(self, image: Image.Image):
        self.image = image

    def to(self, *_args, **_kwargs):
        return self


class LibreFalconOCR(LibreVLMModel):
    """Falcon OCR as an ``ocr`` family of the LibreVLM tier."""

    FAMILY = "falconocr"
    FILENAME_PREFIX = "LibreFalconOCR"
    # Byte-identical mirror (weights, config, tokenizer; no remote code) of
    # tiiuae/Falcon-OCR @ 0bc28f89c90fbe8bdb063097fab144d4ce4c19b5, pinned.
    HF_REPOS: ClassVar[Dict[str, str]] = {"270m": "LibreYOLO/LibreFalconOCR270m"}
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "270m": "00657b06ef217f0eb87120141adf0ea2971c9b41",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"270m": MAX_DIMENSION}
    SUPPORTED_TASKS = ("ocr",)
    DEFAULT_TASK = "ocr"
    MAX_NEW_TOKENS = 8192
    PROMPTS: ClassVar[Dict[str, str]] = CATEGORY_PROMPTS
    LAYOUT_WEIGHTS: ClassVar[str] = "LibrePPDocLayoutV3l.pt"
    SNAPSHOT_IGNORE_PATTERNS: ClassVar[tuple] = (
        LibreVLMModel.SNAPSHOT_IGNORE_PATTERNS + ("*.py", "assets/*", ".eval_results/*")
    )
    TRAIN_UNSUPPORTED_REASON = (
        "Falcon OCR is inference-only in LibreYOLO: the release ships no "
        "fine-tuning recipe."
    )

    def __init__(
        self,
        size: str = "270m",
        *,
        layout: bool = False,
        layout_threshold: float = LAYOUT_THRESHOLD,
        min_dimension: int = MIN_DIMENSION,
        max_dimension: int = MAX_DIMENSION,
        **kwargs,
    ):
        """
        Args:
            size: ``"270m"`` (the only release).
            layout: Run PP-DocLayoutV3 first and read each region (upstream
                ``generate_with_layout``). Default reads the whole page.
            layout_threshold: Layout score threshold (upstream 0.3).
            min_dimension / max_dimension: Page side limits before patching
                (upstream 64 / 1536).
            **kwargs: ``prompt`` (a category or a custom instruction),
                ``device``, ``max_new_tokens``; see ``LibreVLMModel``.
        """
        kwargs.setdefault("nb_classes", 1)
        kwargs.pop("names", None)
        prompt = kwargs.pop("prompt", None)
        self.layout = bool(layout)
        self.layout_threshold = float(layout_threshold)
        self.min_dimension = int(min_dimension)
        self.max_dimension = int(max_dimension)
        self._layout_model = None
        super().__init__(size, **kwargs)
        self.names = {0: "text"}
        self.nb_classes = 1
        self._name_to_id = {"text": 0}
        self.set_prompt(prompt)

    def set_prompt(self, prompt: Optional[str]) -> "LibreFalconOCR":
        """Set the end-to-end prompt: a category or a custom instruction.

        Sticky across later ``predict()`` calls; returns ``self``. Layout mode
        picks the prompt from each region's class instead.
        """
        self._custom_prompt = resolve_prompt(prompt)
        return self

    @property
    def prompt(self) -> str:
        return self._custom_prompt

    # ------------------------------------------------------------------ loading

    def _load_pretrained(self, snapshot_dir: str):
        net, config = build_network(snapshot_dir)
        golden = net.freqs_cis_golden.clone()
        net = net.to(dtype=self._resolve_dtype()).to(self.device).eval()
        # Module.to(dtype) would turn the complex 1D table real and round the
        # 2D table: rebuild the first (complex64, as upstream) and keep the
        # second in float32 (the rotary math runs in float32).
        net.reset_rope_cache()
        net.freqs_cis_golden = golden.to(self.device)
        tokenizer = load_tokenizer(snapshot_dir)
        self._tokens = FalconOCRTokens.from_config(config)
        self._stop_ids = [
            self._tokens.eos_id,
            tokenizer.convert_tokens_to_ids("<|end_of_query|>"),
        ]
        return net, tokenizer

    def _autocast_dtype(self) -> Optional[torch.dtype]:
        dtype = getattr(self, "_model_dtype", torch.float32)
        if self.device.type == "cuda" and dtype != torch.float32:
            return dtype
        return None

    def _get_layout_model(self):
        if self._layout_model is None:
            from ... import LibreYOLO

            self._layout_model = LibreYOLO(self.LAYOUT_WEIGHTS, device=str(self.device))
        return self._layout_model

    # ---------------------------------------------------------------- inference

    def _read(self, image: Image.Image, prompt: str, max_new_tokens=None) -> str:
        tok = self.processor
        chunks = [tok.encode(c) for c in prompt.split(IMAGE_TOKEN)]
        inputs = prepare_inputs(
            image, chunks, self._tokens, self.min_dimension, self.max_dimension
        )
        device = self.device
        new_ids = self.model.generate(
            inputs.ids.to(device),
            inputs.pos_t.to(device),
            inputs.pos_hw.to(device),
            inputs.patches.to(device),
            max_new_tokens=max_new_tokens or self.MAX_NEW_TOKENS,
            stop_ids=self._stop_ids,
            autocast_dtype=self._autocast_dtype(),
        )
        return clean_text(tok.decode(new_ids, skip_special_tokens=False))

    def chat(
        self,
        image: ImageInput,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        color_format: str = "auto",
    ) -> str:
        """Raw generation for a category, an instruction or a full prompt."""
        img = ImageLoader.load(image, color_format=color_format)
        return self._read(img, resolve_prompt(prompt), max_new_tokens)

    def _detect_layout(self, image: Image.Image) -> List[dict]:
        result = self._get_layout_model()(image, conf=self.layout_threshold)
        if result.boxes is None or len(result.boxes) == 0:
            return []
        names = result.names
        dets = []
        # Rows are already in reading order.
        for box, score, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            dets.append(
                {"category": names[int(cls)], "bbox": box, "score": float(score)}
            )
        return dets

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        return _FalconOCRRequest(img), img, img.size, 1.0

    def _forward(self, inputs: _FalconOCRRequest):
        image = inputs.image
        if not self.layout:
            return [(None, self._read(image, self._custom_prompt))]
        dets = filter_nested(self._detect_layout(image))
        return [
            (det, self._read(crop, prompt))
            for det, crop, prompt in layout_crops(image, dets, self.max_dimension)
        ]

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
        return build_ocr_dict(output, original_size, max_det, self.DEFAULT_SCORE)

    # --------------------------------------------------------------- validation

    def val(self, data: Optional[str] = None, **kwargs):
        """OCR validation (detection hmean, end-to-end F1, 1-NED)."""
        kwargs.setdefault("batch", 1)
        return BaseModel.val(self, data=data, **kwargs)


def build_ocr_dict(
    items: List[Tuple[Optional[dict], str]],
    original_size: Tuple[int, int],
    max_det: int,
    score: float,
) -> Dict:
    """``[(detection or None, text)]`` as the ``{"ocr": ...}`` payload.

    ``None`` means the whole page (label ``plain``); regions keep the given
    (reading) order, and empty transcripts are dropped from the regions.
    """
    width, height = original_size
    polygons, texts, labels, det_conf = [], [], [], []
    for det, text in items:
        if not text or len(polygons) >= max_det:
            continue
        if det is None:
            x1, y1, x2, y2 = 0.0, 0.0, float(width), float(height)
            label, dscore = "plain", score
        else:
            x1, y1, x2, y2 = (float(v) for v in det["bbox"])
            label, dscore = det["category"], float(det["score"])
        polygons.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        texts.append(text)
        labels.append(label)
        det_conf.append(dscore)
    return {
        "ocr": {
            "polygons": polygons,
            "texts": texts,
            "confidences": [score] * len(texts),
            "det_confidences": det_conf,
            "labels": labels,
            "markdown": "\n\n".join(texts),
        }
    }


__all__ = [
    "CATEGORY_PROMPTS",
    "LAYOUT_TO_OCR_CATEGORY",
    "LibreFalconOCR",
    "build_network",
    "build_ocr_dict",
    "filter_nested",
    "layout_crops",
    "load_tokenizer",
    "resolve_prompt",
]
