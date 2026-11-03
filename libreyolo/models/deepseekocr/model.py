"""LibreYOLO wrapper for DeepSeek-OCR (deepseek-ai, MIT code and weights).

DeepSeek-OCR is a 3B mixture-of-experts vision-language model built for
documents: a SAM ViT-B plus CLIP-L encoder ("DeepEncoder") compresses each
page into a few hundred visual tokens, and a DeepSeek-V2 style MoE decoder
(about 570M active parameters) writes the text. LibreYOLO runs a vendored
PyTorch port of the release (no remote code), loaded from a byte-identical
LibreYOLO snapshot mirror.

    from libreyolo import LibreVLM
    model = LibreVLM("deepseek-ocr")              # task "ocr", Gundam resolution
    result = model.predict("page.png")
    result.ocr.texts, result.ocr.polygons         # one region per text line
    LibreVLM("deepseek-ocr", prompt="markdown").predict("page.png").ocr.markdown
    model.chat("page.png", "Parse the figure.")   # raw generation

Prompts (``prompt=`` or ``set_prompt()``): ``"ocr"`` (default, grounded text
lines into ``Results.ocr``), ``"markdown"`` (grounded layout blocks plus the
page as markdown in ``Results.ocr.markdown``, layout categories in ``Results.ocr.labels``),
``"free"`` (plain text in ``Results.ocr.markdown``, no regions), or any custom prompt string.

Resolutions (``resolution=``): ``tiny`` (512), ``small`` (640), ``base``
(1024), ``large`` (1280), ``gundam`` (default: 1024 global view plus 2 to 9
local 640 crops for pages larger than 640 px).

Boxes come on a 0 to 999 grid and are scaled to the original image as
axis-aligned 4-point polygons, sorted top to bottom then left to right as the
OCR contract specifies. A generative model emits no score, so ``conf`` and
``det_conf`` are a constant 1.0, the tier convention.

Devices: CUDA runs bf16 weights under ``torch.autocast`` exactly like the
release (fp16 on GPUs without bf16). CPU and MPS run float32 (about 13 GB of
memory) with eager attention; no flash-attention path is required. MPS has
not been verified on hardware. Inference only: ``train()`` and ``export()``
raise; ``val()`` runs the OCR validator.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ...postprocess.ppocr import sort_quads_reading_order
from ...utils.image_loader import ImageInput, ImageLoader
from ..base.model import BaseModel
from ..vlm.base import LibreVLMModel
from . import parsing
from .language import DeepSeekOCRTextConfig, RotaryEmbedding
from .network import DeepSeekOCRNetwork
from .processing import (
    RESOLUTIONS,
    DeepSeekOCRTokenizer,
    build_input_ids,
    normalize_prompt,
    prepare_views,
    resolve_resolution,
)

logger = logging.getLogger(__name__)

PROMPTS: Dict[str, str] = {
    "ocr": "<image>\n<|grounding|>OCR this image.",
    "markdown": "<image>\n<|grounding|>Convert the document to markdown.",
    "free": "<image>\nFree OCR.",
}


class _DeepSeekOCRRequest:
    """One predict call: the image and the prompt. ``to`` is a no-op."""

    def __init__(self, image: Image.Image, prompt: str):
        self.image = image
        self.prompt = prompt

    def to(self, *_args, **_kwargs):
        return self


def _load_safetensors(snapshot_dir: Path) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    index = snapshot_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        files = sorted(set(weight_map.values()))
    else:
        files = sorted(p.name for p in snapshot_dir.glob("*.safetensors"))
    state: Dict[str, torch.Tensor] = {}
    for name in files:
        state.update(load_file(str(snapshot_dir / name), device="cpu"))
    return state


def build_network(snapshot_dir: str | Path) -> DeepSeekOCRNetwork:
    """Build the network on the meta device and load the snapshot strictly."""
    snapshot_dir = Path(snapshot_dir)
    config = DeepSeekOCRTextConfig.from_json(snapshot_dir / "config.json")
    with torch.device("meta"):
        net = DeepSeekOCRNetwork(config)
    state = _load_safetensors(snapshot_dir)
    net.load_state_dict(state, strict=True, assign=True)
    # Non-persistent buffers are not in the checkpoint: rebuild them on CPU.
    backbone = net.model
    backbone.rotary_emb = RotaryEmbedding(
        config.hidden_size // config.num_attention_heads, config.rope_theta
    )
    emb = backbone.vision_model.embeddings
    emb.register_buffer(
        "position_ids",
        torch.arange(emb.num_positions).expand((1, -1)),
        persistent=False,
    )
    leftover = [n for n, b in net.named_buffers() if b.is_meta]
    leftover += [n for n, p in net.named_parameters() if p.is_meta]
    if leftover:
        raise RuntimeError(f"DeepSeek-OCR tensors left uninitialized: {leftover[:5]}")
    return net


class LibreDeepSeekOCR(LibreVLMModel):
    """DeepSeek-OCR as an ``ocr`` family of the LibreVLM tier."""

    FAMILY = "deepseekocr"
    FILENAME_PREFIX = "LibreDeepSeekOCR"
    # Byte-identical mirror of deepseek-ai/DeepSeek-OCR @ 9f30c71f (weights,
    # config, tokenizer; no remote code), pinned.
    HF_REPOS: ClassVar[Dict[str, str]] = {"1": "LibreYOLO/LibreDeepSeekOCR1"}
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "1": "a47622e3fb7279eaed206ab5d656158b71976b5b"
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"1": 1024}
    SUPPORTED_TASKS = ("ocr",)
    DEFAULT_TASK = "ocr"
    # Release defaults for ``infer`` (outside its benchmark ``eval_mode``).
    MAX_NEW_TOKENS = 8192
    NO_REPEAT_NGRAM_SIZE: ClassVar[int] = 20
    DEFAULT_RESOLUTION: ClassVar[str] = "gundam"
    PROMPTS: ClassVar[Dict[str, str]] = PROMPTS
    # Only weights, config and tokenizer are needed: the port never runs the
    # snapshot's Python files.
    SNAPSHOT_IGNORE_PATTERNS: ClassVar[tuple] = (
        LibreVLMModel.SNAPSHOT_IGNORE_PATTERNS
        + (
            "*.py",
            "assets/*",
            ".ipynb_checkpoints/*",
        )
    )
    TRAIN_UNSUPPORTED_REASON = (
        "DeepSeek-OCR is inference-only in LibreYOLO: the release ships no "
        "fine-tuning recipe, and its training is web-scale document pretraining."
    )

    def __init__(
        self,
        size: str = "1",
        *,
        resolution: Optional[str] = None,
        no_repeat_ngram_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            size: Model version; ``"1"`` is DeepSeek-OCR.
            resolution: ``tiny``, ``small``, ``base``, ``large`` or ``gundam``
                (default).
            no_repeat_ngram_size: Ban repeated n-grams of this length while
                decoding (release default 20; 0 disables).
            **kwargs: ``prompt`` (``"ocr"``, ``"markdown"``, ``"free"`` or a
                custom prompt), ``device``, ``max_new_tokens``; see
                ``LibreVLMModel``.
        """
        kwargs.setdefault("nb_classes", 1)
        kwargs.pop("names", None)
        prompt = kwargs.pop("prompt", None)
        self.resolution = resolution or self.DEFAULT_RESOLUTION
        self.no_repeat_ngram_size = (
            self.NO_REPEAT_NGRAM_SIZE
            if no_repeat_ngram_size is None
            else int(no_repeat_ngram_size)
        )
        super().__init__(size, **kwargs)
        self.names = {0: "text"}
        self.nb_classes = 1
        self._name_to_id = {"text": 0}
        self.set_prompt(prompt)

    # ------------------------------------------------------------------ options

    @property
    def resolution(self) -> str:
        return self._resolution

    @resolution.setter
    def resolution(self, value: str) -> None:
        resolve_resolution(value)
        self._resolution = str(value).strip().lower()

    def set_prompt(self, prompt: Optional[str]) -> "LibreDeepSeekOCR":
        """Set the predict prompt: ``"ocr"``, ``"markdown"``, ``"free"`` or text.

        Sticky across later ``predict()`` calls; returns ``self``.
        """
        if prompt is None:
            prompt = "ocr"
        key = str(prompt).strip().lower()
        text = self.PROMPTS.get(key, prompt)
        normalize_prompt(text)  # validates the <image> placeholder
        self._custom_prompt = text
        return self

    @property
    def prompt(self) -> str:
        return self._custom_prompt

    # ------------------------------------------------------------------ loading

    def _load_pretrained(self, snapshot_dir: str):
        net = build_network(snapshot_dir)
        dtype = self._resolve_dtype()
        net = net.to(dtype=dtype).to(self.device).eval()
        tokenizer = DeepSeekOCRTokenizer(snapshot_dir)
        return net, tokenizer

    def _autocast_dtype(self) -> Optional[torch.dtype]:
        dtype = getattr(self, "_model_dtype", torch.float32)
        if self.device.type == "cuda" and dtype != torch.float32:
            return dtype
        return None

    # ---------------------------------------------------------------- inference

    def _generate(self, image: Image.Image, prompt: str, max_new_tokens=None):
        views = prepare_views(image, self.resolution)
        ids, mask = build_input_ids(self.processor, prompt, views.num_tokens)
        dtype = self._model_dtype
        device = self.device
        crops = views.crops.to(dtype).to(device) if views.crops is not None else None
        return self.model.generate(
            torch.tensor(ids, dtype=torch.long, device=device),
            torch.tensor(mask, dtype=torch.bool, device=device),
            views.global_view.to(dtype).to(device),
            crops,
            views.grid,
            max_new_tokens=max_new_tokens or self.MAX_NEW_TOKENS,
            eos_token_id=self.processor.eos_token_id,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            autocast_dtype=self._autocast_dtype(),
        )

    def chat(
        self,
        image: ImageInput,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        color_format: str = "auto",
    ) -> str:
        """Raw generation for any prompt; returns the text with markup intact."""
        img = ImageLoader.load(image, color_format=color_format)
        new_ids = self._generate(img, prompt, max_new_tokens)
        return parsing.strip_eos(self.processor.decode(new_ids))

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        return _DeepSeekOCRRequest(img, self._custom_prompt), img, img.size, 1.0

    def _forward(self, inputs: _DeepSeekOCRRequest):
        return self._generate(inputs.image, inputs.prompt)

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
        text = parsing.strip_eos(self.processor.decode(list(output)))
        return build_ocr_dict(text, original_size, max_det, self.DEFAULT_SCORE)

    # --------------------------------------------------------------- validation

    def val(self, data: Optional[str] = None, **kwargs):
        """OCR validation (detection hmean, end-to-end F1, 1-NED)."""
        kwargs.setdefault("batch", 1)
        return BaseModel.val(self, data=data, **kwargs)


def build_ocr_dict(
    text: str, original_size: Tuple[int, int], max_det: int, score: float
) -> Dict:
    """Parsed generation as the ``{"ocr": ...}`` payload the runner wraps."""
    width, height = original_size
    polygons, texts, labels = [], [], []
    for item in parsing.parse_grounding(text):
        if not item["text"] or len(polygons) >= max_det:
            continue
        x1, y1, x2, y2 = parsing.grid_box_to_pixels(item["box"], width, height)
        polygons.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        texts.append(item["text"])
        labels.append(item["label"])
    order = sort_quads_reading_order(
        np.asarray(polygons, dtype=np.float32).reshape(-1, 4, 2)
    )
    polygons = [polygons[i] for i in order]
    texts = [texts[i] for i in order]
    labels = [labels[i] for i in order]
    return {
        "ocr": {
            "polygons": polygons,
            "texts": texts,
            "confidences": [score] * len(texts),
            "det_confidences": [score] * len(texts),
            "labels": labels,
            "markdown": parsing.to_page_text(text),
        }
    }


__all__ = [
    "LibreDeepSeekOCR",
    "PROMPTS",
    "RESOLUTIONS",
    "build_network",
    "build_ocr_dict",
]
