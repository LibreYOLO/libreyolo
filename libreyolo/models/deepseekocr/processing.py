"""DeepSeek-OCR input preparation: image views, prompt and token layout.

Follows ``DeepseekOCRForCausalLM.infer`` and its helpers in the DeepSeek-OCR
Hugging Face release (deepseek-ai/DeepSeek-OCR @
9f30c71f441d010e5429c532364a86705536c53a, MIT): the same PIL resize and pad
calls, the same crop-grid search, the same placeholder token layout. Two
deliberate differences from the release:

* the BOS token follows the tokenizer config (``add_bos_token`` and
  ``bos_token``) instead of a hard-coded id 0; for DeepSeek-OCR both give the
  same sequence;
* the tokenizer is read straight from ``tokenizer.json``. Rebuilding it from
  the ``LlamaTokenizerFast`` class name on transformers 5 swaps in a different
  pre-tokenizer and drops whitespace, which changes the prompt tokens.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

IMAGE_TOKEN = "<image>"
PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 4
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)
CROP_SIZE = 640

# name -> (base_size, image_size, crop_mode), as documented by the release.
RESOLUTIONS: Dict[str, Tuple[int, int, bool]] = {
    "tiny": (512, 512, False),
    "small": (640, 640, False),
    "base": (1024, 1024, False),
    "large": (1280, 1280, False),
    "gundam": (1024, 640, True),
}


def resolve_resolution(name: str) -> Tuple[int, int, bool]:
    key = str(name).strip().lower()
    if key not in RESOLUTIONS:
        raise ValueError(
            f"Unknown DeepSeek-OCR resolution {name!r}. "
            f"Choose one of: {', '.join(RESOLUTIONS)}."
        )
    return RESOLUTIONS[key]


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def crop_grid(width: int, height: int, min_num=2, max_num=9, image_size=CROP_SIZE):
    """Columns and rows of the local-crop grid for a ``width x height`` image."""
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    return find_closest_aspect_ratio(
        width / height, target_ratios, width, height, image_size
    )


def dynamic_preprocess(image: Image.Image, image_size: int = CROP_SIZE):
    """Resize to the best grid and cut it into ``image_size`` tiles (row-major)."""
    cols, rows = crop_grid(image.size[0], image.size[1], image_size=image_size)
    target_width, target_height = image_size * cols, image_size * rows
    resized = image.resize((target_width, target_height))
    tiles = []
    for i in range(cols * rows):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        tiles.append(resized.crop(box))
    return tiles, (cols, rows)


def to_tensor(image: Image.Image) -> torch.Tensor:
    """``ToTensor`` then ``Normalize(0.5, 0.5)``, as float32 ``(3, H, W)``."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    t = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous().float().div(255)
    mean = torch.tensor(IMAGE_MEAN, dtype=t.dtype).view(-1, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=t.dtype).view(-1, 1, 1)
    return t.sub(mean).div(std)


def _pad_color() -> Tuple[int, int, int]:
    return tuple(int(x * 255) for x in IMAGE_MEAN)


def num_queries(size: int) -> int:
    return math.ceil((size // PATCH_SIZE) / DOWNSAMPLE_RATIO)


def image_token_count(
    base_size: int, image_size: int, crop_mode: bool, grid: Tuple[int, int] = (1, 1)
) -> int:
    """Number of ``<image>`` placeholders the release lays out for one image."""
    if not crop_mode:
        n = num_queries(image_size)
        return (n + 1) * n + 1
    nb = num_queries(base_size)
    count = (nb + 1) * nb + 1
    cols, rows = grid
    if cols > 1 or rows > 1:
        n = num_queries(image_size)
        count += (n * cols + 1) * (n * rows)
    return count


@dataclass
class ImageViews:
    """Normalized views of one image, as float32 CPU tensors."""

    global_view: torch.Tensor  # (1, 3, S, S)
    crops: Optional[torch.Tensor]  # (P, 3, 640, 640) or None
    grid: Tuple[int, int]  # (cols, rows) of the crops; (1, 1) without crops
    num_tokens: int


def prepare_views(image: Image.Image, resolution: str) -> ImageViews:
    base_size, image_size, crop_mode = resolve_resolution(resolution)
    image = image.convert("RGB")
    color = _pad_color()
    if crop_mode:
        crops = None
        grid = (1, 1)
        if not (image.size[0] <= CROP_SIZE and image.size[1] <= CROP_SIZE):
            tiles, grid = dynamic_preprocess(image)
            if grid[0] > 1 or grid[1] > 1:
                crops = torch.stack([to_tensor(t) for t in tiles], dim=0)
        global_view = ImageOps.pad(image, (base_size, base_size), color=color)
    else:
        grid = (1, 1)
        crops = None
        if image_size <= CROP_SIZE:
            image = image.resize((image_size, image_size))
        global_view = ImageOps.pad(image, (image_size, image_size), color=color)
    return ImageViews(
        global_view=to_tensor(global_view).unsqueeze(0),
        crops=crops,
        grid=grid,
        num_tokens=image_token_count(base_size, image_size, crop_mode, grid),
    )


class DeepSeekOCRTokenizer:
    """Thin wrapper over the snapshot's ``tokenizer.json``."""

    def __init__(self, snapshot_dir: str | Path):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # ships with transformers
            raise ImportError(
                "DeepSeek-OCR needs the 'tokenizers' package (installed with the "
                "'vlm' extra): pip install 'libreyolo[vlm]'"
            ) from exc
        snapshot_dir = Path(snapshot_dir)
        self._tok = Tokenizer.from_file(str(snapshot_dir / "tokenizer.json"))
        cfg_path = snapshot_dir / "tokenizer_config.json"
        cfg = (
            json.loads(cfg_path.read_text(encoding="utf-8"))
            if cfg_path.exists()
            else {}
        )
        self.add_bos = bool(cfg.get("add_bos_token", True))
        self.bos_token_id = self._token_id(cfg.get("bos_token"), default=0)
        self.eos_token_id = self._token_id(cfg.get("eos_token"), default=1)
        image_id = self._tok.token_to_id(IMAGE_TOKEN)
        if image_id is None:
            raise ValueError(
                f"{IMAGE_TOKEN!r} is missing from {snapshot_dir}/tokenizer.json"
            )
        self.image_token_id = int(image_id)

    def _token_id(self, token, default: int) -> int:
        if isinstance(token, dict):
            token = token.get("content")
        if token is None:
            return default
        tid = self._tok.token_to_id(str(token))
        return default if tid is None else int(tid)

    def encode(self, text: str) -> List[int]:
        return list(self._tok.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)


def normalize_prompt(prompt: str) -> str:
    """Prompt text as the release's ``plain`` template renders it."""
    text = str(prompt).strip()
    if IMAGE_TOKEN not in text:
        text = f"{IMAGE_TOKEN}\n{text}"
    if text.count(IMAGE_TOKEN) != 1:
        raise ValueError(
            f"DeepSeek-OCR prompts take exactly one {IMAGE_TOKEN!r} placeholder."
        )
    return text


def build_input_ids(
    tokenizer: DeepSeekOCRTokenizer, prompt: str, num_image_tokens: int
) -> Tuple[List[int], List[bool]]:
    """Token ids and image-placeholder mask for one image and one prompt."""
    before, after = normalize_prompt(prompt).split(IMAGE_TOKEN)
    ids: List[int] = []
    mask: List[bool] = []
    if tokenizer.add_bos:
        ids.append(tokenizer.bos_token_id)
        mask.append(False)
    head = tokenizer.encode(before)
    ids += head
    mask += [False] * len(head)
    ids += [tokenizer.image_token_id] * num_image_tokens
    mask += [True] * num_image_tokens
    tail = tokenizer.encode(after)
    ids += tail
    mask += [False] * len(tail)
    return ids, mask
