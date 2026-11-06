"""LibreYOLO wrapper for TII's Falcon Perception open-vocabulary grounder.

Falcon Perception (Apache-2.0 code and weights) is an early-fusion transformer
that reads an image and a natural-language query and emits, per matching
instance, a structured token sequence ``<|coord|> <|size|> <|seg|>``: a box
centre, a box extent, and a mask query. Boxes are decoded by small heads on the
hidden state; masks come from dotting the ``<|seg|>`` embedding with AnyUp
upsampled image features. The network is vendored in this package (no remote
code, no ``trust_remote_code``).

    from libreyolo import LibreVLM
    model = LibreVLM("falcon-perception", names=["cat", "the dog on the left"])
    result = model.predict("image.jpg")                 # Results.boxes
    model.set_task("segment").predict("image.jpg")      # boxes + Results.masks

Each vocabulary entry is one query and one generation; every instance found
gets that entry's class id. ``300m`` is detection only (no mask head).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..vlm.base import _INSTALL_HINT, LibreVLMModel
from .generate import Generation, generate, pair_bbox_entries
from .nn import FalconPerception, get_model_args
from .preprocess import (
    MAX_DIMENSION,
    MIN_DIMENSION,
    PATCH_SIZE,
    FalconTokenizer,
    anyup_canvas_size,
    batch_images_with_mask,
    build_prompt,
    get_pos_thw,
    preprocess_image,
    resize_image_if_necessary,
    tokenize_prompt,
)

logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 2048


class _FalconRequest:
    """One predict call: the image plus one ``(class_id, query)`` per vocabulary entry."""

    def __init__(self, image: Image.Image, queries: List[Tuple[int, str]]):
        self.image = image
        self.queries = queries

    def to(self, _device):
        return self


class LibreFalconPerception(LibreVLMModel):
    """Falcon Perception as a LibreVLM family (detect, and segment on ``600m``)."""

    FAMILY = "falconperception"
    FILENAME_PREFIX = "LibreFalconPerception"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "300m": "tiiuae/Falcon-Perception-300M",
        "600m": "tiiuae/Falcon-Perception",
    }
    # Pinned snapshots (weights, config, tokenizer). The repos also hold remote
    # modeling code, which is never executed: the network is vendored here.
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "300m": "36993684b3ba5945b22d01e705f8ff2e048bf0b0",
        "600m": "a3e21ada0abd533393776ffab8908a728f775509",
    }
    # Nominal only: images are resized to 256..1024 px per side on the fly.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"300m": 1024, "600m": 1024}

    SUPPORTED_TASKS = ("detect", "segment")
    DEFAULT_TASK = "detect"
    SEGMENT_SIZES: ClassVar[Tuple[str, ...]] = ("600m",)

    SNAPSHOT_IGNORE_PATTERNS = LibreVLMModel.SNAPSHOT_IGNORE_PATTERNS + (
        "*.py",
        "*.jpg",
        "*.png",
        ".eval_results/*",
    )

    TRAIN_UNSUPPORTED_REASON = (
        "Falcon Perception is inference-only in LibreYOLO: the upstream release "
        "ships no training recipe. Use qwen3-vl for trainable VLM detection."
    )

    def __init__(
        self,
        size: str = "600m",
        *,
        coord_dedup_threshold: float = 0.01,
        max_new_tokens: Optional[int] = None,
        device: str = "auto",
        **kwargs,
    ):
        """
        Args:
            size: ``"600m"`` (detect and segment) or ``"300m"`` (detect only).
            coord_dedup_threshold: Normalised distance below which a repeated
                box centre is re-decoded from the next-best bin (upstream
                default 0.01). ``0`` disables it.
            max_new_tokens: Generation budget per query (default 2048).
        """
        self.coord_dedup_threshold = float(coord_dedup_threshold)
        self._user_vocab = False
        self.tokenizer: Optional[FalconTokenizer] = None
        # RoPE tables are complex64, which the MPS backend does not implement.
        if device in ("auto", "mps") and not torch.cuda.is_available():
            if device == "mps" or torch.backends.mps.is_available():
                if device == "mps":
                    warnings.warn(
                        "Falcon Perception uses complex RoPE tables that MPS does "
                        "not support; running on CPU.",
                        stacklevel=2,
                    )
                device = "cpu"
        super().__init__(
            size=size,
            device=device,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else MAX_NEW_TOKENS,
            **kwargs,
        )

    # =========================================================================
    # Loading
    # =========================================================================

    def _init_model(self) -> nn.Module:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        snapshot_dir = Path(self._ensure_weights())
        self.tokenizer = FalconTokenizer(str(snapshot_dir))
        self.processor = self.tokenizer
        args = get_model_args(self.size)
        args.update(self.tokenizer)

        net = FalconPerception(args).eval()
        state = load_file(str(snapshot_dir / "model.safetensors"))
        net.load_state_dict(state, strict=True)
        dtype = self._resolve_dtype()
        self._model_dtype = dtype
        return net.to(dtype=dtype)

    @property
    def supports_segmentation(self) -> bool:
        return bool(self.model.args.do_segmentation)

    def set_task(self, task: str) -> "LibreFalconPerception":
        resolved = self._resolve_task(task)
        if resolved == "segment" and not self.supports_segmentation:
            raise ValueError(
                f"Falcon Perception {self.size!r} has no mask head; "
                f"segment needs size in {self.SEGMENT_SIZES}."
            )
        self.task = resolved
        return self

    # =========================================================================
    # Vocabulary
    # =========================================================================

    def set_classes(self, classes: list) -> "LibreFalconPerception":
        result = super().set_classes(classes)
        self._user_vocab = True
        return result

    def _queries(self) -> List[Tuple[int, str]]:
        if not self._user_vocab:
            raise ValueError(
                "Falcon Perception answers one text query at a time; call "
                'set_classes([...]) first, e.g. set_classes(["cat"]) or '
                'set_classes(["the dog on the left"]), or pass names=[...].'
            )
        if self._custom_prompt:
            return [(0, self._custom_prompt)]
        return [(i, self.names[i]) for i in range(len(self.names))]

    def chat(self, *args, **kwargs):
        raise NotImplementedError(
            "Falcon Perception emits coordinate and mask tokens, not free text; "
            "use predict() with set_classes()."
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
        if self.task == "segment" and not self.supports_segmentation:
            raise ValueError(
                f"Falcon Perception {self.size!r} has no mask head; use size "
                f"{self.SEGMENT_SIZES} for segment."
            )
        img = ImageLoader.load(image, color_format=color_format)
        return _FalconRequest(img, self._queries()), img, img.size, 1.0

    def encode_query(self, image: Image.Image, query: str) -> Dict[str, torch.Tensor]:
        """Tokenise one (image, query) pair into the model's input tensors."""
        tok = self.tokenizer
        resized = resize_image_if_necessary(image, MIN_DIMENSION, MAX_DIMENSION)
        pixels = preprocess_image(resized, PATCH_SIZE)  # (1, H, W, 3)
        # The prompt may use whatever context the generation budget leaves; a full
        # 1024x1024 image alone is 4096 patch tokens.
        max_prompt = self.model.args.max_seq_len - self.MAX_NEW_TOKENS
        ids = tokenize_prompt(build_prompt(query), pixels[0], tok, PATCH_SIZE, max_prompt)
        tokens_np = ids[None, :]

        canvas, _ = anyup_canvas_size(
            pixels.shape[1], pixels.shape[2], patch_size=PATCH_SIZE, max_size=MAX_DIMENSION
        )
        processed = batch_images_with_mask([pixels], canvas, canvas)
        pos_t_np, pos_hw_np = get_pos_thw(
            tokens_np, processed["padding_mask"], tok, PATCH_SIZE, pad_token_id=tok.pad_token_id
        )
        dev = self.device
        return {
            "tokens": torch.from_numpy(tokens_np).to(dev),
            "pixel_values": torch.from_numpy(processed["pixel_values"]).to(dev),
            "pixel_mask": torch.from_numpy(processed["padding_mask"]).to(dev),
            "pos_t": torch.from_numpy(pos_t_np).to(dev),
            "pos_hw": torch.from_numpy(pos_hw_np).to(dev),
        }

    def run_query(self, image: Image.Image, query: str, *, segmentation: bool) -> Generation:
        inputs = self.encode_query(image, query)
        tok = self.tokenizer
        return generate(
            self.model,
            tok,
            inputs["tokens"],
            inputs["pos_t"],
            inputs["pos_hw"],
            inputs["pixel_values"],
            inputs["pixel_mask"],
            segmentation=segmentation,
            max_new_tokens=self.MAX_NEW_TOKENS,
            stop_token_ids=[tok.eos_token_id, tok.end_of_query_token_id],
            coord_dedup_threshold=self.coord_dedup_threshold,
        )

    def _forward(self, request: _FalconRequest) -> List[dict]:
        segmentation = self.task == "segment"
        records = []
        for class_id, query in request.queries:
            gen = self.run_query(request.image, query, segmentation=segmentation)
            records.append({"class_id": class_id, "generation": gen})
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
        width, height = original_size
        allowed = set(kwargs["classes"]) if kwargs.get("classes") is not None else None
        score = float(self._score_detections(output))
        boxes: List[List[float]] = []
        class_ids: List[int] = []
        masks: List[np.ndarray] = []
        want_masks = self.task == "segment"

        if score >= conf_thres and max_det > 0:
            for record in output:
                class_id = record["class_id"]
                if allowed is not None and class_id not in allowed:
                    continue
                gen: Generation = record["generation"]
                for idx, det in enumerate(pair_bbox_entries(gen.boxes_raw)):
                    cx, cy = det["x"] * width, det["y"] * height
                    bw, bh = det["w"] * width, det["h"] * height
                    boxes.append(
                        [
                            max(0.0, cx - bw / 2.0),
                            max(0.0, cy - bh / 2.0),
                            min(float(width), cx + bw / 2.0),
                            min(float(height), cy + bh / 2.0),
                        ]
                    )
                    class_ids.append(class_id)
                    if want_masks:
                        masks.append(_mask_to_original(gen.masks, idx, width, height))
                    if len(boxes) >= max_det:
                        break
                if len(boxes) >= max_det:
                    break

        detections = {
            "boxes": boxes,
            "scores": [score] * len(boxes),
            "classes": class_ids,
            "num_detections": len(boxes),
        }
        if masks:
            detections["masks"] = np.stack(masks).astype(np.uint8)
        return detections


def _mask_to_original(
    masks: Optional[torch.Tensor], idx: int, width: int, height: int
) -> np.ndarray:
    """Nearest-neighbour resize of one feature-resolution mask to the original image.

    Matches the upstream renderer; instances without a ``<|seg|>`` token get an
    empty mask so box and mask counts stay aligned.
    """
    if masks is None or idx >= masks.shape[0]:
        return np.zeros((height, width), dtype=np.uint8)
    m = masks[idx].to(torch.uint8).cpu().numpy()
    if m.shape != (height, width):
        m = np.array(Image.fromarray(m).resize((width, height), Image.NEAREST))
    return m
