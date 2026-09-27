"""LibreYOLO wrapper for NVIDIA's LocateAnything vision-language grounder.

LocateAnything is a generative grounding model: it emits semantic text plus
structured coordinate blocks such as ``<box><x1><y1><x2><y2></box>`` for boxes
and ``<box><x><y></box>`` for pointing. Coordinates are quantized on a 0-1000
image-relative scale, so this adapter parses the tag grammar into the same
``Results`` geometry used by the rest of LibreYOLO.

The model repository is not redistributed by LibreYOLO. Its weights and remote
model code are downloaded from Hugging Face at runtime after a license notice.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch

from ...utils.image_loader import ImageInput, ImageLoader
from .base import _INSTALL_HINT, LibreVLMModel
from .parsing import build_detection_dict, normalize_bbox, resolve_label

_COORD_BLOCK = re.compile(
    r"<\s*(?P<tag>box|point)\s*>\s*(?P<body>.*?)\s*<\s*/\s*(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_REF_BLOCK = re.compile(
    r"<\s*ref\s*>\s*(.*?)\s*<\s*/\s*ref\s*>", re.IGNORECASE | re.DOTALL
)
_TAG = re.compile(r"<[^>]+>")
_NUMBER = re.compile(r"(?<![A-Za-z])-?\d+(?:\.\d+)?(?![A-Za-z])")
_LABEL_SPLIT = re.compile(r"</c>|[\r\n;:]+")


def _numbers_from_coord_block(body: str) -> list[float]:
    values = []
    for match in _NUMBER.findall(body):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def _clean_label(fragment: str) -> Optional[str]:
    """Best-effort extraction of the semantic label before a coordinate block."""
    refs = _REF_BLOCK.findall(fragment)
    raw = refs[-1] if refs else fragment
    raw = _TAG.sub(" ", raw)
    parts = [part.strip(" \t:,.") for part in _LABEL_SPLIT.split(raw)]
    parts = [part for part in parts if part]
    if not parts:
        return None
    return re.sub(r"\s+", " ", parts[-1]) or None


def _parse_locateanything_coordinates(text: str) -> list[dict]:
    """Parse LocateAnything coordinate blocks into tagged raw coordinates."""
    if not isinstance(text, str) or not text.strip():
        return []
    items = []
    current_label = None
    previous_end = 0
    for match in _COORD_BLOCK.finditer(text):
        label = _clean_label(text[previous_end : match.start()])
        if label is not None:
            current_label = label
        numbers = _numbers_from_coord_block(match.group("body"))
        if len(numbers) == 4:
            items.append({"label": current_label, "bbox": numbers})
        elif len(numbers) == 2:
            items.append({"label": current_label, "point": numbers})
        previous_end = match.end()
    return items


def _parse_locateanything_boxes(text: str) -> list[dict]:
    return [item for item in _parse_locateanything_coordinates(text) if "bbox" in item]


def _parse_locateanything_points(text: str) -> list[dict]:
    return [item for item in _parse_locateanything_coordinates(text) if "point" in item]


def _resolve_label_or_singleton(label, name_to_id: Dict[str, int]) -> Optional[int]:
    class_id = resolve_label(label, name_to_id)
    if class_id is not None:
        return class_id
    if label is None and len(name_to_id) == 1:
        return next(iter(name_to_id.values()))
    return None


def _attach_singleton_labels(
    items: list[dict], name_to_id: Dict[str, int]
) -> list[dict]:
    if len(name_to_id) != 1:
        return items
    only_name = next(iter(name_to_id.keys()))
    return [{**item, "label": item.get("label") or only_name} for item in items]


def build_point_dict(
    items: List[dict],
    name_to_id: Dict[str, int],
    original_size: Tuple[int, int],
    conf_thres: float = 0.0,
    max_det: int = 300,
    classes: Optional[List[int]] = None,
    default_score: float = 1.0,
    coord_divisor: float = 1000.0,
) -> dict:
    """Turn parsed point items into the point-task detection dict."""
    if max_det <= 0:
        return {"points": [], "num_detections": 0}

    width, height = original_size
    points: List[List[float]] = []
    allowed_classes = set(classes) if classes is not None else None
    seen = set()

    for item in items:
        class_id = _resolve_label_or_singleton(item.get("label"), name_to_id)
        if class_id is None:
            continue
        if allowed_classes is not None and class_id not in allowed_classes:
            continue
        raw = item.get("point")
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            continue
        try:
            x = float(raw[0]) / coord_divisor
            y = float(raw[1]) / coord_divisor
        except (TypeError, ValueError):
            continue
        if any(v != v or v in (float("inf"), float("-inf")) for v in (x, y)):
            continue
        if default_score < conf_thres:
            continue
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        key = (class_id, round(x, 3), round(y, 3))
        if key in seen:
            continue
        seen.add(key)
        points.append([x * width, y * height, float(class_id), default_score])
        if len(points) >= max_det:
            break

    return {"points": points, "num_detections": len(points)}


class LibreLocateAnything(LibreVLMModel):
    """LocateAnything used as an open-vocabulary detector and pointer."""

    FAMILY = "locateanything"
    FILENAME_PREFIX = "LocateAnything"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "3b": "nvidia/LocateAnything-3B",
    }
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "3b": "c32291ca5e996f5a7a485845b4f57a233936bba0",
    }
    # Nominal only; the processor keeps source resolution and handles resizing.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "3b": 2500,
    }

    SUPPORTED_TASKS = ("detect", "point")
    DEFAULT_TASK = "detect"
    MAX_NEW_TOKENS = 8192
    COORD_DIVISOR = 1000.0
    TRUST_REMOTE_CODE = True

    TRAIN_UNSUPPORTED_REASON = (
        "LocateAnything's upstream weights are research-only non-commercial, "
        "so fine-tuned derivatives cannot be shipped for commercial use, and "
        "its training loop is a bespoke upstream pipeline. Use qwen3-vl for "
        "trainable VLM detection."
    )

    _LICENSE_NOTICE = (
        "\n"
        "----------------------------------------------------------------\n"
        "LocateAnything-3B is provided by NVIDIA under the NVIDIA\n"
        "License for non-commercial use. LibreYOLO does not redistribute\n"
        "the model repository, weights, or remote-code files. Loading this\n"
        "model downloads and executes Hugging Face remote code from NVIDIA's\n"
        "model repo; you are responsible for complying with those terms:\n"
        "  https://huggingface.co/nvidia/LocateAnything-3B\n"
        "----------------------------------------------------------------\n"
    )

    def __init__(
        self,
        size: str,
        *,
        generation_mode: str = "hybrid",
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        self.generation_mode = generation_mode
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.verbose = verbose
        self.tokenizer = None
        super().__init__(size=size, **kwargs)

    def _load_pretrained(self, snapshot_dir: str):
        try:
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        dtype = self._resolve_dtype()
        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot_dir, trust_remote_code=self.TRUST_REMOTE_CODE
        )
        processor = AutoProcessor.from_pretrained(
            snapshot_dir, trust_remote_code=self.TRUST_REMOTE_CODE
        )
        try:
            model = AutoModel.from_pretrained(
                snapshot_dir, dtype=dtype, trust_remote_code=self.TRUST_REMOTE_CODE
            )
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            model = AutoModel.from_pretrained(
                snapshot_dir,
                torch_dtype=dtype,
                trust_remote_code=self.TRUST_REMOTE_CODE,
            )
        return model, processor

    def _labels_for_prompt(self) -> str:
        return "</c>".join(self.names[i] for i in range(len(self.names)))

    def _detection_prompt(self) -> str:
        if self._custom_prompt:
            return self._custom_prompt
        labels = self._labels_for_prompt()
        if self.task == "point":
            return f"Point to: {labels}."
        return f"Locate all the instances that matches the following description: {labels}."

    def _processor_inputs(self, img, prompt: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": str(prompt)},
                ],
            }
        ]
        template_fn = getattr(
            self.processor,
            "py_apply_chat_template",
            getattr(self.processor, "apply_chat_template", None),
        )
        if template_fn is None:
            raise AttributeError(
                "LocateAnything processor has no chat-template method."
            )
        text = template_fn(messages, tokenize=False, add_generation_prompt=True)
        if hasattr(self.processor, "process_vision_info"):
            images, videos = self.processor.process_vision_info(messages)
        else:
            images, videos = [img], None
        kwargs = {
            "text": [text],
            "images": images,
            "return_tensors": "pt",
        }
        if videos is not None:
            kwargs["videos"] = videos
        return self.processor(**kwargs)

    def _generate(self, inputs: Any, max_new_tokens: Optional[int] = None):
        inputs = self._prepare_generation_inputs(inputs)
        generation_kwargs = dict(inputs)
        generation_kwargs.update(
            {
                "tokenizer": self.tokenizer,
                "max_new_tokens": max_new_tokens or self.MAX_NEW_TOKENS,
                "use_cache": True,
                "generation_mode": self.generation_mode,
                "do_sample": self.do_sample,
                "repetition_penalty": self.REPETITION_PENALTY,
                "verbose": self.verbose,
            }
        )
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p
        return self.model.generate(**generation_kwargs)

    def _decode_output(self, output: Any, inputs: Any | None = None) -> str:
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, str):
            return output
        if isinstance(output, list) and output and isinstance(output[0], str):
            return output[0]
        if isinstance(output, torch.Tensor):
            tokens = output
            if (
                inputs is not None
                and isinstance(inputs, MutableMapping)
                and "input_ids" in inputs
                and tokens.ndim == 2
                and tokens.shape[1] > inputs["input_ids"].shape[1]
            ):
                tokens = tokens[:, inputs["input_ids"].shape[1] :]
            decoder = (
                self.processor
                if hasattr(self.processor, "batch_decode")
                else self.tokenizer
            )
            return decoder.batch_decode(tokens, skip_special_tokens=False)[0]
        return str(output)

    def chat(
        self,
        image: ImageInput,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        color_format: str = "auto",
    ) -> str:
        img = ImageLoader.load(image, color_format=color_format)
        inputs = self._processor_inputs(img, prompt).to(self.device)
        with torch.no_grad():
            output = self._generate(inputs, max_new_tokens=max_new_tokens)
        return self._decode_output(output, inputs)

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        inputs = self._processor_inputs(img, self._detection_prompt())
        return inputs, img, img.size, 1.0

    def _forward(self, inputs: Any) -> Any:
        return self._generate(inputs)

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
        text = self._decode_output(output)
        if self.task == "point":
            points = _parse_locateanything_points(text)
            return build_point_dict(
                points,
                self._name_to_id,
                original_size,
                conf_thres=conf_thres,
                max_det=max_det,
                classes=kwargs.get("classes"),
                default_score=self._score_detections(points),
                coord_divisor=self.COORD_DIVISOR,
            )

        boxes = _attach_singleton_labels(
            _parse_locateanything_boxes(text), self._name_to_id
        )
        normalized = []
        for item in boxes:
            raw = item.get("bbox")
            if isinstance(raw, (list, tuple)) and len(raw) == 4:
                try:
                    scaled = [float(v) / self.COORD_DIVISOR for v in raw]
                except (TypeError, ValueError):
                    continue
                if normalize_bbox(scaled) is not None:
                    normalized.append(item)
        return build_detection_dict(
            normalized,
            self._name_to_id,
            original_size,
            conf_thres=conf_thres,
            max_det=max_det,
            classes=kwargs.get("classes"),
            default_score=self._score_detections(normalized),
            bbox_key=self.BBOX_KEY,
            coord_divisor=self.COORD_DIVISOR,
            box_format=self.BOX_FORMAT,
            iou_thres=iou_thres,
        )
