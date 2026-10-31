"""LibreYOLO wrapper for Tencent's WeVisOne-VL unified perception VLM.

WeVisOne-VL (Apache-2.0 code and weights) is a Qwen3-VL model fine-tuned to
answer perception requests as structured text: boxes and points are coordinate
tokens on a 0-1000 grid, absent targets come back as ``None``, and an ``<OBJ>``
token after a box asks a small (about 9.6M-parameter) pixel decoder for that
instance's alpha matte. The backbone loads through the native transformers
Qwen3-VL classes (no remote code); the pixel decoder is a minimal port shipped
in this package and loads the checkpoint's ``mask_decoder.pt`` sidecar.

    from libreyolo import LibreVLM
    model = LibreVLM("wevisone-vl", names=["person", "dog"])
    result = model.predict("image.jpg")                  # Results.boxes
    model.set_task("segment").predict("image.jpg")        # boxes + masks
    model.set_task("matte").predict("image.jpg")          # Results.matte
    LibreVLM("wevisone-vl", names=["the man in red"], referring=True)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...postprocess.ppocr import sort_quads_reading_order
from ...utils.image_loader import ImageInput, ImageLoader
from ..vlm.base import _INSTALL_HINT, LibreVLMModel
from ..vlm.locateanything import build_point_dict
from ..vlm.parsing import build_detection_dict, normalize_bbox, resolve_label
from . import parsing, prompts
from .feature_bridge import LastHiddenCapture, VisualFeatureCapture, premerge_grid_hw
from .mask_decoder import Qwen3VLSegConfig, build_mask_decoder

logger = logging.getLogger(__name__)

MASK_DECODER_FILE = "mask_decoder.pt"


class _WeVisOneRequest:
    """One predict call: the image plus the ``(class_id, prompt)`` queries.

    ``class_id`` is set when a query asks about a single vocabulary entry
    (point, referring, segment and matte issue one query per phrase) and None
    when one generation covers the whole vocabulary. The runner calls ``to``;
    inputs are encoded per query inside ``_forward``.
    """

    def __init__(self, image: Image.Image, queries: List[Tuple[Optional[int], str]]):
        self.image = image
        self.queries = queries

    def to(self, _device):
        return self


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class LibreWeVisOneVL(LibreVLMModel):
    """WeVisOne-VL as a multi-task LibreVLM family."""

    FAMILY = "wevisonevl"
    FILENAME_PREFIX = "LibreWeVisOneVL"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "2b": "tencent/WeVisOne-VL-2B",
        "4b": "tencent/WeVisOne-VL-4B",
    }
    # Weight-only repos (no remote code); pinned so the decoder sidecar and the
    # backbone it was trained against always come from the same snapshot.
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "2b": "798ffc7c87ad6104410812b589e240ba32d92315",
        "4b": "4a7bed92750ac9d8b8a124cede14759036932dc2",
    }
    # Nominal only: the Qwen processor smart-resizes within the pixel budget
    # below (2025 merged 32x32 tokens is about 1440x1440).
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "2b": 1440,
        "4b": 1440,
    }

    SUPPORTED_TASKS = ("detect", "point", "segment", "matte", "ocr")
    DEFAULT_TASK = "detect"
    # Upstream inference settings for the released checkpoints.
    MAX_NEW_TOKENS = 4096
    REPETITION_PENALTY = 1.05
    MIN_PIXELS: ClassVar[int] = 16 * 32 * 32
    MAX_PIXELS: ClassVar[int] = 2025 * 32 * 32
    COORD_DIVISOR = parsing.COORD_SCALE

    TRAIN_UNSUPPORTED_REASON = (
        "WeVisOne-VL is inference-only in LibreYOLO: the upstream release "
        "ships no training recipe for the backbone or the pixel decoder. Use "
        "qwen3-vl for trainable VLM detection."
    )

    def __init__(
        self,
        size: str = "2b",
        *,
        referring: bool = False,
        mask_threshold: float = 0.5,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        decoder_batch_size: int = 1,
        **kwargs,
    ):
        """
        Args:
            size: ``"2b"`` or ``"4b"``.
            referring: Treat every vocabulary entry as a referring expression
                for ``detect``: one generation per phrase with the referring
                instruction, which answers ``None`` (no boxes) for absent
                targets. ``point``, ``segment`` and ``matte`` always query
                one phrase at a time.
            mask_threshold: Alpha threshold that turns a predicted matte into
                the binary ``segment`` mask.
            min_pixels / max_pixels: Pixel budget of the Qwen image processor
                (defaults to the upstream ``16*32*32`` / ``2025*32*32``). A
                lower ``max_pixels`` trades accuracy for speed.
            decoder_batch_size: Instances refined per pixel-decoder call. All
                instances of a response are always encoded together, so this
                bounds memory only and does not change the masks. Refinement
                runs on a full 1024x1024 canvas at about 2 GB of FP32
                activations per instance.
        """
        self.referring = bool(referring)
        self.mask_threshold = float(mask_threshold)
        self.min_pixels = int(min_pixels) if min_pixels is not None else self.MIN_PIXELS
        self.max_pixels = int(max_pixels) if max_pixels is not None else self.MAX_PIXELS
        self.decoder_batch_size = max(1, int(decoder_batch_size))
        self._user_vocab = False
        # Filled by _load_pretrained.
        self.mask_decoder: Optional[nn.Module] = None
        self.decoder_cfg = Qwen3VLSegConfig()
        self.seg_token_id: Optional[int] = None
        self.spatial_merge_size = 2
        super().__init__(size=size, **kwargs)

    # =========================================================================
    # Loading
    # =========================================================================

    def _load_pretrained(self, snapshot_dir: str):
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        model = AutoModelForImageTextToText.from_pretrained(
            snapshot_dir, dtype=self._resolve_dtype()
        )
        processor = AutoProcessor.from_pretrained(
            snapshot_dir,
            size={"shortest_edge": self.min_pixels, "longest_edge": self.max_pixels},
        )
        self._load_mask_decoder(Path(snapshot_dir), model, processor)
        return model, processor

    def _load_mask_decoder(
        self, snapshot_dir: Path, model: nn.Module, processor
    ) -> None:
        sidecar_path = snapshot_dir / MASK_DECODER_FILE
        if not sidecar_path.is_file():
            raise FileNotFoundError(
                f"WeVisOne-VL pixel decoder weights are missing: {sidecar_path}. "
                "Delete the snapshot directory to re-download it."
            )
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=True)
        if not isinstance(sidecar, dict) or not isinstance(
            sidecar.get("state_dict"), dict
        ):
            raise ValueError(
                f"Invalid WeVisOne-VL mask decoder sidecar: {sidecar_path}"
            )
        cfg = Qwen3VLSegConfig.from_dict(sidecar.get("cfg"))
        tokenizer = processor.tokenizer
        if parsing.OBJ_TOKEN not in tokenizer.get_vocab():
            raise RuntimeError("The WeVisOne-VL tokenizer has no <OBJ> token.")
        seg_token_id = int(tokenizer.convert_tokens_to_ids(parsing.OBJ_TOKEN))
        saved_id = sidecar.get("seg_token_id")
        if saved_id is not None and int(saved_id) != seg_token_id:
            raise RuntimeError(
                f"<OBJ> token mismatch: tokenizer={seg_token_id}, decoder={saved_id}"
            )
        decoder = build_mask_decoder(
            visual_dim=model.config.vision_config.hidden_size,
            llm_dim=model.config.text_config.hidden_size,
            cfg=cfg,
        )
        decoder.load_state_dict(sidecar["state_dict"], strict=True)
        self.mask_decoder = decoder.float().eval()
        self.decoder_cfg = cfg
        self.seg_token_id = seg_token_id
        self.spatial_merge_size = int(
            sidecar.get("spatial_merge_size")
            or model.config.vision_config.spatial_merge_size
        )

    # =========================================================================
    # Vocabulary and prompts
    # =========================================================================

    def set_classes(self, classes: list) -> "LibreWeVisOneVL":
        result = super().set_classes(classes)
        self._user_vocab = True
        return result

    def _require_user_vocab(self, what: str) -> None:
        if not self._user_vocab:
            raise ValueError(
                f"{what} queries one phrase at a time; call "
                'set_classes([...]) first, e.g. set_classes(["person"]) or '
                'set_classes(["the dog on the left"]).'
            )

    def _labels(self) -> List[str]:
        return [self.names[i] for i in range(len(self.names))]

    def _queries(self) -> List[Tuple[Optional[int], str]]:
        if self._custom_prompt:
            return [(None, self._custom_prompt)]
        if self.task == "ocr":
            return [(None, prompts.OCR_PROMPT)]
        if self.task == "point":
            # Multi-category point prompts degrade into malformed tags on the
            # 2B checkpoint; one category per query stays well-formed.
            return [
                (i, prompts.point_prompt([label]))
                for i, label in enumerate(self._labels())
            ]
        if self.task in ("segment", "matte"):
            self._require_user_vocab(f"The {self.task} task")
            return [
                (i, prompts.matting_prompt(label))
                for i, label in enumerate(self._labels())
            ]
        if self.referring:
            self._require_user_vocab("Referring detection")
            return [
                (i, prompts.referring_prompt(label))
                for i, label in enumerate(self._labels())
            ]
        return [(None, prompts.detection_prompt(self._labels()))]

    # =========================================================================
    # Generation
    # =========================================================================

    def _encode(self, img: Image.Image, prompt: str, response: Optional[str] = None):
        messages = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": str(prompt)},
                ],
            },
        ]
        if response is not None:
            messages.append({"role": "assistant", "content": response})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=response is None
        )
        inputs = self.processor(
            text=[text], images=[img], padding=True, return_tensors="pt"
        ).to(self.device)
        return self._prepare_generation_inputs(inputs)

    def _generate(
        self, img: Image.Image, prompt: str, max_new_tokens: Optional[int] = None
    ) -> str:
        inputs = self._encode(img, prompt)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=self.REPETITION_PENALTY,
                use_cache=True,
            )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        # Special tokens are kept: <OBJ> must survive for the pixel decoder.
        text = self.processor.batch_decode(
            new_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        return parsing.strip_response(text)

    def chat(
        self,
        image: ImageInput,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        color_format: str = "auto",
    ) -> str:
        """Raw generation with the checkpoint's system prompt; returns the text
        verbatim, including coordinate tags such as ``<box><12><40>...</box>``."""
        img = ImageLoader.load(image, color_format=color_format)
        return self._generate(img, str(prompt), max_new_tokens=max_new_tokens)

    # =========================================================================
    # Pixel decoder
    # =========================================================================

    def _decode_instances(
        self, img: Image.Image, prompt: str, response: str
    ) -> List[dict]:
        """Teacher-force ``response`` and decode one alpha matte per ``<OBJ>``.

        Returns ``{"label", "box", "alpha"}`` dicts: the ``<ref>`` label, the
        normalized ``xyxy`` box and an ``(H, W)`` float32 matte in ``[0, 1]``
        on the original image canvas.
        """
        entries = parsing.parse_obj_boxes(response)
        if not any(entry is not None for entry in entries):
            return []
        teacher = self._encode(img, prompt, response=response)
        seg_positions = (
            (teacher["input_ids"][0] == self.seg_token_id).nonzero().flatten()
        )
        if int(seg_positions.numel()) < len(entries):
            raise RuntimeError(
                f"The teacher-forced response has {len(entries)} <OBJ> tags but "
                f"tokenized to {int(seg_positions.numel())} <OBJ> tokens."
            )
        # The response is the last turn and the chat template closes it with
        # <|im_end|> only, so its <OBJ> tokens are the trailing ones; any
        # earlier ones come from the prompt.
        obj_positions = seg_positions[int(seg_positions.numel()) - len(entries) :]
        with (
            torch.no_grad(),
            VisualFeatureCapture(
                self.model, self.decoder_cfg.vision_feature_layers
            ) as visual,
            LastHiddenCapture(self.model) as hidden,
        ):
            self.model(**teacher, use_cache=False, logits_to_keep=1)
        if hidden.hidden is None:
            raise RuntimeError("Qwen3-VL returned no hidden state for mask decoding.")
        t_seg = hidden.hidden[0, obj_positions]
        grid_hw = premerge_grid_hw(teacher["image_grid_thw"])
        visual_maps = visual.feature_maps(grid_hw, self.spatial_merge_size)

        selected = [
            (ordinal, entry)
            for ordinal, entry in enumerate(entries)
            if entry is not None
        ]
        decoder = self.mask_decoder.to(self.device)
        size = int(self.decoder_cfg.cnn_image_size)
        canvas = np.asarray(
            img.convert("RGB").resize((size, size), resample=Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
        canvas = torch.from_numpy(canvas / 255.0).permute(2, 0, 1)[None].to(self.device)
        ordinals = torch.tensor(
            [ordinal for ordinal, _ in selected], device=t_seg.device
        )
        boxes = torch.tensor(
            [entry[1] for _, entry in selected], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            queries, pixel_feature = decoder.encode(
                canvas, visual_maps, t_seg[ordinals].to(self.device), boxes
            )
        width, height = img.size
        instances: List[dict] = []
        for start in range(0, len(selected), self.decoder_batch_size):
            batch = selected[start : start + self.decoder_batch_size]
            with torch.no_grad():
                logits = decoder.refine(
                    canvas, pixel_feature, queries[start : start + len(batch)]
                )
                alphas = F.interpolate(
                    logits.float().sigmoid()[None],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            for (_, (label, box)), alpha in zip(batch, alphas):
                instances.append(
                    {
                        "label": label,
                        "box": box,
                        "alpha": alpha.cpu().numpy().astype(np.float32),
                    }
                )
        return instances

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
        return _WeVisOneRequest(img, self._queries()), img, img.size, 1.0

    def _forward(self, request: _WeVisOneRequest) -> List[dict]:
        decode = self.task in ("segment", "matte")
        records = []
        for class_id, prompt in request.queries:
            text = self._generate(request.image, prompt)
            instances = (
                self._decode_instances(request.image, prompt, text) if decode else []
            )
            records.append({"class_id": class_id, "text": text, "instances": instances})
        return records

    def _class_id(self, record: dict, label: Optional[str]) -> Optional[int]:
        """Resolve a ``<ref>`` label against the vocabulary.

        A per-phrase query owns its answers whatever the model echoes back; a
        bare tag from a single-class vocabulary belongs to that class.
        """
        if record["class_id"] is not None:
            return record["class_id"]
        class_id = resolve_label(label, self._name_to_id)
        if class_id is None and len(self.names) == 1:
            return 0
        return class_id

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
        classes = kwargs.get("classes")
        if self.task == "ocr":
            return self._postprocess_ocr(output, original_size, max_det)
        if self.task == "segment":
            return self._postprocess_segment(
                output, conf_thres, iou_thres, original_size, max_det, classes
            )
        if self.task == "matte":
            return self._postprocess_matte(output, original_size, classes)
        kind = "point" if self.task == "point" else "box"
        items = []
        for record in output:
            for shape in parsing.parse_shapes(record["text"]):
                if shape["kind"] != kind:
                    continue
                class_id = self._class_id(record, shape["label"])
                if class_id is None:
                    continue
                key = "point" if kind == "point" else "bbox"
                items.append({"label": self.names[class_id], key: shape["coords"]})
        if kind == "point":
            return build_point_dict(
                items,
                self._name_to_id,
                original_size,
                conf_thres=conf_thres,
                max_det=max_det,
                classes=classes,
                default_score=self._score_detections(items),
                coord_divisor=self.COORD_DIVISOR,
            )
        return build_detection_dict(
            items,
            self._name_to_id,
            original_size,
            conf_thres=conf_thres,
            max_det=max_det,
            classes=classes,
            default_score=self._score_detections(items),
            bbox_key="bbox",
            coord_divisor=self.COORD_DIVISOR,
            box_format="xyxy",
            iou_thres=iou_thres,
        )

    def _postprocess_segment(
        self, output, conf_thres, iou_thres, original_size, max_det, classes
    ) -> Dict:
        width, height = original_size
        allowed = set(classes) if classes is not None else None
        score = self._score_detections(output)
        boxes, norm_boxes, class_ids, masks = [], [], [], []
        if score >= conf_thres and max_det > 0:
            for record in output:
                for instance in record["instances"]:
                    class_id = self._class_id(record, instance["label"])
                    if class_id is None or (
                        allowed is not None and class_id not in allowed
                    ):
                        continue
                    box = normalize_bbox(list(instance["box"]))
                    if box is None or any(
                        kept_class == class_id and _box_iou(box, kept) > iou_thres
                        for kept, kept_class in zip(norm_boxes, class_ids)
                    ):
                        continue
                    norm_boxes.append(box)
                    boxes.append(
                        [
                            box[0] * width,
                            box[1] * height,
                            box[2] * width,
                            box[3] * height,
                        ]
                    )
                    class_ids.append(class_id)
                    masks.append(instance["alpha"] >= self.mask_threshold)
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

    def _postprocess_matte(self, output, original_size, classes) -> Dict:
        """Union (pixelwise max) of the instance mattes of every phrase."""
        width, height = original_size
        allowed = set(classes) if classes is not None else None
        matte = np.zeros((height, width), dtype=np.float32)
        for record in output:
            for instance in record["instances"]:
                class_id = self._class_id(record, instance["label"])
                if class_id is None or (
                    allowed is not None and class_id not in allowed
                ):
                    continue
                np.maximum(matte, instance["alpha"], out=matte)
        return {"matte": np.clip(matte, 0.0, 1.0)}

    def _postprocess_ocr(self, output, original_size, max_det) -> Dict:
        width, height = original_size
        polygons, texts = [], []
        for record in output:
            for shape in parsing.parse_shapes(record["text"]):
                if len(polygons) >= max_det:
                    break
                if shape["kind"] == "box":
                    x1, y1, x2, y2 = (v / self.COORD_DIVISOR for v in shape["coords"])
                    quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                elif shape["kind"] == "polygon" and len(shape["coords"]) == 8:
                    c = [v / self.COORD_DIVISOR for v in shape["coords"]]
                    quad = [c[k : k + 2] for k in range(0, 8, 2)]
                else:
                    continue
                polygons.append([[x * width, y * height] for x, y in quad])
                texts.append(shape["label"] or "")
        order = sort_quads_reading_order(
            np.asarray(polygons, dtype=np.float32).reshape(-1, 4, 2)
        )
        polygons = [polygons[i] for i in order]
        texts = [texts[i] for i in order]
        score = self._score_detections(texts)
        return {
            "ocr": {
                "polygons": polygons,
                "texts": texts,
                "confidences": [score] * len(texts),
                "det_confidences": [score] * len(texts),
            }
        }
