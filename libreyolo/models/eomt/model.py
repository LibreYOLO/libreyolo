"""LibreEoMT semantic and instance segmentation wrapper."""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.ops import batched_nms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TVF

from ...tasks import normalize_task
from ...utils.amp import normalize_amp_dtype, torch_amp_dtype
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.serialization import load_untrusted_torch_file
from ..base.model import BaseModel
from .nn import LibreEoMTNet, normalize_eomt_state_dict

logger = logging.getLogger(__name__)


def _extract_state(loaded: dict[str, Any]) -> dict[str, Any]:
    for key in ("model", "state_dict"):
        if key in loaded and isinstance(loaded[key], dict):
            return loaded[key]
    return loaded


def _eomt_keys(weights_dict: dict[str, Any]) -> set[str]:
    return set(normalize_eomt_state_dict(weights_dict))


class LibreEoMT(BaseModel):
    """Encoder-only Mask Transformer for semantic and instance segmentation."""

    FAMILY: ClassVar[str] = "eomt"
    FILENAME_PREFIX: ClassVar[str] = "LibreEoMT"
    # Capturable once the attention-mask schedule stays on the host; see
    # LibreEoMTNet._apply.
    SUPPORTS_CUDA_GRAPH = True
    WEIGHT_EXT: ClassVar[str] = ".pt"

    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("semantic", "segment", "panoptic")
    DEFAULT_TASK: ClassVar[str] = "semantic"
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = True
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"s": 512, "b": 512, "l": 512}
    TASK_INPUT_SIZES: ClassVar[Dict[str, Dict[str, int]]] = {
        "semantic": {"s": 512, "b": 512, "l": 512},
        "segment": {"s": 640, "b": 640, "l": 640},
        "panoptic": {"s": 640, "b": 640, "l": 640},
    }

    # Panoptic merge constants (Mask2Former/MaskFormer inference recipe, and the
    # upstream EoMT post-process defaults). A query's binarized mask must survive
    # the per-pixel argmax with at least PANOPTIC_OVERLAP_THRESHOLD of its own
    # area, else it is dropped as fully occluded.
    PANOPTIC_SCORE_THRESHOLD: ClassVar[float] = 0.8
    PANOPTIC_MASK_THRESHOLD: ClassVar[float] = 0.5
    PANOPTIC_OVERLAP_THRESHOLD: ClassVar[float] = 0.8

    # Flip-TTA only: mask-IoU threshold above which same-class queries from
    # the two views are treated as the same real instance and merged before
    # fusion (see _dedup_panoptic_queries). Matches PQ_IOU_THRESHOLD's
    # convention for "same instance" matching.
    PANOPTIC_TTA_DEDUP_IOU: ClassVar[float] = 0.5

    WEIGHT_VARIANTS: ClassVar[Tuple[str, ...]] = ("1280",)
    # Sizes s/b are published for panoptic only.
    UNPUBLISHED_WEIGHTS: ClassVar[Dict[str, str]] = {
        f"LibreEoMT{size}-{suffix}": (
            f"EoMT {suffix} weights are published for size l only; use "
            f"LibreEoMTl-{suffix}.pt or LibreEoMT{size}-panoptic.pt."
        )
        for size in ("s", "b")
        for suffix in ("seg", "sem")
    }

    semantic_resize_mode: ClassVar[str] = "split"
    semantic_imgsz_divisor: ClassVar[int] = 16

    _EMBED_DIM_TO_SIZE: ClassVar[Dict[int, str]] = {384: "s", 768: "b", 1024: "l"}
    _UPSTREAM_URL: ClassVar[str] = "https://github.com/tue-mps/eomt"
    SUPPORTS_BATCHED_PREDICT: ClassVar[bool] = False

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys = _eomt_keys(weights_dict)
        return {
            "query.weight",
            "mask_head.fc1.weight",
            "mask_head.fc2.weight",
            "mask_head.fc3.weight",
            "class_predictor.weight",
            "embeddings.patch_embeddings.projection.weight",
        }.issubset(keys)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        state = normalize_eomt_state_dict(weights_dict)
        for key in ("query.weight", "class_predictor.weight"):
            tensor = state.get(key)
            if tensor is not None and getattr(tensor, "ndim", 0) >= 2:
                return cls._EMBED_DIM_TO_SIZE.get(int(tensor.shape[-1]))
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        state = normalize_eomt_state_dict(weights_dict)
        weight = state.get("class_predictor.weight")
        if weight is not None and getattr(weight, "ndim", 0) >= 1:
            return max(1, int(weight.shape[0]) - 1)
        return None

    @classmethod
    def detect_num_queries(cls, weights_dict: dict) -> Optional[int]:
        state = normalize_eomt_state_dict(weights_dict)
        weight = state.get("query.weight")
        if weight is not None and getattr(weight, "ndim", 0) >= 1:
            return int(weight.shape[0])
        return None

    @classmethod
    def detect_image_size(
        cls, weights_dict: dict, patch_size: int = 16
    ) -> Optional[int]:
        """Infer image_size from position embedding shape: sqrt(num_positions) * patch_size."""
        state = normalize_eomt_state_dict(weights_dict)
        weight = state.get("embeddings.position_embeddings.weight")
        if weight is not None and getattr(weight, "ndim", 0) >= 1:
            num_positions = int(weight.shape[0])
            side = int(round(num_positions**0.5))
            if side * side == num_positions:
                return side * patch_size
        return None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        if not cls.can_load(state_dict):
            return None
        nc = cls.detect_nb_classes(state_dict)
        # nc==80 is the COCO instance checkpoint convention; all other class
        # counts (ADE20K 150, panoptic 133, custom) default to semantic.
        if nc is not None and nc == 80:
            return "segment"
        return "semantic"

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> Optional[dict]:
        # Raw HF EoMT checkpoints must go through weights/convert_eomt_weights.py
        # so DINOv2-only provenance and ADE20K metadata are enforced.
        return None

    def __init__(
        self,
        model_path=None,
        size: str = "l",
        nb_classes: int = 150,
        device: str = "auto",
        task: str | None = None,
        # New parameters go after the complete v1.3 signature so positional
        # calls like LibreEoMT(None, "l", 150, "cpu") keep working.
        num_queries: int = 100,
        **kwargs,
    ) -> None:
        if size is None:
            size = "l"
        self.num_queries = int(num_queries)

        if isinstance(model_path, dict) and not model_path:
            weight_source = None
        elif isinstance(model_path, str):
            weight_source = self._resolve_weights_path(model_path)
        else:
            weight_source = model_path

        # When no explicit task is given, try to infer from the filename so
        # that LibreEoMT("LibreEoMTb-seg.pt") and panoptic checkpoints work
        # without requiring the caller to spell out task=.
        if task is None and isinstance(weight_source, (str, Path)):
            filename = Path(weight_source).name
            inferred = self.detect_task_from_filename(filename)
            if inferred is None and "-panoptic" in filename.lower():
                inferred = "segment"
            if inferred is not None:
                task = inferred

        # BaseModel._resolve_task validates task against SUPPORTED_TASKS.
        super().__init__(
            model_path=None,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=task,
            **kwargs,
        )

        if weight_source is not None:
            self._load_weights(weight_source)
            # BaseModel.__init__ received model_path=None (EoMT loads its own
            # weights above), so it left self.model_path unset. Restore the
            # resolved path so direct ``LibreEoMT("...")`` construction matches
            # the factory path, which sets model_path post-construction.
            if isinstance(weight_source, (str, Path)):
                self.model_path = str(weight_source)
        self.model.eval()

    def _init_model(self) -> nn.Module:
        return LibreEoMTNet(
            config=self.size,
            nb_classes=self.nb_classes,
            image_size=self.input_size,
            num_queries=getattr(self, "num_queries", 100),
        )

    def _strict_loading(self) -> bool:
        return True

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        self.nb_classes = int(new_nb_classes)
        self.names = {i: f"class_{i}" for i in range(self.nb_classes)}
        self.model = self._init_model()
        self.model.to(self.device)

    def _rebuild_for_new_queries(self, new_num_queries: int):
        self.num_queries = int(new_num_queries)
        self.model = self._init_model()
        self.model.to(self.device)

    def _rebuild_for_new_image_size(self, new_image_size: int):
        self.input_size = int(new_image_size)
        self.model = self._init_model()
        self.model.to(self.device)

    def _rebuild_for_new_size(self, new_size: str):
        self.size = new_size
        self.input_size = self._get_task_input_sizes()[new_size]
        self.model = self._init_model()
        self.model.to(self.device)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        core = getattr(self.model, "eomt", self.model)
        layers: Dict[str, nn.Module] = {}
        for name in ("embeddings", "layers", "mask_head", "class_predictor"):
            module = getattr(core, name, None)
            if module is not None:
                layers[name] = module
        return layers

    @staticmethod
    def _get_preprocess_numpy():
        import cv2
        import numpy as _np

        def _preprocess_numpy(img_rgb_hwc, input_size=512):
            h = input_size if isinstance(input_size, int) else input_size[0]
            w = input_size if isinstance(input_size, int) else input_size[1]
            resized = cv2.resize(img_rgb_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
            arr = _np.ascontiguousarray(resized, dtype=_np.float32) / 255.0
            return arr.transpose(2, 0, 1), 1.0

        return _preprocess_numpy

    @staticmethod
    def _shortest_edge_size(height: int, width: int, size: int) -> tuple[int, int]:
        """Match HF EoMT shortest-edge resize integer semantics."""
        if (height <= width and height == size) or (width <= height and width == size):
            return height, width
        if width < height:
            return int(size * height / width), size
        return size, int(size * width / height)

    @staticmethod
    def _split_resized_tensor(
        tensor: torch.Tensor,
        patch_size: int,
    ) -> tuple[list[torch.Tensor], list[tuple[int, int, int]]]:
        """Split a shortest-edge-resized image into HF-style EoMT patches."""
        _, height, width = tensor.shape
        longer_side = max(height, width)
        num_patches = int(math.ceil(longer_side / patch_size))
        total_overlap = num_patches * patch_size - longer_side
        overlap_per_patch = total_overlap / (num_patches - 1) if num_patches > 1 else 0

        patches: list[torch.Tensor] = []
        offsets: list[tuple[int, int, int]] = []
        for i in range(num_patches):
            start = int(i * (patch_size - overlap_per_patch))
            end = start + patch_size
            patch = (
                tensor[:, start:end, :] if height > width else tensor[:, :, start:end]
            )
            if patch.shape[-2:] != (patch_size, patch_size):
                raise RuntimeError(
                    "LibreEoMT split preprocessing produced a non-square patch "
                    f"{patch.shape[-2:]} for resized image {(height, width)}."
                )
            patches.append(patch)
            offsets.append((0, start, end))
        return patches, offsets

    def _preprocess_pil_split(
        self,
        img: Image.Image,
        input_size: int,
    ) -> tuple[torch.Tensor, tuple[int, int], list[tuple[int, int, int]]]:
        orig_w, orig_h = img.size
        resized_h, resized_w = self._shortest_edge_size(orig_h, orig_w, input_size)
        resized = TVF.resize(
            TVF.pil_to_tensor(img).unsqueeze(0),
            [resized_h, resized_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )[0].float()
        resized.div_(255.0)
        patches, offsets = self._split_resized_tensor(resized, input_size)
        tensors = [patch.contiguous() for patch in patches]
        return torch.stack(tensors, dim=0), (resized_h, resized_w), offsets

    @staticmethod
    def _stitch_patch_logits(
        logits: torch.Tensor,
        patch_offsets: list[tuple[int, int, int]],
        *,
        resized_shape: tuple[int, int],
        original_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Merge per-patch semantic logits using HF EoMT overlap averaging."""
        resized_h, resized_w = resized_shape
        orig_h, orig_w = original_shape
        num_classes = int(logits.shape[1])
        merged = torch.zeros(
            (num_classes, resized_h, resized_w),
            dtype=logits.dtype,
            device=logits.device,
        )
        counts = torch.zeros(
            (1, resized_h, resized_w),
            dtype=logits.dtype,
            device=logits.device,
        )

        vertical = resized_h > resized_w
        for patch_idx, (_, start, end) in enumerate(patch_offsets):
            if vertical:
                merged[:, start:end, :] += logits[patch_idx]
                counts[:, start:end, :] += 1
            else:
                merged[:, :, start:end] += logits[patch_idx]
                counts[:, :, start:end] += 1

        averaged = merged / counts.clamp(min=1)
        return F.interpolate(
            averaged.unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )[0]

    @staticmethod
    def _coco_content_size(orig_h: int, orig_w: int, size: int) -> Tuple[int, int]:
        """Aspect-preserving size whose longest edge is ``size``.

        Mirrors the DETR-style ``get_size_with_aspect_ratio`` that the upstream
        EoMT image processor uses for the COCO checkpoints, where both
        ``shortest_edge`` and ``longest_edge`` equal the model resolution.
        """
        min_o, max_o = float(min(orig_h, orig_w)), float(max(orig_h, orig_w))
        target = size
        raw: Optional[float] = None
        if max_o / min_o * size > size:
            raw = size * min_o / max_o
            target = int(round(raw))
        if (orig_h <= orig_w and orig_h == target) or (
            orig_w <= orig_h and orig_w == target
        ):
            return orig_h, orig_w
        if orig_w < orig_h:
            out_w = target
            out_h = (
                int(raw * orig_h / orig_w)
                if raw is not None
                else int(size * orig_h / orig_w)
            )
        else:
            out_h = target
            out_w = (
                int(raw * orig_w / orig_h)
                if raw is not None
                else int(size * orig_w / orig_h)
            )
        return out_h, out_w

    def _preprocess_pil_pad(
        self,
        img: Image.Image,
        input_size: int,
    ) -> tuple[torch.Tensor, Tuple[int, int]]:
        """Resize the longest edge to ``input_size`` and zero-pad to a square.

        Padding is applied to the raw [0, 1] image; :class:`LibreEoMTNet`
        normalizes afterwards, so the padded region lands on ``-mean/std``,
        which is exactly what the upstream processor produces.
        """
        orig_w, orig_h = img.size
        content_h, content_w = self._coco_content_size(orig_h, orig_w, input_size)
        resized = TVF.resize(
            TVF.pil_to_tensor(img).unsqueeze(0),
            [content_h, content_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )[0].float()
        resized.div_(255.0)
        canvas = torch.zeros((3, input_size, input_size), dtype=resized.dtype)
        canvas[:, :content_h, :content_w] = resized
        return canvas.unsqueeze(0), (content_h, content_w)

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_res = input_size if input_size is not None else self.input_size
        if effective_res % self.semantic_imgsz_divisor:
            raise ValueError(
                f"LibreEoMT imgsz={effective_res} must be divisible "
                f"by {self.semantic_imgsz_divisor} (EoMT patch grid)."
            )
        if effective_res != self.input_size:
            raise ValueError(
                f"LibreEoMT requires imgsz={self.input_size}; got imgsz="
                f"{effective_res}. The EoMT checkpoint uses fixed "
                "position embeddings."
            )
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size

        # Upstream splits only the ADE20K semantic checkpoint into sliding-window
        # patches (do_split_image=True). The COCO instance and panoptic
        # checkpoints are resized to fit the longest edge and zero-padded to a
        # square (do_split_image=False, do_pad=True). Splitting those would hand
        # the same object to two patches as two independent queries.
        if self.task == "semantic":
            img_tensor, resized_shape, patch_offsets = self._preprocess_pil_split(
                img,
                effective_res,
            )
            self._last_eomt_resized_shape = resized_shape
            self._last_eomt_patch_offsets = patch_offsets
            self._last_eomt_content_size = None
        else:
            img_tensor, content_size = self._preprocess_pil_pad(img, effective_res)
            self._last_eomt_resized_shape = None
            self._last_eomt_patch_offsets = None
            self._last_eomt_content_size = content_size

        return img_tensor, img, (orig_w, orig_h), 1.0

    def _unpad_and_resize_mask_logits(
        self,
        mask_logits: torch.Tensor,
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Crop the zero-padded border off (Q, S, S) logits and resize to the image.

        Bilinear interpolation happens on logits, not on sigmoid probabilities,
        matching the upstream post-process.
        """
        orig_w, orig_h = original_size
        content = getattr(self, "_last_eomt_content_size", None)
        if content is not None:
            content_h, content_w = content
            mask_logits = mask_logits[:, :content_h, :content_w]
        return F.interpolate(
            mask_logits.unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )[0]

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    @staticmethod
    def _masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
        """Convert (K, H, W) masks to (K, 4) xyxy boxes in absolute pixel coords."""
        k = masks.shape[0]
        if k == 0:
            return torch.zeros((0, 4), dtype=torch.float32, device=masks.device)
        boxes = []
        for mask in masks:
            nonzero = mask.nonzero()
            if len(nonzero) == 0:
                boxes.append(torch.zeros(4, dtype=torch.float32, device=masks.device))
            else:
                y1, x1 = nonzero.min(0).values.float()
                y2, x2 = nonzero.max(0).values.float()
                boxes.append(torch.stack([x1, y1, x2, y2]))
        return torch.stack(boxes)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        if self.task == "segment":
            return self._postprocess_segment(
                output, conf_thres, iou_thres, original_size, max_det
            )
        if self.task == "panoptic":
            return self._postprocess_panoptic(output, conf_thres, original_size)
        return self._postprocess_semantic(output, original_size)

    def _stuff_class_ids(self) -> set[int]:
        """Class ids that are 'stuff' (fused into one segment per category).

        thing-vs-stuff is a per-category property of the label set, carried on
        the checkpoint as ``thing_class_ids`` (see ``docs/dataset_schema.md``).
        Without it we cannot tell stuff from things, so nothing is fused and
        every region becomes its own segment: wrong, but loudly wrong rather
        than silently mislabeled.
        """
        thing_ids = getattr(self, "thing_class_ids", None)
        if not thing_ids:
            logger.warning(
                "LibreEoMT panoptic checkpoint has no 'thing_class_ids' metadata; "
                "treating every category as a thing. Stuff regions will not be "
                "fused into single segments. Re-convert with "
                "weights/convert_eomt_weights.py to embed the split."
            )
            return set()
        return set(range(self.nb_classes)) - set(int(i) for i in thing_ids)

    def _panoptic_queries(
        self, output: Any, original_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode raw panoptic forward output into filtered (score, label,
        full-resolution mask-probability) queries.

        Drops no-object and low-confidence queries (``PANOPTIC_SCORE_THRESHOLD``).
        Shared by ``_postprocess_panoptic`` (single-view predict/val) and
        ``LibreEoMT._predict_augment_panoptic`` (flip TTA), which concatenates
        queries from both views before the shared ``_fuse_panoptic_queries``
        assignment step. Returns 0-length tensors when nothing survives.
        """
        if not isinstance(output, dict):
            raise ValueError("LibreEoMT panoptic forward must return a dict.")
        class_logits = output.get("class_queries_logits")  # (P, Q, C+1)
        mask_logits = output.get("masks_queries_logits")  # (P, Q, h, w)
        if class_logits is None or mask_logits is None:
            raise ValueError(
                "LibreEoMT panoptic forward did not include class_queries_logits "
                "and masks_queries_logits."
            )

        orig_w, orig_h = original_size
        nc = int(class_logits.shape[-1]) - 1  # last logit is the null/no-object class

        # The query score threshold is a merge hyperparameter, not a detection
        # confidence, so panoptic ignores predict(conf=...) exactly as semantic
        # does. Tune via PANOPTIC_SCORE_THRESHOLD.
        scores, labels = class_logits[0].softmax(dim=-1).max(-1)  # over C+1
        keep = (labels != nc) & (scores > self.PANOPTIC_SCORE_THRESHOLD)
        if not keep.any():
            return scores[keep], labels[keep], mask_logits.new_zeros((0, orig_h, orig_w))
        scores, labels = scores[keep], labels[keep]

        # Crop the zero-padded border, resize logits to the image, then sigmoid.
        mask_probs = self._unpad_and_resize_mask_logits(
            mask_logits[0][keep], original_size
        ).sigmoid()
        return scores, labels, mask_probs

    def _fuse_panoptic_queries(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        mask_probs: torch.Tensor,
        original_size: Tuple[int, int],
    ) -> Dict:
        """Merge per-query classes and masks into one non-overlapping segment map.

        Implements the standard MaskFormer/Mask2Former panoptic inference recipe:
        assign every pixel to the query with the highest score-weighted mask
        probability, discard queries whose surviving area falls below
        ``PANOPTIC_OVERLAP_THRESHOLD`` of their own binarized area, and fuse
        all stuff segments of the same category. ``scores``/``labels``/
        ``mask_probs`` come from ``_panoptic_queries`` — concatenated across
        views for flip TTA, so this step is agnostic to how many queries
        there are or which view(s) they came from.
        """
        orig_w, orig_h = original_size
        empty = {
            "panoptic": torch.zeros((orig_h, orig_w), dtype=torch.int32),
            "segments_info": [],
        }
        if mask_probs.shape[0] == 0:
            return empty

        # Every pixel goes to the query with the highest score-weighted mask
        # probability; a segment is the intersection of the pixels it won with
        # its own binarized mask. Everything else stays void (segment id 0).
        winner = (mask_probs * scores.view(-1, 1, 1)).argmax(dim=0)

        stuff_ids = self._stuff_class_ids()
        segmentation = torch.zeros(
            (orig_h, orig_w), dtype=torch.int32, device=mask_probs.device
        )
        segments_info: list[dict] = []
        stuff_memory: Dict[int, int] = {}
        current_id = 0

        for k in range(mask_probs.shape[0]):
            label = int(labels[k])
            is_stuff = label in stuff_ids

            won = winner == k
            own_mask = mask_probs[k] >= self.PANOPTIC_MASK_THRESHOLD
            final_mask = won & own_mask
            won_area = int(won.sum())
            own_area = int(own_mask.sum())
            if won_area == 0 or own_area == 0 or int(final_mask.sum()) == 0:
                continue
            # Mostly-occluded queries are dropped rather than left as slivers.
            if won_area / own_area <= self.PANOPTIC_OVERLAP_THRESHOLD:
                continue

            if is_stuff and label in stuff_memory:
                # One segment per stuff category: grow the existing one.
                segmentation[final_mask] = stuff_memory[label]
                continue

            current_id += 1
            if is_stuff:
                stuff_memory[label] = current_id
            segmentation[final_mask] = current_id
            segments_info.append(
                {
                    "id": current_id,
                    "category_id": label,
                    "isthing": not is_stuff,
                    "score": round(float(scores[k]), 6),
                }
            )

        if not segments_info:
            return empty
        return {"panoptic": segmentation.cpu(), "segments_info": segments_info}

    def _postprocess_panoptic(
        self,
        output: Any,
        conf_thres: float,
        original_size: Tuple[int, int],
    ) -> Dict:
        """Merge per-query classes and masks into one non-overlapping segment map.

        Drops no-object and low-confidence queries, assigns every pixel to
        the query with the highest score-weighted mask probability, discards
        mostly-occluded queries, and fuses same-category stuff segments —
        see ``_panoptic_queries``/``_fuse_panoptic_queries`` for the two
        halves of this recipe.

        Unlike ``_postprocess_segment`` this drops queries whose argmax over the
        ``C + 1`` logits is the null class, which is what keeps a panoptic map
        from filling with junk segments.
        """
        scores, labels, mask_probs = self._panoptic_queries(output, original_size)
        return self._fuse_panoptic_queries(scores, labels, mask_probs, original_size)

    def _dedup_panoptic_queries(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        mask_probs: torch.Tensor,
        view_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge same-class queries from different flip-TTA views that
        describe the same real instance, before fusion.

        Without this, two views correctly agreeing on the same object each
        contribute a near-duplicate query. ``_fuse_panoptic_queries``'s
        per-pixel winner-take-all then splits that object's pixels roughly
        evenly between the two duplicates, so *each* one's own overlap
        fraction can fall below ``PANOPTIC_OVERLAP_THRESHOLD`` and get
        dropped as "mostly occluded" — silently losing an object BOTH views
        detected correctly (measured empirically: 2 of 7 real segments lost
        on a real image before this dedup step was added).

        Greedy NMS-style clustering: process queries highest-score first; a
        lower-score query of the same class with binarized-mask IoU above
        ``PANOPTIC_TTA_DEDUP_IOU`` against an already-kept query is folded
        into it (mask probabilities and scores averaged) instead of
        competing against it separately in the fusion step.

        ``view_ids`` (one int per query, matching ``scores``) restricts each
        group to at most one query per view. Without it, two genuinely
        distinct same-class instances the model detected within a single
        view (e.g. two overlapping people in a crowd) could have high enough
        mask IoU to be wrongly folded into one segment — a risk this
        function only exists to dedup cross-view duplicates, not to
        second-guess a single view's own instance separation. The same cap
        also stops one broad anchor query from absorbing *two* distinct,
        correctly-separated queries from the opposite view: greedy
        clustering otherwise has no limit on how many same-class queries a
        single anchor can accumulate, so an imprecise mask that happens to
        overlap two real neighboring instances above the threshold would
        merge all three into one, silently losing an instance. Pass ``None``
        (the default) to merge on mask IoU alone, ignoring view origin —
        including this per-view cap.
        """
        n = mask_probs.shape[0]
        if n <= 1:
            return scores, labels, mask_probs

        binary_masks = mask_probs >= self.PANOPTIC_MASK_THRESHOLD
        areas = binary_masks.flatten(1).sum(dim=1).float()
        order = torch.argsort(scores, descending=True).tolist()

        consumed = [False] * n
        out_scores: list[torch.Tensor] = []
        out_labels: list[torch.Tensor] = []
        out_masks: list[torch.Tensor] = []

        for idx in order:
            if consumed[idx]:
                continue
            consumed[idx] = True
            group = [idx]
            group_views = {int(view_ids[idx])} if view_ids is not None else None
            for jdx in order:
                if jdx == idx or consumed[jdx] or int(labels[jdx]) != int(labels[idx]):
                    continue
                jdx_view = int(view_ids[jdx]) if view_ids is not None else None
                if group_views is not None and jdx_view in group_views:
                    # Either the anchor's own view, or a view already
                    # represented in this group — cap one query per view.
                    continue
                inter = (binary_masks[idx] & binary_masks[jdx]).sum().float()
                union = areas[idx] + areas[jdx] - inter
                iou = (inter / union) if union > 0 else inter.new_zeros(())
                if iou > self.PANOPTIC_TTA_DEDUP_IOU:
                    consumed[jdx] = True
                    group.append(jdx)
                    if group_views is not None:
                        group_views.add(jdx_view)

            group_t = torch.tensor(group, device=mask_probs.device)
            out_scores.append(scores[group_t].mean())
            out_labels.append(labels[idx])
            out_masks.append(mask_probs[group_t].mean(dim=0))

        return torch.stack(out_scores), torch.stack(out_labels), torch.stack(out_masks)

    def _predict_augment_panoptic(
        self,
        img_pil,
        image_path,
        original_size: Tuple[int, int],
        effective_imgsz,
        color_format: str,
        **kwargs,
    ) -> Any:
        """Flip-only TTA for panoptic segmentation.

        Concatenates the original and flipped views' surviving (score,
        label, mask) queries, merges same-instance duplicates across views
        (``_dedup_panoptic_queries``), and re-runs the existing per-pixel
        winner-take-all fusion once over the deduplicated set — the same
        generalization detection TTA uses (concatenate augmented-view
        boxes, then run NMS once).
        """
        from PIL import Image as PILImage

        from ...utils.results import PanopticSegmentation, Results

        orig_w, orig_h = original_size
        scores_views = []
        labels_views = []
        mask_probs_views = []
        view_ids_views = []
        # Each view's preprocess -> forward -> query-decode runs in sequence,
        # not interleaved: _preprocess stashes per-call instance state
        # (self._last_eomt_content_size) that _panoptic_queries reads back
        # via _unpad_and_resize_mask_logits, so a later _preprocess call
        # must not run before the earlier view's decode has consumed it.
        for is_flipped in (False, True):
            src = (
                img_pil.transpose(PILImage.Transpose.FLIP_LEFT_RIGHT)
                if is_flipped
                else img_pil
            )
            tensor, _, orig_size, _ = self._preprocess(
                src, color_format, input_size=effective_imgsz
            )
            with torch.no_grad():
                raw = self._forward(tensor.to(self.device))
            scores, labels, mask_probs = self._panoptic_queries(raw, orig_size)
            if is_flipped and mask_probs.shape[0] > 0:
                mask_probs = mask_probs.flip(-1)
            scores_views.append(scores)
            labels_views.append(labels)
            mask_probs_views.append(mask_probs)
            # Tags each query with the view it came from, so dedup only ever
            # merges cross-view duplicates of the same real object — never
            # two genuinely distinct same-class instances one view detected
            # on its own (e.g. two overlapping people in a crowd).
            view_ids_views.append(
                torch.full((scores.shape[0],), int(is_flipped), dtype=torch.long)
            )

        scores = torch.cat(scores_views, dim=0)
        labels = torch.cat(labels_views, dim=0)
        mask_probs = torch.cat(mask_probs_views, dim=0)
        view_ids = torch.cat(view_ids_views, dim=0)
        scores, labels, mask_probs = self._dedup_panoptic_queries(
            scores, labels, mask_probs, view_ids
        )
        detections = self._fuse_panoptic_queries(scores, labels, mask_probs, original_size)

        return Results(
            boxes=None,
            orig_shape=(orig_h, orig_w),
            path=str(image_path) if image_path else None,
            names=self.names,
            panoptic=PanopticSegmentation(
                detections["panoptic"].long(),
                detections.get("segments_info") or [],
                (orig_h, orig_w),
            ),
        )

    def _postprocess_semantic_logits(
        self, output: Any, original_size: Tuple[int, int], **kwargs
    ) -> torch.Tensor:
        """Interpolate/stitch raw semantic logits to ``original_size``, pre-argmax.

        Shared by ``_postprocess_semantic`` (single-view predict/val),
        ``LibreEoMT.val()``'s own per-image augment branch, and
        ``BaseModel._predict_augment_semantic`` (flip TTA), which needs the
        pre-argmax logits to average across augmented views. Always returns
        a ``[1, C, H, W]`` tensor, whether or not the patch-stitch branch
        (only taken when ``_preprocess`` split the image into sliding-window
        patches, i.e. per-image predict — never during batched validation)
        fires.
        """
        logits = output
        if isinstance(logits, dict):
            logits = logits.get("semantic_logits", logits.get("logits"))
        if logits is None:
            raise ValueError(
                "LibreEoMT forward output did not include semantic logits."
            )
        orig_w, orig_h = original_size
        patch_offsets = getattr(self, "_last_eomt_patch_offsets", None)
        resized_shape = getattr(self, "_last_eomt_resized_shape", None)
        if (
            patch_offsets
            and resized_shape
            and len(patch_offsets) == int(logits.shape[0])
        ):
            logits_hw = self._stitch_patch_logits(
                logits.float(),
                patch_offsets,
                resized_shape=resized_shape,
                original_shape=(orig_h, orig_w),
            )
            return logits_hw.unsqueeze(0)

        return F.interpolate(
            logits.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

    def _postprocess_semantic(
        self, output: Any, original_size: Tuple[int, int]
    ) -> Dict:
        logits = self._postprocess_semantic_logits(output, original_size)
        return {"semantic": logits.argmax(dim=1)[0].cpu()}

    def _postprocess_segment(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
    ) -> Dict:
        """Decode instance segmentation from per-query class and mask logits.

        The COCO instance checkpoints are padded, not split, so there is a single
        forward pass whose masks are cropped back to the unpadded content before
        being resized onto the original canvas.
        """
        if not isinstance(output, dict):
            raise ValueError("LibreEoMT segment forward must return a dict.")
        class_logits = output.get("class_queries_logits")  # (B, Q, C+1)
        mask_logits = output.get("masks_queries_logits")  # (B, Q, H, W)
        if class_logits is None or mask_logits is None:
            raise ValueError(
                "LibreEoMT segment forward did not include class_queries_logits "
                "and masks_queries_logits."
            )
        orig_w, orig_h = original_size
        patch_offsets = getattr(self, "_last_eomt_patch_offsets", None)
        resized_shape = getattr(self, "_last_eomt_resized_shape", None)
        num_patches = int(class_logits.shape[0])
        resized_h, resized_w = resized_shape if resized_shape else (orig_h, orig_w)
        has_patches = (
            patch_offsets is not None
            and resized_shape is not None
            and len(patch_offsets) == num_patches
        )

        all_scores: list[torch.Tensor] = []
        all_classes: list[torch.Tensor] = []
        all_boxes: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []

        for patch_idx in range(num_patches):
            cls_logit = class_logits[patch_idx]  # (Q, C+1)
            msk_logit = mask_logits[patch_idx]  # (Q, H, W)

            # DETR-style decoding: softmax, exclude null/background class
            scores_per_query = cls_logit.softmax(-1)[..., :-1]  # (Q, C)
            scores, labels = scores_per_query.max(-1)  # (Q,), (Q,)

            keep = scores > conf_thres
            if not keep.any():
                continue

            scores = scores[keep]
            labels = labels[keep]

            if has_patches:
                # Legacy split path (semantic-style preprocessing).
                binary_masks = (msk_logit[keep].sigmoid() > 0.5).float()
                _, start, end = patch_offsets[patch_idx]
                vertical = resized_h > resized_w
                full = torch.zeros(
                    (len(binary_masks), resized_h, resized_w),
                    dtype=binary_masks.dtype,
                    device=binary_masks.device,
                )
                if vertical:
                    full[:, start:end, :] = binary_masks
                else:
                    full[:, :, start:end] = binary_masks
                masks_orig = F.interpolate(
                    full.unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            else:
                # Padded path: crop the pad off the logits, resize, then binarize.
                masks_orig = self._unpad_and_resize_mask_logits(
                    msk_logit[keep], original_size
                ).sigmoid()
            masks_orig = (masks_orig > 0.5).float()

            boxes = self._masks_to_boxes(masks_orig)

            all_scores.append(scores.cpu())
            all_classes.append(labels.cpu())
            all_boxes.append(boxes.cpu())
            all_masks.append(masks_orig.cpu())

        if not all_scores:
            return {
                "boxes": [],
                "scores": [],
                "classes": [],
                "num_detections": 0,
                "masks": torch.zeros((0, orig_h, orig_w), dtype=torch.float32),
            }

        boxes_t = torch.cat(all_boxes, dim=0)  # (N, 4)
        scores_t = torch.cat(all_scores, dim=0)  # (N,)
        labels_t = torch.cat(all_classes, dim=0)  # (N,)
        masks_t = torch.cat(all_masks, dim=0)  # (N, H, W)

        if num_patches > 1:
            # Multi-patch: NMS merges predictions of the same object detected
            # in overlapping patches (cross-patch duplicates are expected).
            keep_idx = batched_nms(
                boxes_t.float(), scores_t.float(), labels_t, iou_thres
            )
            if len(keep_idx) > max_det:
                keep_idx = keep_idx[:max_det]
        else:
            # Single-patch: EoMT is DETR-style (queries are uniquely assigned by
            # bipartite matching); applying NMS can suppress valid overlapping
            # detections. Use top-k by confidence score instead.
            keep_idx = scores_t.argsort(descending=True)[:max_det]

        return {
            "boxes": boxes_t[keep_idx].tolist(),
            "scores": scores_t[keep_idx].tolist(),
            "classes": labels_t[keep_idx].tolist(),
            "num_detections": int(len(keep_idx)),
            "masks": masks_t[keep_idx],
        }

    def _load_weights(self, model_path: str | dict[str, Any]) -> None:
        if isinstance(model_path, (str, Path)):
            path = Path(model_path)
            if not path.exists():
                from ...utils.download import download_weights

                download_weights(str(model_path), self.size)
                path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model weights not found at {model_path}")
            loaded = load_untrusted_torch_file(
                path,
                map_location="cpu",
                context="EoMT weights",
            )
        else:
            loaded = model_path

        if not isinstance(loaded, dict):
            raise TypeError("LibreEoMT checkpoints must be dictionaries.")

        has_libreyolo_metadata = isinstance(loaded.get("model"), dict) and all(
            key in loaded
            for key in (
                "schema_version",
                "libreyolo_version",
                "model_family",
                "size",
                "task",
                "nc",
                "names",
                "imgsz",
            )
        )
        if not has_libreyolo_metadata:
            raise RuntimeError(
                "Raw EoMT state dicts are not loaded directly. Convert the "
                "approved DINOv2 ADE20K checkpoint with "
                "weights/convert_eomt_weights.py so LibreYOLO metadata and "
                "DINOv2-only provenance checks are applied."
            )

        ckpt_family = loaded.get("model_family", "")
        if ckpt_family and ckpt_family != self.FAMILY:
            raise RuntimeError(
                f"Checkpoint was trained with model_family='{ckpt_family}' "
                f"but is being loaded into '{self.FAMILY}'."
            )

        ckpt_task = loaded.get("task")
        if isinstance(ckpt_task, str):
            normalized_ckpt_task = normalize_task(ckpt_task)
            if normalized_ckpt_task != self.task:
                raise RuntimeError(
                    f"Checkpoint task={normalized_ckpt_task!r} does not match "
                    f"model task={self.task!r}. Pass task={normalized_ckpt_task!r} "
                    "when constructing LibreEoMT, or use the correct checkpoint."
                )

        state = _extract_state(loaded)
        state = normalize_eomt_state_dict(state)
        if not self.can_load(state):
            raise RuntimeError(
                "Checkpoint does not look like a LibreEoMT model "
                "(missing EoMT query, mask head, class head, or patch embedding keys)."
            )

        # Detect backbone size (s/b/l) from embed dim before any other rebuild,
        # since size determines the architecture that all other rebuilds use.
        ckpt_size = loaded.get("size") or self.detect_size(state)
        if ckpt_size is not None and ckpt_size != self.size:
            self._rebuild_for_new_size(ckpt_size)

        ckpt_nc = loaded.get("nc")
        if ckpt_nc is None:
            names = loaded.get("names")
            ckpt_nc = len(names) if names else None
        if ckpt_nc is None:
            ckpt_nc = self.detect_nb_classes(state)
        if ckpt_nc is not None and int(ckpt_nc) != self.nb_classes:
            self._rebuild_for_new_classes(int(ckpt_nc))

        ckpt_num_queries = self.detect_num_queries(state)
        if ckpt_num_queries is not None and ckpt_num_queries != self.num_queries:
            self._rebuild_for_new_queries(ckpt_num_queries)

        # State-dict detection is authoritative: position embeddings encode the
        # native resolution and cannot be wrong. Metadata imgsz is a hint only.
        ckpt_imgsz = self.detect_image_size(state)
        if ckpt_imgsz is None:
            ckpt_imgsz = loaded.get("imgsz") if isinstance(loaded, dict) else None
        if ckpt_imgsz is not None and int(ckpt_imgsz) != self.input_size:
            self._rebuild_for_new_image_size(int(ckpt_imgsz))

        result = self.model.load_state_dict(state, strict=self._strict_loading())
        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        if missing:
            logger.debug("LibreEoMT missing checkpoint keys: %s", missing[:8])
        if unexpected:
            logger.debug("LibreEoMT unexpected checkpoint keys: %s", unexpected[:8])

        ckpt_names = loaded.get("names")
        if ckpt_names is not None:
            self.names = self._sanitize_names(ckpt_names, self.nb_classes)

        # Panoptic label sets carry their thing/stuff split as category metadata;
        # the panoptic merge needs it to know which categories to fuse.
        ckpt_thing_ids = loaded.get("thing_class_ids")
        if ckpt_thing_ids is not None:
            self.thing_class_ids = [int(i) for i in ckpt_thing_ids]

        self.model.to(self.device)

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training LibreEoMT is out of scope for LibreYOLO v1. "
            f"Fine-tune upstream at {self._UPSTREAM_URL} and convert the result "
            "with weights/convert_eomt_weights.py."
        )

    def export(self, format: str = "onnx", *, opset: int = 17, **kwargs) -> str:
        if self.task == "semantic":
            return super().export(format=format, opset=opset, **kwargs)
        raise NotImplementedError(
            "LibreEoMT instance and panoptic export need query-mask runtime contracts."
        )

    def val(
        self,
        data: str | None = None,
        batch: int = 1,
        imgsz: int | None = None,
        conf: float = 0.001,
        iou: float = 0.6,
        workers: int = 0,
        allow_download_scripts: bool = False,
        device: str | None = None,
        split: str = "val",
        augment: bool = False,
        save_json: bool = False,
        verbose: bool = True,
        *args,
        plots: bool | None = None,
        save_plots: bool = False,
        save_dir: str | None = None,
        half: bool = False,
        amp_dtype: str = "float16",
        **kwargs,
    ):
        amp_dtype = normalize_amp_dtype(amp_dtype)
        if self.task in ("segment", "panoptic"):
            # segment scores mask mAP (SegmentationValidator); panoptic scores
            # Panoptic Quality (PanopticValidator). Both use the base dispatch;
            # only semantic needs the custom dense-mask mIoU loop below.
            return super().val(
                data=data,
                batch=batch,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                workers=workers,
                allow_download_scripts=allow_download_scripts,
                device=device,
                split=split,
                augment=augment,
                save_json=save_json,
                verbose=verbose,
                plots=plots,
                save_plots=save_plots,
                save_dir=save_dir,
                half=half,
                amp_dtype=amp_dtype,
                **kwargs,
            )

        conf_thres = float(conf)
        iou_thres = float(iou)
        if not 0 <= conf_thres < 1:
            raise ValueError(f"conf must be in [0, 1), got {conf_thres}.")
        if not 0 < iou_thres < 1:
            raise ValueError(f"iou must be in (0, 1), got {iou_thres}.")
        if args:
            raise TypeError(
                "LibreEoMT.val() does not accept extra positional arguments."
            )
        data_dir = kwargs.pop("data_dir", None)
        max_det = kwargs.pop("max_det", 300)
        eval_max_det = kwargs.pop("eval_max_det", None)
        iou_thresholds = kwargs.pop("iou_thresholds", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(
                f"LibreEoMT.val() got unexpected keyword argument(s): {names}."
            )
        if data_dir is not None:
            raise ValueError(
                "LibreEoMT validation requires data= (a semantic dataset YAML); "
                "data_dir is not supported."
            )
        if max_det != 300:
            logger.warning("LibreEoMT semantic validation ignores max_det=%s.", max_det)
        if eval_max_det is not None:
            logger.warning(
                "LibreEoMT semantic validation ignores eval_max_det=%s.",
                eval_max_det,
            )
        if iou_thresholds is not None:
            logger.warning("LibreEoMT semantic validation ignores iou_thresholds.")
        if int(batch) != 1:
            logger.warning(
                "LibreEoMT validation processes split-image inference one image "
                "at a time; batch=%s is ignored.",
                batch,
            )
        if int(workers) != 0:
            logger.warning(
                "LibreEoMT validation does not use dataloader workers; workers=%s "
                "is ignored.",
                workers,
            )
        if save_json:
            raise ValueError(
                "LibreEoMT semantic validation does not support save_json output."
            )
        if plots is not None and not save_plots:
            save_plots = bool(plots)
        if save_plots:
            logger.warning("LibreEoMT validation does not generate plots yet.")
        if data is None:
            raise ValueError("LibreEoMT validation requires data= (a dataset YAML).")
        effective_imgsz = self.input_size if imgsz is None else int(imgsz)
        if effective_imgsz != self.input_size:
            raise ValueError(
                f"LibreEoMT validation requires imgsz={self.input_size}; got "
                f"imgsz={effective_imgsz}. The EoMT checkpoint uses fixed "
                "position embeddings."
            )

        from ...data.semantic_dataset import (
            _apply_label_mapping,
            _load_mask_image,
            img2mask_paths,
            resolve_semantic_data,
        )
        from ...data.utils import get_img_files
        from ...utils.tta import average_flip_softmax

        if device is not None and str(device).lower() != "auto":
            device_str = f"cuda:{device}" if str(device).isdigit() else str(device)
            self.device = torch.device(device_str)
            self.model.to(self.device)

        data_config = resolve_semantic_data(
            data,
            allow_scripts=allow_download_scripts,
        )
        split_value = data_config.get(split)
        if not split_value:
            raise ValueError(f"Semantic dataset config has no '{split}' split.")
        img_files = data_config.get(f"{split}_img_files") or get_img_files(split_value)
        if not img_files:
            raise FileNotFoundError(
                f"No images found for semantic split '{split}' at {split_value}."
            )
        masks_dir = data_config.get("masks_dir")
        if not masks_dir:
            raise ValueError("LibreEoMT validation requires dense PNG masks.")
        mask_files = img2mask_paths(img_files, str(masks_dir))
        missing = [str(path) for path in mask_files if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} semantic mask file(s) missing for split "
                f"{split!r}; first missing: {missing[0]}"
            )

        nc = int(data_config.get("nc") or len(data_config.get("names") or {}))
        if int(self.nb_classes) != nc:
            raise ValueError(
                f"Semantic dataset has {nc} classes but LibreEoMT predicts "
                f"{self.nb_classes}."
            )
        names = data_config.get("names") or {}
        if isinstance(names, list):
            names = {i: name for i, name in enumerate(names)}
        names = {int(k): str(v) for k, v in names.items()}

        ignore_index = int(data_config.get("ignore_index", 255))
        raw_mapping = data_config.get("label_mapping") or None
        label_mapping = (
            {int(k): int(v) for k, v in raw_mapping.items()} if raw_mapping else None
        )

        confusion = torch.zeros((nc, nc), dtype=torch.int64)
        self.model.eval()
        start_time = time.time()
        preprocess_time = 0.0
        inference_time = 0.0
        postprocess_time = 0.0
        iterator = tqdm(
            list(zip(img_files, mask_files)),
            desc="Validating",
            total=len(img_files),
            disable=not verbose or not sys.stderr.isatty(),
            file=sys.stderr,
        )

        with torch.no_grad():
            for img_path, mask_path in iterator:
                with Image.open(img_path) as img_pil:
                    orig_shape = (img_pil.height, img_pil.width)

                target_np = _load_mask_image(mask_path)
                if target_np.shape != orig_shape:
                    raise ValueError(
                        f"Semantic mask {mask_path} shape {target_np.shape} "
                        f"does not match image shape {orig_shape}."
                    )
                if label_mapping:
                    target_np = _apply_label_mapping(
                        target_np,
                        label_mapping,
                        ignore_index,
                    )
                invalid = (target_np != ignore_index) & (
                    (target_np < 0) | (target_np >= nc)
                )
                if bool(invalid.any()):
                    bad = sorted(np.unique(target_np[invalid]).tolist())[:5]
                    raise ValueError(
                        f"Semantic mask {mask_path} contains class IDs {bad} "
                        f"outside 0..{nc - 1} (ignore={ignore_index}). "
                        "Use label_mapping to remap source IDs."
                    )

                # Original and (if augment) flipped views each run their own
                # preprocess -> forward -> postprocess in sequence, not
                # interleaved: _preprocess stashes per-call instance state
                # (self._last_eomt_patch_offsets / _resized_shape) that
                # _postprocess_semantic_logits reads back, so a later
                # _preprocess call must not run before the earlier view's
                # postprocess has consumed its own state.
                t1 = time.time()
                tensor, loaded_img, original_size, _ = self._preprocess(
                    img_path,
                    color_format="rgb",
                    input_size=effective_imgsz,
                )
                preprocess_time += time.time() - t1

                t2 = time.time()
                if half and self.device.type == "cuda":
                    with torch.amp.autocast(
                        "cuda", dtype=torch_amp_dtype(amp_dtype)
                    ):
                        output = self._forward(tensor.to(self.device))
                else:
                    output = self._forward(tensor.to(self.device))
                inference_time += time.time() - t2

                t3 = time.time()
                logits = self._postprocess_semantic_logits(output, original_size)
                if not augment:
                    pred = logits.argmax(dim=1)[0].cpu().long().view(-1)
                postprocess_time += time.time() - t3

                if augment:
                    # Reuse the image _preprocess already loaded from disk
                    # instead of reading img_path a second time.
                    t1f = time.time()
                    flipped = loaded_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    flip_tensor, _, flip_original_size, _ = self._preprocess(
                        flipped,
                        color_format="rgb",
                        input_size=effective_imgsz,
                    )
                    preprocess_time += time.time() - t1f

                    t2f = time.time()
                    if half and self.device.type == "cuda":
                        with torch.amp.autocast(
                            "cuda", dtype=torch_amp_dtype(amp_dtype)
                        ):
                            flip_output = self._forward(flip_tensor.to(self.device))
                    else:
                        flip_output = self._forward(flip_tensor.to(self.device))
                    inference_time += time.time() - t2f

                    t3f = time.time()
                    flip_logits = self._postprocess_semantic_logits(
                        flip_output, flip_original_size
                    ).flip(-1)
                    avg_probs = average_flip_softmax(logits, flip_logits)
                    pred = avg_probs.argmax(dim=1)[0].cpu().long().view(-1)
                    postprocess_time += time.time() - t3f

                target = (
                    torch.from_numpy(np.ascontiguousarray(target_np)).long().view(-1)
                )
                valid = target != ignore_index
                if not bool(valid.any()):
                    continue
                target_valid = target[valid]
                pred_valid = pred[valid].clamp_(0, nc - 1)
                index = target_valid * nc + pred_valid
                counts = torch.bincount(index, minlength=nc**2)
                confusion += counts.reshape(nc, nc)

        total = confusion.sum()
        true_positive = confusion.diag().double()
        union = (
            confusion.sum(dim=0).double()
            + confusion.sum(dim=1).double()
            - true_positive
        )
        per_class_iou = torch.full((nc,), float("nan"), dtype=torch.float64)
        present = union > 0
        per_class_iou[present] = true_positive[present] / union[present]
        observed = ~torch.isnan(per_class_iou)
        miou = float(per_class_iou[observed].mean()) if bool(observed.any()) else 0.0
        accuracy = float(confusion.diag().sum() / total) if total > 0 else 0.0

        if verbose:
            logger.info("=" * 50)
            logger.info("LibreEoMT Semantic Segmentation Validation Results")
            logger.info("=" * 50)
            for class_id, value in enumerate(per_class_iou):
                if torch.isnan(value):
                    continue
                logger.info(
                    "  IoU %-20s %.4f", names.get(class_id, str(class_id)), float(value)
                )
            logger.info("  mIoU:           %.4f", miou)
            logger.info("  pixel accuracy: %.4f", accuracy)
            logger.info("=" * 50)

        elapsed = max(time.time() - start_time, 0.0)
        seen = len(img_files)
        metrics = {
            "metrics/mIoU": miou,
            "metrics/pixel_accuracy": accuracy,
            "fitness": miou,
            "speed/preprocess_ms": preprocess_time / seen * 1000,
            "speed/inference_ms": inference_time / seen * 1000,
            "speed/postprocess_ms": postprocess_time / seen * 1000,
            "speed/total_ms": elapsed / seen * 1000,
            "speed/total_s": elapsed,
            "speed/images_seen": seen,
        }
        if save_dir is not None:
            import yaml

            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            config = {
                "data": data,
                "split": split,
                "batch_size": int(batch),
                "imgsz": effective_imgsz,
                "conf_thres": conf_thres,
                "iou_thres": iou_thres,
                "num_workers": int(workers),
                "allow_download_scripts": bool(allow_download_scripts),
                "device": str(self.device),
                "augment": bool(augment),
                "save_json": bool(save_json),
                "save_plots": bool(save_plots),
                "half": bool(half),
                "amp_dtype": amp_dtype,
            }
            with open(save_path / "config.yaml", "w") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
        return metrics


__all__ = ["LibreEoMT"]
