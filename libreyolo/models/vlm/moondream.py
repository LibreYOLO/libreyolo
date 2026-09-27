"""LibreYOLO wrapper for Moondream vision-language models.

Moondream exposes typed CV skills (``detect``, ``point``, ``query``,
``caption``) instead of a chat-template JSON ask. ``detect`` returns
``{x_min, y_min, x_max, y_max}`` floats in ``[0, 1]`` for one object
prompt at a time, so this family fans out ``set_classes()`` the same way
North Micro Vision does: one skill call per label, then the shared
``Results`` builder.

Two sizes:

* ``2`` — ``vikhyatk/moondream2`` (Apache-2.0, ~2B). Default.
* ``3`` — ``moondream/moondream3-preview`` (Business Source License 1.1,
  9B/2B-active MoE, ~24 GB). Logs a one-time license notice.

Both load through Hugging Face remote code, so every size pins a commit
SHA. Both sizes are mirrored on the LibreYOLO org. Size 3 ships the
upstream BSL 1.1 verbatim and logs a one-time notice; official notes ask
for about 24 GB of GPU memory.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch

from ...utils.image_loader import ImageInput, ImageLoader
from .base import _INSTALL_HINT, LibreVLMModel
from .locateanything import build_point_dict
from .parsing import build_detection_dict

logger = logging.getLogger(__name__)

_MD3_LICENSE_URL = "https://huggingface.co/moondream/moondream3-preview/blob/main/LICENSE.md"
_POST_INIT_PATCHED = False


def _ensure_remote_post_init(pretrained_cls) -> None:
    """Make transformers 5 finalize work with Moondream's remote model class."""
    global _POST_INIT_PATCHED
    if _POST_INIT_PATCHED:
        return
    original = pretrained_cls._finalize_model_loading

    def _finalize(model, load_config, loading_info):
        if not hasattr(model, "all_tied_weights_keys"):
            try:
                model.post_init()
            except Exception:
                model.all_tied_weights_keys = {}
        return original(model, load_config, loading_info)

    pretrained_cls._finalize_model_loading = staticmethod(_finalize)
    _POST_INIT_PATCHED = True


class _ImageCarrier:
    """PIL image with the ``.to(device)`` no-op InferenceRunner expects."""

    def __init__(self, img):
        self.img = img

    def to(self, *args, **kwargs):
        return self


def objects_to_box_items(objects, label: str) -> List[dict]:
    """Turn a Moondream ``detect()`` objects list into parser items."""
    items = []
    if not isinstance(objects, (list, tuple)):
        return items
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        try:
            box = [obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]]
        except (KeyError, TypeError):
            continue
        items.append({"label": label, "bbox": box})
    return items


def objects_to_point_items(points, label: str) -> List[dict]:
    """Turn a Moondream ``point()`` list into LocateAnything-shaped items."""
    items = []
    if not isinstance(points, (list, tuple)):
        return items
    for obj in points:
        if not isinstance(obj, dict):
            continue
        try:
            items.append({"label": label, "point": [obj["x"], obj["y"]]})
        except (KeyError, TypeError):
            continue
    return items


class LibreMoondream(LibreVLMModel):
    """Moondream used as an open-vocabulary detector (native detect skill)."""

    FAMILY = "moondream"
    FILENAME_PREFIX = "LibreMoondream"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "2": "LibreYOLO/LibreMoondream2",
        "3": "LibreYOLO/LibreMoondream3",
    }
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "2": "148fe3489ad456f3b0e5301d684116eb3ad2bece",
        "3": "27c8082b22b45de224c30431b3961ef8ee6c740e",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "2": 384,
        "3": 512,
    }
    MAX_OBJECTS: ClassVar[Dict[str, int]] = {
        "2": 50,
        "3": 150,
    }

    SUPPORTED_TASKS = ("detect", "point")
    DEFAULT_TASK = "detect"
    TRUST_REMOTE_CODE = True
    BBOX_KEY = "bbox"
    COORD_DIVISOR = 1.0
    BOX_FORMAT = "xyxy"

    _MD3_NOTICE = (
        "\n"
        "----------------------------------------------------------------\n"
        "Moondream 3 Preview weights are under the Business Source\n"
        "License 1.1 with an Additional Use Grant (no third-party service).\n"
        "Personal, research, and most in-product commercial use is allowed;\n"
        "selling a hosted/competing vision API is not. LibreYOLO hosts a\n"
        "verbatim-license mirror; you must comply with those terms. Full\n"
        "license:\n"
        f"  {_MD3_LICENSE_URL}\n"
        "----------------------------------------------------------------\n"
    )
    _MD3_NOTICE_SHOWN: ClassVar[bool] = False

    def _notify_license_once(self) -> None:
        if self.size != "3":
            return
        cls = type(self)
        if cls._MD3_NOTICE_SHOWN:
            return
        cls._MD3_NOTICE_SHOWN = True
        logger.warning(self._MD3_NOTICE)

    def _load_pretrained(self, snapshot_dir: str):
        try:
            from transformers import AutoModelForCausalLM
            from transformers.modeling_utils import PreTrainedModel
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        # Transformers 5.x sets ``all_tied_weights_keys`` in ``post_init``.
        # Moondream's remote HfMoondream never calls it, so finalize crashes.
        _ensure_remote_post_init(PreTrainedModel)
        dtype = self._resolve_dtype()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                snapshot_dir,
                dtype=dtype,
                trust_remote_code=self.TRUST_REMOTE_CODE,
            )
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            model = AutoModelForCausalLM.from_pretrained(
                snapshot_dir,
                torch_dtype=dtype,
                trust_remote_code=self.TRUST_REMOTE_CODE,
            )
        # Skills take a PIL image; there is no separate processor.
        return model, None

    def _max_objects(self) -> int:
        return int(self.MAX_OBJECTS.get(self.size, 50))

    def _labels_for_call(self) -> List[Tuple[int, str]]:
        return sorted(self.names.items())

    def chat(
        self,
        image: ImageInput,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        color_format: str = "auto",
    ) -> str:
        img = ImageLoader.load(image, color_format=color_format)
        query_kwargs = {"reasoning": False}
        if max_new_tokens is not None:
            query_kwargs["settings"] = {"max_tokens": max_new_tokens}
        with torch.no_grad():
            result = self.model.query(img, str(prompt), **query_kwargs)
        answer = result.get("answer", "") if isinstance(result, dict) else result
        if not isinstance(answer, str):
            answer = "".join(str(chunk) for chunk in answer)
        # This remote query() can loop on transformers 5; keep the first line.
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        return lines[0] if lines else answer

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size=None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        return _ImageCarrier(img), img, img.size, 1.0

    def _forward(self, inputs: Any) -> Any:
        # Do not pass a settings dict. This remote encode_image() requires
        # settings["variant"] whenever settings is not None, so {max_objects: N}
        # crashes. Upstream defaults are 50 (v2) and 150 (v3).
        img = inputs.img
        rows = []
        for _class_id, label in self._labels_for_call():
            if self.task == "point":
                result = self.model.point(img, label)
                rows.append((label, result.get("points", []) if isinstance(result, dict) else []))
            else:
                result = self.model.detect(img, label)
                rows.append((label, result.get("objects", []) if isinstance(result, dict) else []))
        return rows

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
        if self.task == "point":
            items = [
                item
                for label, points in output
                for item in objects_to_point_items(points, label)
            ]
            return build_point_dict(
                items,
                self._name_to_id,
                original_size,
                conf_thres=conf_thres,
                max_det=max_det,
                classes=kwargs.get("classes"),
                default_score=self._score_detections(items),
                coord_divisor=1.0,
            )

        items = [
            item
            for label, objects in output
            for item in objects_to_box_items(objects, label)
        ]
        return build_detection_dict(
            items,
            self._name_to_id,
            original_size,
            conf_thres=conf_thres,
            max_det=max_det,
            classes=kwargs.get("classes"),
            default_score=self._score_detections(items),
            bbox_key=self.BBOX_KEY,
            coord_divisor=self.COORD_DIVISOR,
            box_format=self.BOX_FORMAT,
            iou_thres=iou_thres,
        )
