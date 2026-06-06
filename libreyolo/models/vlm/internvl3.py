"""LibreYOLO wrapper for OpenGVLab's InternVL3 vision-language models.

InternVL3 (native ``-hf`` checkpoints) is a strong open-weight VLM with good
grounding. The InternVL code is MIT, but the ``-hf`` weights use a Qwen LLM
backbone and inherit the Qwen License (not Apache/MIT), so the download is
notice-gated. It emits JSON with a ``bbox`` key, but on a 0-1000 scale and
with each object's box(es) wrapped in an extra list (its ``<box>[[...]]</box>``
heritage), e.g. ``[{"label": "boat", "bbox": [[120, 400, 250, 550]]}]`` and a
single object may carry several boxes. That nesting does not fit the simple
key/scale/format knobs, so this family overrides ``_postprocess`` to flatten the
boxes before the shared builder runs. It is the worked example of the override
path for a model whose output shape differs from the default.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Tuple

from .base import LibreVLMModel
from .parsing import build_detection_dict, extract_detections


class LibreInternVL3(LibreVLMModel):
    """InternVL3 used as an open-vocabulary detector (nested 0-1000 boxes)."""

    FAMILY = "internvl3"
    FILENAME_PREFIX = "LibreInternVL3"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "1b": "OpenGVLab/InternVL3-1B-hf",
        "2b": "OpenGVLab/InternVL3-2B-hf",
        "8b": "OpenGVLab/InternVL3-8B-hf",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "1b": 448,
        "2b": 448,
        "8b": 448,
    }

    # InternVL emits ``bbox`` corners on a 0-1000 scale.
    BBOX_KEY = "bbox"
    COORD_DIVISOR = 1000.0
    BOX_FORMAT = "xyxy"

    # Weights use a Qwen LLM backbone and carry the Qwen License (not Apache/MIT),
    # so the download is gated behind a one-time notice.
    _LICENSE_NOTICE = (
        "\n"
        "----------------------------------------------------------------\n"
        "InternVL3 -hf weights use a Qwen LLM backbone and are licensed\n"
        "under the Qwen License (not Apache/MIT). By downloading them you\n"
        "accept those terms. Full license:\n"
        "  https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE\n"
        "----------------------------------------------------------------\n"
    )

    def _format_detection_prompt(self, labels: str) -> str:
        return (
            f"Detect all instances of: {labels}. Respond with a JSON array "
            '[{"label": ..., "bbox": [x1, y1, x2, y2]}] using corner format. '
            "If none, respond with []."
        )

    @staticmethod
    def _flatten_nested(items):
        """Flatten InternVL's nested boxes so the shared builder can scale them.

        ``{"label": L, "bbox": [[box], [box]]}`` becomes one item per box; a flat
        ``bbox`` item (or any other shape) passes through unchanged.
        """
        flat = []
        for item in items:
            box = item.get("bbox")
            if isinstance(box, list) and box and isinstance(box[0], list):
                label = item.get("label")
                flat.extend({"label": label, "bbox": one} for one in box)
            else:
                flat.append(item)
        return flat

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
        text = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        flat = self._flatten_nested(extract_detections(text))
        return build_detection_dict(
            flat,
            self._name_to_id,
            original_size,
            conf_thres=conf_thres,
            max_det=max_det,
            classes=kwargs.get("classes"),
            default_score=self._score_detections(flat),
            bbox_key=self.BBOX_KEY,
            coord_divisor=self.COORD_DIVISOR,
            box_format=self.BOX_FORMAT,
        )
