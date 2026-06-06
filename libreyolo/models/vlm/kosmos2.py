"""LibreYOLO wrapper for Microsoft's Kosmos-2 grounding model.

Kosmos-2 (MIT, native ``Kosmos2ForConditionalGeneration``) is a grounded
multimodal model: given ``<grounding>`` text it generates a caption and grounds
the noun phrases, and its processor's ``post_process_generation`` returns the
entities with NORMALIZED [0,1] xyxy boxes. So this family overrides the inference
hooks (non-chat processor + entity post-processing) and scales the normalized
boxes to pixels.

Kosmos-2 is a 2023-era model: it loads cleanly and is a useful different
mechanism, but its boxes are coarse compared with newer grounders.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

from ...utils.image_loader import ImageInput, ImageLoader
from .base import LibreVLMModel


class LibreKosmos2(LibreVLMModel):
    """Kosmos-2 used as an open-vocabulary detector (grounded entities)."""

    FAMILY = "kosmos2"
    FILENAME_PREFIX = "LibreKosmos2"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "224": "microsoft/kosmos-2-patch14-224",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "224": 224,
    }

    # MIT weights: no restrictive-license notice needed.
    _LICENSE_NOTICE = ""

    def _match_label(self, name: str) -> Optional[int]:
        # Kosmos grounds noun phrases ("the boats"), so match leniently against
        # the vocabulary in addition to exact lookup.
        key = str(name).strip().lower()
        if key in self._name_to_id:
            return self._name_to_id[key]
        for cname, cid in self._name_to_id.items():
            if cname in key or key in cname:
                return cid
        return None

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size=None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        query = ", ".join(self.names[i] for i in range(len(self.names)))
        prompt = f"<grounding> Detect: {query}."
        inputs = self.processor(text=prompt, images=img, return_tensors="pt")
        return inputs, img, img.size, 1.0

    def _forward(self, inputs: Any) -> Any:
        inputs = self._prepare_generation_inputs(inputs)
        return self.model.generate(
            **inputs, max_new_tokens=self.MAX_NEW_TOKENS, do_sample=False
        )

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
        _caption, entities = self.processor.post_process_generation(text)
        width, height = original_size
        boxes, scores, classes = [], [], []
        allowed_classes = (
            set(kwargs["classes"]) if kwargs.get("classes") is not None else None
        )
        if max_det <= 0:
            return {
                "boxes": boxes,
                "scores": scores,
                "classes": classes,
                "num_detections": 0,
            }
        # Every box carries the placeholder score, so conf filtering is all-or-nothing.
        scored = entities if self.DEFAULT_SCORE >= conf_thres else []
        for name, _span, entity_boxes in scored:
            if len(boxes) >= max_det:
                break
            class_id = self._match_label(name)
            if class_id is None:
                continue
            if allowed_classes is not None and class_id not in allowed_classes:
                continue
            for box in entity_boxes:  # normalized [0,1] xyxy
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
                scores.append(self.DEFAULT_SCORE)
                classes.append(class_id)
                if len(boxes) >= max_det:
                    break
        return {
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "num_detections": len(boxes),
        }

    def chat(self, *args, **kwargs):
        raise NotImplementedError(
            "Kosmos-2 is driven by grounding prompts, not free-form chat; use predict()."
        )
