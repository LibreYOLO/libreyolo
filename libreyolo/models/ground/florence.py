"""Florence-2 used as a phrase grounder: box center becomes the click.

MIT weights, native transformers (``florence-community/*``), same load path
as ``LibreFlorence2``. Phrase grounding returns pixel boxes; this family
reduces each box to its center and emits ``Results.points``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Tuple

from ...utils.image_loader import ImageInput, ImageLoader
from ..vlm.base import _INSTALL_HINT
from .base import LibreGroundModel
from .parsing import build_point_dict


class LibreGroundFlorence2(LibreGroundModel):
    FAMILY = "ground_florence2"
    FILENAME_PREFIX = "LibreGroundFlorence2"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "base": "LibreYOLO/LibreGroundFlorence2base",
        "large": "florence-community/Florence-2-large",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "base": 768,
        "large": 768,
    }
    COORD_SPACE = "pixel"
    TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
    NUM_BEAMS = 3
    MAX_NEW_TOKENS = 256

    def _load_pretrained(self, snapshot_dir: str):
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        try:
            from transformers import Florence2ForConditionalGeneration as model_cls
        except ImportError:
            from transformers import AutoModelForImageTextToText as model_cls
        model = model_cls.from_pretrained(
            snapshot_dir,
            dtype=self._resolve_dtype(),
            trust_remote_code=self.TRUST_REMOTE_CODE,
        )
        processor = AutoProcessor.from_pretrained(
            snapshot_dir, trust_remote_code=self.TRUST_REMOTE_CODE
        )
        return model, processor

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size=None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        self._view_size = img.size
        inputs = self.processor(
            text=self.TASK + self._active_query(),
            images=img,
            return_tensors="pt",
        )
        return inputs, img, img.size, 1.0

    def _forward(self, inputs: Any) -> Any:
        inputs = self._prepare_generation_inputs(inputs)
        return self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=self.MAX_NEW_TOKENS,
            num_beams=self.NUM_BEAMS,
            do_sample=False,
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
        text = self.processor.batch_decode(output, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text, task=self.TASK, image_size=original_size
        )
        payload = parsed.get(self.TASK, {})
        boxes = payload.get("bboxes", [])
        labels = payload.get("bboxes_labels", payload.get("labels", []))
        items = []
        query = self._active_query()
        query_key = query.strip().lower()
        width, height = original_size
        frame_area = max(float(width) * float(height), 1.0)
        scored = []
        for index, box in enumerate(boxes):
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in box)
            except (TypeError, ValueError):
                continue
            label = labels[index] if index < len(labels) else query
            area = max(0.0, (x2 - x1) * (y2 - y1))
            scored.append(
                (
                    area,
                    {
                        "label": str(label) if label else query,
                        "point": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                    },
                )
            )
        # Drop whole-image boxes; phrase grounding often emits those first.
        compact = [item for area, item in scored if area < 0.5 * frame_area]
        pool = compact or [item for _, item in scored]
        matched = [
            item
            for item in pool
            if query_key in str(item.get("label") or "").strip().lower()
        ]
        if matched:
            pool = matched
        if pool:
            # Tightest remaining box is the click target.
            items = [
                min(
                    (
                        (area, item)
                        for area, item in scored
                        if item in pool
                    ),
                    key=lambda pair: pair[0],
                )[1]
            ]
        return build_point_dict(
            items,
            getattr(self, "_name_to_id", {}) or {query.lower(): 0},
            original_size,
            coord_space="pixel",
            conf_thres=conf_thres,
            max_det=1 if max_det is None else min(int(max_det), 1),
            classes=kwargs.get("classes"),
            default_score=self._score_detections(items),
        )

    def chat(self, *args, **kwargs):
        raise NotImplementedError(
            "Florence-2 is driven by task tokens, not free-form chat; use predict()."
        )
