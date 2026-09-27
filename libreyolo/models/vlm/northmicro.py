"""LibreYOLO wrapper for Cohere Labs' North Micro Vision VLM.

North Micro Vision (``CohereLabs/North-Micro-Vision-Instruct``, Apache-2.0) is
a 2.4B native-resolution VLM: a 400M vision encoder over a 2B North Micro LLM,
with grounding trained on boxes normalized to ``[x1, y1, x2, y2]`` on a 0-1000
scale.

It does not follow the tier's labeled-JSON detection ask: prompted for
``[{"label", "bbox"}]`` objects it answers with bare box arrays (or ``[]``),
matching its RefCOCO-style grounding training. So this family queries one class
per generation ("Locate every <label> ...") and assigns the label itself,
overriding ``_preprocess``/``_forward``/``_postprocess`` like the other
non-chat-format families. Cost: ``predict()`` runs one generation per
vocabulary entry, so keep ``set_classes()`` lists short; a query for an absent
but plausible label can hallucinate a box (RefCOCO-style grounders always try
to point at something).

The architecture (``CohereCompassForConditionalGeneration``) is native to
transformers from 5.16.0 (no remote code), so this family fails fast with an
install hint on older transformers instead of downloading ~5 GB of weights and
then crashing on an unknown ``model_type``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Tuple

from ...utils.image_loader import ImageInput, ImageLoader
from .base import LibreVLMModel
from .parsing import build_detection_dict, extract_bare_boxes

_MIN_TRANSFORMERS = (5, 16)
_VERSION_HINT = (
    "North Micro Vision requires transformers>=5.16.0 (the CohereCompass "
    "architecture). Upgrade with:\n"
    "    pip install -U 'transformers>=5.16.0'\n"
    "or, until 5.16.0 is on PyPI:\n"
    "    pip install git+https://github.com/huggingface/transformers"
)


class _ImageCarrier:
    """Wraps the loaded PIL image with the ``.to(device)`` no-op the shared
    ``InferenceRunner`` calls on every ``_preprocess`` output. Tokenization
    happens per class inside ``_forward``, so there is nothing to move yet."""

    def __init__(self, img):
        self.img = img

    def to(self, *args, **kwargs):
        return self


def _require_transformers() -> None:
    """Fail before the snapshot download if transformers cannot load the model."""
    import transformers

    parts = transformers.__version__.split(".")
    try:
        version = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):  # exotic dev version string: let it try
        return
    if version < _MIN_TRANSFORMERS:
        raise ImportError(
            f"{_VERSION_HINT}\n(found transformers {transformers.__version__})"
        )


class LibreNorthMicroVision(LibreVLMModel):
    """North Micro Vision repurposed as a closed-set object detector."""

    FAMILY = "northmicrovision"
    FILENAME_PREFIX = "LibreNorthMicroVision"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "2.4b": "CohereLabs/North-Micro-Vision-Instruct",
    }
    # Nominal only; the processor owns the real native-resolution handling.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "2.4b": 1024,
    }

    # Boxes come back on a 0-1000 scale (verified empirically on a known box:
    # px [200,150,400,300] of 800x600 comes back as ~[250,245,518,508]).
    COORD_DIVISOR = 1000.0

    # Per-class grounding ask. This is the phrasing the model actually follows;
    # it answers with ``[[x1, y1, x2, y2], ...]`` (no labels) or with a prose
    # refusal when the class is absent, which parses to zero boxes.
    _CLASS_PROMPT = (
        "Locate every {label} in the image and output the bounding boxes in "
        "JSON format: [[x1, y1, x2, y2], ...]. "
        "Coordinates are on a 0-1000 scale."
    )

    # Apache-2.0 weights: no restrictive-license notice needed.
    _LICENSE_NOTICE = ""

    def __init__(self, size: str, **kwargs):
        _require_transformers()
        super().__init__(size, **kwargs)

    def _class_prompt(self, label: str) -> str:
        # ``prompt=`` acts as a per-class template here; a ``{label}``
        # placeholder is substituted, a literal prompt is used as-is.
        template = self._custom_prompt or self._CLASS_PROMPT
        return template.format(label=label) if "{label}" in template else template

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size=None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        # Generation happens once per class, so tokenization is deferred to
        # ``_forward``; the "model input" is the loaded image itself.
        img = ImageLoader.load(image, color_format=color_format)
        return _ImageCarrier(img), img, img.size, 1.0

    def _forward(self, inputs: Any) -> Any:
        # One grounding query per vocabulary entry; the label is assigned by
        # the caller, not parsed from the generated text.
        return [
            (class_id, self.chat(inputs.img, self._class_prompt(label)))
            for class_id, label in sorted(self.names.items())
        ]

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
        items = [
            {"label": self.names[class_id], "bbox": box}
            for class_id, text in output
            for box in extract_bare_boxes(text)
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
