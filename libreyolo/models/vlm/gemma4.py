"""LibreYOLO wrapper for Google's Gemma 4 vision-language models.

Gemma 4 (Apache-2.0) is a native multimodal family with object detection as a
documented skill. Detection replies are a JSON array of
``{"box_2d": [ymin, xmin, ymax, xmax], "label": ...}`` on a 0-1000 grid
(y-first, same convention as Gemini). The shared parser handles the JSON;
this family only sets the coordinate knobs, turns thinking off, and raises
the image token budget so small objects survive the resize.

Gemma 3 is intentionally not wrapped: its weights are gated under the Gemma
license and it has no native location format. Use the E2B / E4B instruct
checkpoints; larger Gemma 4 sizes are out of the small-VLM band.

Requires transformers 5.10+ (``AutoModelForMultimodalLM`` / Gemma 4 native
classes). Older installs fail before the weight download.
"""

from __future__ import annotations

from typing import ClassVar, Dict

from .base import _INSTALL_HINT, LibreVLMModel

_MIN_TRANSFORMERS = (5, 10)
_VERSION_HINT = (
    "Gemma 4 requires transformers>=5.10.0 (native Gemma 4 multimodal "
    "classes). Upgrade with:\n"
    "    pip install -U 'transformers>=5.10.0'"
)


def _require_transformers() -> None:
    import transformers

    parts = transformers.__version__.split(".")
    try:
        version = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return
    if version < _MIN_TRANSFORMERS:
        raise ImportError(
            f"{_VERSION_HINT}\n(found transformers {transformers.__version__})"
        )


class LibreGemma4(LibreVLMModel):
    """Gemma 4 used as an open-vocabulary detector (y-first box_2d)."""

    FAMILY = "gemma4"
    FILENAME_PREFIX = "LibreGemma4"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "e2b": "LibreYOLO/LibreGemma4e2b",
        "e4b": "LibreYOLO/LibreGemma4e4b",
    }
    # Nominal only; the processor owns variable-resolution resize.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "e2b": 896,
        "e4b": 896,
    }

    BBOX_KEY = "box_2d"
    COORD_DIVISOR = 1000.0
    BOX_FORMAT = "yxyx"
    # Official detection recipes use 560+ visual tokens; 70 misses cars/people.
    MAX_SOFT_TOKENS = 560

    _LICENSE_NOTICE = ""

    def __init__(self, size: str, **kwargs):
        _require_transformers()
        super().__init__(size, **kwargs)

    def _chat_template_kwargs(self) -> Dict[str, object]:
        return {"enable_thinking": False}

    def _format_detection_prompt(self, labels: str) -> str:
        # Official Gemma 4 detection ask: "detect person and cat, output only ```json"
        joined = labels.replace(", ", " and ")
        return f"detect {joined}, output only ```json"

    def _load_pretrained(self, snapshot_dir: str):
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        try:
            from transformers import AutoModelForMultimodalLM as ModelCls
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelCls

        model = ModelCls.from_pretrained(
            snapshot_dir,
            dtype=self._resolve_dtype(),
            trust_remote_code=self.TRUST_REMOTE_CODE,
        )
        processor = AutoProcessor.from_pretrained(
            snapshot_dir, trust_remote_code=self.TRUST_REMOTE_CODE
        )
        for obj in (processor, getattr(processor, "image_processor", None)):
            if obj is not None and hasattr(obj, "max_soft_tokens"):
                obj.max_soft_tokens = self.MAX_SOFT_TOKENS
        return model, processor
