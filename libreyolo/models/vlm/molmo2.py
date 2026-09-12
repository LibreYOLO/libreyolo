"""Molmo2 single-image pointing through pinned Apache-2.0 HF remote code.

One generation per vocabulary label preserves class identity even when the
model paraphrases its response. Coordinates are returned on the original
canvas, with synthetic confidence 1.0. Detection is not verified or exposed.
Install ``libreyolo[molmo2]`` in a separate environment from the other VLMs:
the pinned upstream code requires Transformers 4.57.1. See NOTICE.
"""

from __future__ import annotations

from typing import ClassVar

from ...utils.image_loader import ImageLoader
from .base import LibreVLMModel
from .locateanything import build_point_dict
from .moondream import _ImageCarrier
from .parsing import extract_molmo_points

_INSTALL_HINT = (
    "Molmo2's pinned remote code requires transformers==4.57.1 and einops. "
    "Install in a separate environment with: pip install 'libreyolo[molmo2]'"
)


def _check_dependencies():
    try:
        import einops  # noqa: F401
        import transformers
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    if transformers.__version__ != "4.57.1":
        raise ImportError(_INSTALL_HINT)


class LibreMolmo2(LibreVLMModel):
    """Molmo2 pointer; ``names``/``set_classes`` define the objects to locate.

    ``prompt`` overrides the pointing request for each label and must contain
    ``{label}`` so each generation remains tied to its vocabulary entry.
    ``chat`` accepts arbitrary prompts and returns unparsed text.
    """

    FAMILY = "molmo2"
    FILENAME_PREFIX = "LibreMolmo2"
    HF_REPOS: ClassVar[dict[str, str]] = {
        "4b": "LibreYOLO/LibreMolmo24b",
        "8b": "LibreYOLO/LibreMolmo28b",
        "o-7b": "LibreYOLO/LibreMolmo2o-7b",
    }
    HF_REVISIONS: ClassVar[dict[str, str]] = {
        "4b": "e318fb1b9f461de558f30799671e8a166df9a562",
        "8b": "e8879879aed29ca448b7a4a14e39d6a831bcb469",
        "o-7b": "047d61ee534ed3a808a0d5666b2006e62bd5a7dd",
    }
    INPUT_SIZES: ClassVar[dict[str, int]] = {"4b": 378, "8b": 378, "o-7b": 378}
    SUPPORTED_TASKS = ("point",)
    DEFAULT_TASK = "point"
    TRUST_REMOTE_CODE = True
    # Molmo2 uses token_type_ids for bidirectional image attention.
    UNSUPPORTED_GENERATE_INPUTS = ()
    TRAINABLE = False
    TRAIN_UNSUPPORTED_REASON = "Molmo2 pointing fine-tuning is not implemented."
    COORD_DIVISOR = 1.0  # Parser normalizes both output grammars.
    REPETITION_PENALTY = 1.0  # Preserve upstream's greedy generation defaults.

    def __init__(self, size="4b", **kwargs):
        prompt = kwargs.get("prompt")
        if prompt is not None and "{label}" not in prompt:
            raise ValueError(
                "Molmo2 prompt must contain {label} for per-class pointing."
            )
        super().__init__(size=size, **kwargs)

    def _ensure_weights(self):
        _check_dependencies()  # Fail before downloading multi-GB weights.
        return super()._ensure_weights()

    def _load_pretrained(self, snapshot_dir):
        _check_dependencies()
        return super()._load_pretrained(snapshot_dir)

    def _preprocess(self, image, color_format="auto", input_size=None):
        img = ImageLoader.load(image, color_format=color_format)
        return _ImageCarrier(img), img, img.size, 1.0

    def _forward(self, inputs):
        rows = []
        for label in self.names.values():
            prompt = (
                self._custom_prompt.replace("{label}", label)
                if self._custom_prompt is not None
                else f"Point to the {label}."
            )
            rows.extend(extract_molmo_points(self.chat(inputs.img, prompt), label))
        return rows

    def _postprocess(
        self,
        output,
        conf_thres,
        iou_thres,
        original_size,
        max_det=300,
        ratio=1.0,
        **kwargs,
    ):
        return build_point_dict(
            output,
            self._name_to_id,
            original_size,
            conf_thres=conf_thres,
            max_det=max_det,
            classes=kwargs.get("classes"),
            default_score=self._score_detections(output),
            coord_divisor=self.COORD_DIVISOR,
        )
