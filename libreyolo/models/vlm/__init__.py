"""LibreVLM: vision-language models used as open-vocabulary object detectors.

User-facing entry point is the ``LibreVLM(...)`` factory, a sibling to the
``LibreYOLO(...)`` factory. It returns a model instance that behaves like any
YOLO model (predict on image/folder/video, track, identical ``Results``), but
is backed by a generative VLM, so the class list is open vocabulary.

    from libreyolo import LibreVLM
    model = LibreVLM()                       # defaults to Qwen3-VL-4B, autodownloads
    model.set_classes(["pink car", "wheel"]) # open vocabulary: any words
    results = model.predict("image.jpg")     # same Results as a YOLO model
    results = model.predict("folder/")       # folders, video, track() all work
    text = model.chat("image.jpg", "How many cars are pink?")  # raw escape hatch

See ``docs/librevlm_design.md`` and ``docs/adr/0002-librevlm-contract.md``.
"""

from __future__ import annotations

from typing import Dict, Tuple, Type

from .base import LibreVLMModel
from .lfm2 import LibreLFM2VL
from .qwen3vl import LibreQwen3VL

# alias -> (family class, size)
_ALIASES: Dict[str, Tuple[Type[LibreVLMModel], str]] = {
    "qwen3-vl": (LibreQwen3VL, "4b"),
    "qwen3-vl-2b": (LibreQwen3VL, "2b"),
    "qwen3-vl-4b": (LibreQwen3VL, "4b"),
    "qwen3-vl-8b": (LibreQwen3VL, "8b"),
    "lfm2-vl": (LibreLFM2VL, "450m"),
    "lfm2-vl-450m": (LibreLFM2VL, "450m"),
    "lfm2-vl-1.6b": (LibreLFM2VL, "1.6b"),
}

_DEFAULT_MODEL = "qwen3-vl-4b"


def LibreVLM(model: str = _DEFAULT_MODEL, **kwargs) -> LibreVLMModel:
    """Load a vision-language detector by name.

    Args:
        model: Model alias (e.g. ``"qwen3-vl-4b"``, ``"lfm2-vl-450m"``).
            Defaults to Qwen3-VL-4B (Apache-2.0).
        **kwargs: Forwarded to the family constructor: ``device``, ``names``
            (initial class vocabulary, same as calling ``set_classes`` after
            load), ``prompt`` (override the detection prompt), ``max_new_tokens``.

    Returns:
        A ``LibreVLMModel`` instance with the standard predict/track surface.
    """
    key = str(model).strip().lower()
    match = _ALIASES.get(key)
    if match is None:
        raise ValueError(
            f"Unknown VLM model {model!r}. Known aliases: "
            f"{', '.join(sorted(_ALIASES))}."
        )
    family_cls, size = match
    return family_cls(size=size, **kwargs)


__all__ = ["LibreVLM", "LibreVLMModel", "LibreLFM2VL", "LibreQwen3VL"]
