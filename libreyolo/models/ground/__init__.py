"""LibreGround: instruction → click point.

Sibling factory to ``LibreVLM`` / ``LibreOpenVocab``. Returns the same
``Results.points`` payload as FOMO / LocateAnything.

    from libreyolo import LibreGround

    r = LibreGround()("screen.png", prompt="Bluetooth")
    r.points.xy
"""

from __future__ import annotations

from typing import Dict, Tuple, Type

from .base import GroundAPIMixin, LibreGroundModel
from .florence import LibreGroundFlorence2
from .qwen3vl import LibreGroundQwen3VL
from .showui import LibreShowUI

# Public aliases are only the families that load and return one click.
# TinyClick / Holo / UI-TARS / LocateAnything / Moondream stay out
# until they are load-tested and contract-compliant.
_ALIASES: Dict[str, Tuple[Type, str]] = {
    "showui": (LibreShowUI, "2b"),
    "show-ui": (LibreShowUI, "2b"),
    "showui-2b": (LibreShowUI, "2b"),
    "show-ui-2b": (LibreShowUI, "2b"),
    "florence-2": (LibreGroundFlorence2, "base"),
    "florence2": (LibreGroundFlorence2, "base"),
    "florence-2-base": (LibreGroundFlorence2, "base"),
    "florence-2-large": (LibreGroundFlorence2, "large"),
    "qwen3-vl": (LibreGroundQwen3VL, "2b"),
    "qwen3vl": (LibreGroundQwen3VL, "2b"),
    "qwen3-vl-2b": (LibreGroundQwen3VL, "2b"),
    "qwen3-vl-4b": (LibreGroundQwen3VL, "4b"),
    "qwen3-vl-8b": (LibreGroundQwen3VL, "8b"),
}

_UNVERIFIED_ALIASES: Dict[str, str] = {
    "holo": "Holo is not a factory alias until a size is load-tested.",
    "holo1.5": "Holo is not a factory alias until a size is load-tested.",
    "holo1.5-7b": "Holo is not a factory alias until a size is load-tested.",
    "holo-7b": "Holo is not a factory alias until a size is load-tested.",
    "holo1-7b": "Holo is not a factory alias until a size is load-tested.",
    "ui-tars": "UI-TARS is not a factory alias until a size is load-tested.",
    "uitars": "UI-TARS is not a factory alias until a size is load-tested.",
    "ui-tars-7b": "UI-TARS is not a factory alias until a size is load-tested.",
    "uitars-7b": "UI-TARS is not a factory alias until a size is load-tested.",
    "ui-tars-1.5": "UI-TARS is not a factory alias until a size is load-tested.",
    "ui-tars-1.5-7b": "UI-TARS is not a factory alias until a size is load-tested.",
    "tinyclick": "TinyClick is not a factory alias: the 2024 checkpoint does not load on transformers 5.",
    "tiny-click": "TinyClick is not a factory alias: the 2024 checkpoint does not load on transformers 5.",
    "moondream": "Moondream 2 is not a LibreGround alias: click-in-box testing showed center-biased output.",
    "moondream2": "Moondream 2 is not a LibreGround alias: click-in-box testing showed center-biased output.",
    "moondream-2": "Moondream 2 is not a LibreGround alias: click-in-box testing showed center-biased output.",
    "moondream3": "Moondream 3 is BSL 1.1 and is not a LibreGround factory alias.",
    "moondream-3": "Moondream 3 is BSL 1.1 and is not a LibreGround factory alias.",
    "locate-anything": "LocateAnything is NVIDIA non-commercial and is not a LibreGround factory alias.",
    "locateanything": "LocateAnything is NVIDIA non-commercial and is not a LibreGround factory alias.",
    "locate-anything-3b": "LocateAnything is NVIDIA non-commercial and is not a LibreGround factory alias.",
}

# The three snapshot-mirrored families hosted on the LibreYOLO org.
# alias -> (family class name, size, hf repo id)
HOSTED_SNAPSHOTS: Dict[str, Tuple[str, str, str]] = {
    "florence-2-base": (
        "LibreGroundFlorence2",
        "base",
        "LibreYOLO/LibreGroundFlorence2base",
    ),
    "showui-2b": ("LibreShowUI", "2b", "LibreYOLO/LibreShowUI2b"),
    "qwen3-vl-2b": ("LibreGroundQwen3VL", "2b", "LibreYOLO/LibreGroundQwen3VL2b"),
}

_DEFAULT_MODEL = "showui-2b"


def LibreGround(model: str = _DEFAULT_MODEL, **kwargs):
    """Load a grounding model by alias.

    Args:
        model: Alias such as ``"showui-2b"``, ``"florence-2-base"``,
            or ``"qwen3-vl-2b"``. Defaults to ShowUI-2B.
        **kwargs: Forwarded to the family constructor: ``device``,
            ``query`` / ``prompt`` (initial instruction), ``max_new_tokens``.

    Returns:
        A model with ``set_query`` / ``predict(..., prompt=)`` returning
        ``Results.points``.
    """
    key = str(model).strip().lower().replace("_", "-")
    blocked = _UNVERIFIED_ALIASES.get(key)
    if blocked is not None:
        raise ValueError(blocked)
    match = _ALIASES.get(key)
    if match is None:
        raise ValueError(
            f"Unknown grounding model {model!r}. Known aliases: "
            f"{', '.join(sorted(set(_ALIASES)))}."
        )
    family_cls, size = match
    return family_cls(size=size, **kwargs)


__all__ = [
    "LibreGround",
    "LibreGroundModel",
    "GroundAPIMixin",
    "HOSTED_SNAPSHOTS",
    "LibreShowUI",
    "LibreGroundFlorence2",
    "LibreGroundQwen3VL",
]
