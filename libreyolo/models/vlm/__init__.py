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
from .florence2 import LibreFlorence2
from .gemma4 import LibreGemma4
from .internvl3 import LibreInternVL3
from .kosmos2 import LibreKosmos2
from .lfm2 import LibreLFM2VL
from .locateanything import LibreLocateAnything
from .molmo2 import LibreMolmo2
from .moondream import LibreMoondream
from .northmicro import LibreNorthMicroVision
from .qwen3vl import LibreQwen3VL
from .smolvlm import LibreSmolVLM2

# alias -> (family class, size)
_ALIASES: Dict[str, Tuple[Type[LibreVLMModel], str]] = {
    "qwen3-vl": (LibreQwen3VL, "4b"),
    "qwen3-vl-2b": (LibreQwen3VL, "2b"),
    "qwen3-vl-4b": (LibreQwen3VL, "4b"),
    "qwen3-vl-8b": (LibreQwen3VL, "8b"),
    "lfm2-vl": (LibreLFM2VL, "450m"),
    "lfm2-vl-450m": (LibreLFM2VL, "450m"),
    "lfm2-vl-1.6b": (LibreLFM2VL, "1.6b"),
    "lfm2-vl-3b": (LibreLFM2VL, "3b"),
    "north-micro-vision": (LibreNorthMicroVision, "2.4b"),
    "north-micro-vision-2.4b": (LibreNorthMicroVision, "2.4b"),
    "northmicrovision": (LibreNorthMicroVision, "2.4b"),
    "internvl3": (LibreInternVL3, "2b"),
    "internvl3-1b": (LibreInternVL3, "1b"),
    "internvl3-2b": (LibreInternVL3, "2b"),
    "internvl3-8b": (LibreInternVL3, "8b"),
    "smolvlm2": (LibreSmolVLM2, "2.2b"),
    "smolvlm2-2.2b": (LibreSmolVLM2, "2.2b"),
    "smolvlm2-500m": (LibreSmolVLM2, "500m"),
    "florence-2": (LibreFlorence2, "base"),
    "florence2": (LibreFlorence2, "base"),
    "florence-2-base": (LibreFlorence2, "base"),
    "florence-2-large": (LibreFlorence2, "large"),
    "kosmos-2": (LibreKosmos2, "224"),
    "kosmos2": (LibreKosmos2, "224"),
    "locate-anything": (LibreLocateAnything, "3b"),
    "locateanything": (LibreLocateAnything, "3b"),
    "locate-anything-3b": (LibreLocateAnything, "3b"),
    "locateanything-3b": (LibreLocateAnything, "3b"),
    "gemma-4": (LibreGemma4, "e4b"),
    "gemma4": (LibreGemma4, "e4b"),
    "gemma-4-e4b": (LibreGemma4, "e4b"),
    "gemma4-e4b": (LibreGemma4, "e4b"),
    "gemma-4-e2b": (LibreGemma4, "e2b"),
    "gemma4-e2b": (LibreGemma4, "e2b"),
    "molmo2": (LibreMolmo2, "4b"),
    "molmo2-4b": (LibreMolmo2, "4b"),
    "molmo2-8b": (LibreMolmo2, "8b"),
    "molmo2-o-7b": (LibreMolmo2, "o-7b"),
    "moondream": (LibreMoondream, "2"),
    "moondream-2": (LibreMoondream, "2"),
    "moondream2": (LibreMoondream, "2"),
    "moondream-3": (LibreMoondream, "3"),
    "moondream3": (LibreMoondream, "3"),
}

# SenseNova-Vision lives in ``models/sensenova`` (it vendors its own
# architecture) and imports this package's base class, so its aliases resolve
# lazily to avoid a circular import at package-init time.
_LAZY_ALIASES: Dict[str, str] = {
    "sensenova-vision": "7b",
    "sensenova-vision-7b": "7b",
    "sensenovavision": "7b",
}
_MODUS_ALIASES: Dict[str, str] = {
    "libremodus": "14b-a7b",
    "libremodus-14b-a7b": "14b-a7b",
    "modus": "14b-a7b",
    "modus-14b-a7b": "14b-a7b",
}

_DEFAULT_MODEL = "qwen3-vl-4b"


def _load_checkpoint(path, **kwargs) -> LibreVLMModel:
    """Load a fine-tune checkpoint directory produced by ``train()``."""
    from .training.checkpoint import read_contract

    contract = read_contract(path)
    family_classes = {cls.FAMILY: cls for cls, _size in _ALIASES.values()}
    family_cls = family_classes.get(contract["family"])
    if family_cls is None:
        raise ValueError(
            f"VLM checkpoint {path} was trained on unknown family "
            f"{contract['family']!r}; this libreyolo build knows "
            f"{sorted(family_classes)}."
        )
    kwargs.setdefault("names", list(contract["names"]))
    return family_cls(size=contract["size"], checkpoint_dir=str(path), **kwargs)


def LibreVLM(model: str = _DEFAULT_MODEL, **kwargs) -> LibreVLMModel:
    """Load a vision-language detector by name or fine-tune checkpoint path.

    Args:
        model: Model alias (e.g. ``"qwen3-vl-4b"``, ``"lfm2-vl-450m"``), or a
            path to a fine-tune checkpoint directory produced by ``train()``
            (it carries ``libreyolo_vlm.json``). Defaults to Qwen3-VL-4B
            (Apache-2.0).
        **kwargs: Forwarded to the family constructor: ``device``, ``names``
            (initial class vocabulary, same as calling ``set_classes`` after
            load), ``prompt`` (override the detection prompt), ``max_new_tokens``.
            When loading a checkpoint, ``names`` defaults to the vocabulary the
            fine-tune was trained on.

    Returns:
        A ``LibreVLMModel`` instance with the standard predict/track surface.
    """
    from .training.checkpoint import is_vlm_checkpoint

    if is_vlm_checkpoint(model):
        return _load_checkpoint(model, **kwargs)
    key = str(model).strip().lower()
    if key in _LAZY_ALIASES:
        from ..sensenova import LibreSenseNovaVision

        return LibreSenseNovaVision(size=_LAZY_ALIASES[key], **kwargs)
    if key in _MODUS_ALIASES:
        from ..modus import LibreMODUS

        return LibreMODUS(size=_MODUS_ALIASES[key], **kwargs)
    match = _ALIASES.get(key)
    if match is None:
        raise ValueError(
            f"Unknown VLM model {model!r}. Known aliases: "
            f"{', '.join(sorted(set(_ALIASES) | set(_LAZY_ALIASES) | set(_MODUS_ALIASES)))}."
        )
    family_cls, size = match
    return family_cls(size=size, **kwargs)


def __getattr__(name: str):
    if name == "LibreSenseNovaVision":
        from ..sensenova import LibreSenseNovaVision

        return LibreSenseNovaVision
    if name in {"LibreMODUS", "LibreModus"}:
        from ..modus import LibreMODUS

        return LibreMODUS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LibreVLM",
    "LibreVLMModel",
    "LibreLFM2VL",
    "LibreQwen3VL",
    "LibreSmolVLM2",
    "LibreInternVL3",
    "LibreFlorence2",
    "LibreKosmos2",
    "LibreLocateAnything",
    "LibreGemma4",
    "LibreMoondream",
    "LibreMolmo2",
    "LibreNorthMicroVision",
    "LibreSenseNovaVision",
    "LibreMODUS",
    "LibreModus",
]
