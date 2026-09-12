"""LibreVLA: vision-language-action policies behind the LibreYOLO surface.

User-facing entry point is the ``LibreVLA(...)`` factory, a sibling to
``LibreYOLO(...)`` and ``LibreVLM(...)``. Frames plus the robot state plus an
instruction go in; an action chunk comes out as ``Results.actions``.

    from libreyolo import LibreVLA

    model = LibreVLA()                                   # smolvla-base, autodownloads
    model.set_instruction("pick up the red cube")
    result = model.predict(frame, state=q)               # Results.actions is (T, D)
    result.actions.first                                 # the next action

    results = model.train(data="lerobot/svla_so101_pickplace", epochs=5)
    model = LibreVLA(results["best"])

See ``docs/librevla.md`` and ``docs/adr/0028-librevla-contract.md``.
"""

from __future__ import annotations

from typing import Dict, Tuple, Type

from .act_policy import LibreACT
from .base import LibreVLAModel
from .checkpoint import CONTRACT_FILENAME, is_vla_checkpoint, read_contract
from .smolvla import LibreSmolVLA
from .diffusion_policy import LibreDiffusionPolicy
from .xvla import LibreXVLA

# alias -> (family class, size)
_ALIASES: Dict[str, Tuple[Type[LibreVLAModel], str]] = {
    "xvla": (LibreXVLA, "base"),
    "xvla-base": (LibreXVLA, "base"),
    "diffusion": (LibreDiffusionPolicy, "base"),
    "diffusion-policy": (LibreDiffusionPolicy, "base"),
    "act": (LibreACT, "base"),
    "act-policy": (LibreACT, "base"),
    "smolvla": (LibreSmolVLA, "base"),
    "smolvla-base": (LibreSmolVLA, "base"),
}

# Families lerobot ships that are not factory aliases until an adapter is
# load-tested against the real checkpoint (ADR 0028 "Families").
_RESERVED_ALIASES: Dict[str, str] = {
    "pi0": "pi0 is reserved: the adapter lands once it is load-tested. Its PaliGemma base is under Gemma terms.",
    "pi0-base": "pi0 is reserved: the adapter lands once it is load-tested. Its PaliGemma base is under Gemma terms.",
    "pi05": "pi0.5 is reserved: the adapter lands once it is load-tested.",
    "pi0.5": "pi0.5 is reserved: the adapter lands once it is load-tested.",
    "molmoact2": "MolmoAct2 is reserved: the adapter lands once it is load-tested.",
    "molmoact": "MolmoAct is reserved: the adapter lands once it is load-tested.",
    "groot": "GR00T is reserved: the adapter lands once it is load-tested.",
    "groot-n1.5": "GR00T is reserved: the adapter lands once it is load-tested.",
    "openvla": "OpenVLA is reserved: it loads through transformers remote code, not lerobot; adapter later.",
}

_DEFAULT_MODEL = "smolvla-base"


def _load_checkpoint(path, **kwargs) -> LibreVLAModel:
    """Load a fine-tune checkpoint directory produced by ``train()``."""
    contract = read_contract(path)
    family_classes = {cls.FAMILY: cls for cls, _size in _ALIASES.values()}
    family_cls = family_classes.get(contract["family"])
    if family_cls is None:
        raise ValueError(
            f"VLA checkpoint {path} was trained on unknown family "
            f"{contract['family']!r}; this libreyolo build knows "
            f"{sorted(family_classes)}."
        )
    return family_cls(size=contract["size"], checkpoint_dir=str(path), **kwargs)


def LibreVLA(model: str = _DEFAULT_MODEL, **kwargs) -> LibreVLAModel:
    """Load a vision-language-action policy by alias or checkpoint path.

    Args:
        model: Alias (``"smolvla-base"``) or a path to a checkpoint directory
            produced by ``train()`` (it carries ``libreyolo_vla.json``).
        **kwargs: Forwarded to the family constructor: ``device``,
            ``instruction`` (sticky task text), ``cameras`` (user camera
            names in slot order).

    Returns:
        A ``LibreVLAModel`` with ``predict`` / ``train`` / ``val``.
    """
    if is_vla_checkpoint(model):
        return _load_checkpoint(model, **kwargs)
    key = str(model).strip().lower().replace("_", "-")
    reserved = _RESERVED_ALIASES.get(key)
    if reserved is not None:
        raise ValueError(reserved)
    match = _ALIASES.get(key)
    if match is None:
        raise ValueError(
            f"Unknown VLA model {model!r}. Known aliases: "
            f"{', '.join(sorted(set(_ALIASES)))}."
        )
    family_cls, size = match
    return family_cls(size=size, **kwargs)


__all__ = [
    "CONTRACT_FILENAME",
    "LibreVLA",
    "LibreVLAModel",
    "LibreSmolVLA",
    "LibreXVLA",
    "LibreACT",
    "LibreDiffusionPolicy",
    "is_vla_checkpoint",
    "read_contract",
]
