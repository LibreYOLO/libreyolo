"""LibreGen: generative models behind the LibreYOLO call surface.

User-facing entry point is the ``LibreGen(...)`` factory, a sibling to
``LibreYOLO(...)``, ``LibreVLM(...)`` and ``LibreVLA(...)``. Text goes in and
an image comes out as ``Results.generated``, or an image goes in and a 3D
object comes out as ``Results.gaussians``.

    from libreyolo import LibreGen

    model = LibreGen("flux2-klein-4b")          # autodownloads
    result = model("a red fox sleeping in fresh snow")
    result.save("fox.png")

    model = LibreGen("speridlabs-idol")
    model("chair.jpg").save("chair.ply")

See ``docs/libregen.md`` and ADRs 0030 and 0031 in ``docs/adr/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Type

from .base import LibreGenModel
from .base3d import LibreGen3DModel
from .diffusers_family import DiffusersGenFamily
from .flux2_klein import LibreFlux2Klein
from .lgm import LibreLGM
from .prx import LibrePRX, LibrePRXPixel
from .sana_sprint import LibreSanaSprint
from .speridlabs_idol import LibreSperidlabsIdol
from .trellis import LibreTRELLIS
from .triposplat import LibreTripoSplat

# alias -> (family class, size)
_ALIASES: Dict[str, Tuple[Type[LibreGenModel], str]] = {
    "flux2-klein": (LibreFlux2Klein, "4b"),
    "flux2-klein-4b": (LibreFlux2Klein, "4b"),
    "flux.2-klein": (LibreFlux2Klein, "4b"),
    "flux.2-klein-4b": (LibreFlux2Klein, "4b"),
    "sana-sprint": (LibreSanaSprint, "0.6b"),
    "sana-sprint-0.6b": (LibreSanaSprint, "0.6b"),
    "prx": (LibrePRX, "1024-beta"),
    **{f"prx-{size}": (LibrePRX, size) for size in LibrePRX.HF_REPOS},
    "prxpixel": (LibrePRXPixel, "7b"),
    "prxpixel-7b": (LibrePRXPixel, "7b"),
    "prx-pixel": (LibrePRXPixel, "7b"),
    "speridlabs-idol": (LibreSperidlabsIdol, "0.6b"),
    "speridlabs-idol-0.6b": (LibreSperidlabsIdol, "0.6b"),
    "triposplat": (LibreTripoSplat, "0.6b"),
    "triposplat-0.6b": (LibreTripoSplat, "0.6b"),
    "trellis": (LibreTRELLIS, "l"),
    "trellis-image-large": (LibreTRELLIS, "l"),
    "lgm": (LibreLGM, "0.4b"),
    "lgm-0.4b": (LibreLGM, "0.4b"),
}

_DEFAULT_MODEL = "flux2-klein-4b"


def _families() -> Tuple[Type[LibreGenModel], ...]:
    return tuple(dict.fromkeys(cls for cls, _size in _ALIASES.values()))


def _is_directory(model) -> bool:
    try:
        return Path(model).is_dir()
    except (OSError, TypeError, ValueError):
        return False


def LibreGen(model: str = _DEFAULT_MODEL, **kwargs) -> LibreGenModel:
    """Load a generative model by alias or local model directory.

    Args:
        model: Alias (``"flux2-klein-4b"``, ``"sana-sprint-0.6b"``,
            ``"prx-1024-beta"`` and the other ``prx-*`` checkpoints,
            ``"prxpixel-7b"``, ``"speridlabs-idol"``, ``"triposplat"``, ``"trellis"``) or a path to a local model directory in a
            supported family's upstream format, such as a fine-tune or an
            offline copy.
        **kwargs: Forwarded to the family constructor: ``device``, ``dtype``
            (``"auto"``, ``"bf16"``, ``"fp16"``, ``"fp32"``; image-to-3D
            families take ``"auto"`` only), ``offload`` (text-to-image: keep
            idle components off the GPU), ``compile`` (image-to-3D:
            ``torch.compile`` the sampler on CUDA) and, for a directory,
            ``size``.

    Returns:
        A ``LibreGenModel``; call it with a prompt (text-to-image) or an
        image (image-to-3D).
    """
    # Aliases win, so a folder that happens to share an alias name cannot
    # shadow it; pass "./name" to load such a folder.
    key = str(model).strip().lower().replace("_", "-")
    match = _ALIASES.get(key)
    if match is None:
        if _is_directory(model):
            for family_cls in _families():
                if family_cls.matches_directory(model):
                    return family_cls(checkpoint_dir=str(model), **kwargs)
            raise ValueError(
                f"{model} is not a model directory of a known LibreGen family "
                f"({', '.join(cls.FAMILY for cls in _families())})."
            )
        raise ValueError(
            f"Unknown LibreGen model {model!r}. Known aliases: "
            f"{', '.join(sorted(_ALIASES))}."
        )
    family_cls, size = match
    return family_cls(size=size, **kwargs)


__all__ = [
    "LibreGen",
    "LibreGenModel",
    "LibreGen3DModel",
    "DiffusersGenFamily",
    "LibreFlux2Klein",
    "LibreSanaSprint",
    "LibrePRX",
    "LibrePRXPixel",
    "LibreSperidlabsIdol",
    "LibreTripoSplat",
    "LibreTRELLIS",
    "LibreLGM",
]
