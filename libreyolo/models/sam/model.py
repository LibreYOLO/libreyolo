"""LibreSAM: promptable segmentation models behind a familiar surface.

User-facing entry point is the ``LibreSAM(...)`` factory, a sibling to the
``LibreYOLO(...)`` and ``LibreVLM(...)`` factories. It returns a promptable
segmenter with an interactive, encode-once / prompt-many surface:

    from libreyolo import LibreSAM

    model = LibreSAM("base")                 # autodownloads (Apache-2.0)
    r = model.predict("image.jpg", points=[900, 370], labels=[1])  # point -> mask
    r = model.predict("image.jpg", bboxes=[100, 100, 200, 200])    # box   -> mask
    r = model.predict("image.jpg")                                  # segment everything
    r.masks.xy          # polygons
    r.boxes.xyxy        # tight boxes derived from masks

    model.set_image("image.jpg")             # encode once...
    a = model.predict(points=[500, 375], labels=[1])   # ...prompt cheaply
    b = model.predict(bboxes=[100, 100, 200, 200])
    model.reset_image()

The default family is SAM-1, whose code and weights are Apache-2.0, loaded
through the permissive ``transformers`` model API. See
``docs/adr/0007-libresam-contract.md``.
"""

from __future__ import annotations

from typing import Dict, Tuple, Type

from .base import LibreSAMModel
from .edgetam import LibreEdgeTAM
from .sam2 import LibreSAM2
from .sam3 import LibreSAM3


class LibreSAM1(LibreSAMModel):
    """SAM-1 promptable segmenter (ViT-B/L/H image encoder)."""

    FAMILY = "sam"
    FILENAME_PREFIX = "LibreSAM"
    # SAM-1's weights are Apache-2.0, which permits redistribution, so these are
    # mirrored on the LibreYOLO org rather than fetched from Meta. The mirrors
    # carry Meta's weights and config byte for byte, plus the licence and a
    # NOTICE; upstream's duplicate pytorch_model.bin and tf_model.h5 are not
    # mirrored because the loader reads the safetensors file.
    HF_REPOS = {
        "base": "LibreYOLO/LibreSAMbase",
        "large": "LibreYOLO/LibreSAMlarge",
        "huge": "LibreYOLO/LibreSAMhuge",
    }
    INPUT_SIZES = {"base": 1024, "large": 1024, "huge": 1024}


# alias -> (family class, size)
_MOBILE_SAM = "mobilesam"
_PICO_SAM3 = "picosam3"

_ALIASES: Dict[str, Tuple[Type[LibreSAMModel] | str, str]] = {
    "base": (LibreSAM1, "base"),
    "large": (LibreSAM1, "large"),
    "huge": (LibreSAM1, "huge"),
    "b": (LibreSAM1, "base"),
    "l": (LibreSAM1, "large"),
    "h": (LibreSAM1, "huge"),
    "sam-base": (LibreSAM1, "base"),
    "sam-large": (LibreSAM1, "large"),
    "sam-huge": (LibreSAM1, "huge"),
    "sam_b": (LibreSAM1, "base"),
    "sam_l": (LibreSAM1, "large"),
    "sam_h": (LibreSAM1, "huge"),
    "sam2-tiny": (LibreSAM2, "tiny"),
    "sam2-small": (LibreSAM2, "small"),
    "sam2-base-plus": (LibreSAM2, "base-plus"),
    "sam2-baseplus": (LibreSAM2, "base-plus"),
    "sam2-large": (LibreSAM2, "large"),
    "sam2-t": (LibreSAM2, "tiny"),
    "sam2-s": (LibreSAM2, "small"),
    "sam2-bp": (LibreSAM2, "base-plus"),
    "sam2-l": (LibreSAM2, "large"),
    "sam2_t": (LibreSAM2, "tiny"),
    "sam2_s": (LibreSAM2, "small"),
    "sam2_bp": (LibreSAM2, "base-plus"),
    "sam2_l": (LibreSAM2, "large"),
    "edgetam": (LibreEdgeTAM, "edge"),
    "edge-tam": (LibreEdgeTAM, "edge"),
    "edgetam-edge": (LibreEdgeTAM, "edge"),
    "sam3": (LibreSAM3, "large"),
    "sam-3": (LibreSAM3, "large"),
    "sam3-large": (LibreSAM3, "large"),
    "mobilesam": (_MOBILE_SAM, "tiny"),
    "mobilesam-tiny": (_MOBILE_SAM, "tiny"),
    "mobilesam_t": (_MOBILE_SAM, "tiny"),
    "mobile-sam": (_MOBILE_SAM, "tiny"),
    "mobile-sam-tiny": (_MOBILE_SAM, "tiny"),
    "picosam3": (_PICO_SAM3, "pico"),
    "picosam3-pico": (_PICO_SAM3, "pico"),
    "picosam3_pico": (_PICO_SAM3, "pico"),
    "pico-sam3": (_PICO_SAM3, "pico"),
}

_DEFAULT_MODEL = "base"


def LibreSAM(model: str = _DEFAULT_MODEL, **kwargs) -> LibreSAMModel:
    """Load a promptable segmentation model by name.

    Args:
        model: Size alias — ``"base"`` (default), ``"large"``, or ``"huge"``
            (also ``"sam_b"``/``"sam_l"``/``"sam_h"``, or ``"b"``/``"l"``/``"h"``).
            SAM-2 aliases use an explicit prefix, for example
            ``"sam2-tiny"`` / ``"sam2_t"``.
            EdgeTAM uses ``"edgetam"`` / ``"edge-tam"``.
            SAM 3 uses ``"sam3"`` / ``"sam-3"`` / ``"sam3-large"``.
            MobileSAM aliases resolve to its single ``"tiny"`` size.
        **kwargs: Forwarded to the family constructor: ``device``, and
            ``multimask`` (when True, ``predict`` returns all of SAM's ambiguity
            masks per prompt — 3 whole-vs-part masks — instead of the single
            best one; can also be set per-call on ``predict``).

    Returns:
        A ``LibreSAMModel`` with the interactive ``set_image``/``predict`` surface.
    """
    key = str(model).strip().lower()
    match = _ALIASES.get(key)
    if match is None:
        raise ValueError(
            f"Unknown SAM model {model!r}. Known aliases: "
            f"{', '.join(sorted(_ALIASES))}."
        )
    family_cls, size = match
    if family_cls == _MOBILE_SAM:
        from ..mobilesam import LibreMobileSAM

        family_cls = LibreMobileSAM
    elif family_cls == _PICO_SAM3:
        from ..picosam3 import LibrePicoSAM3

        family_cls = LibrePicoSAM3
    elif isinstance(family_cls, str):
        raise RuntimeError(f"Invalid lazy SAM family target: {family_cls!r}")
    return family_cls(size=size, **kwargs)


__all__ = ["LibreSAM", "LibreSAM1", "LibreSAM2", "LibreEdgeTAM", "LibreSAM3"]
