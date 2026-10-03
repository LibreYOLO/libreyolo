"""Convert upstream YOLO9 weights to LibreYOLO key layout.

The upstream YOLO9 release (MultimediaTechLab/YOLO, MIT) ships plain
``state_dict`` checkpoints that use numbered layer indices (``0.``, ``1.``,
``2.`` …) while LibreYOLO uses semantic module names (``backbone.conv0``,
``neck.elan_up1`` …). This module owns the index remapping so both the
offline ``weights/convert_yolo9_weights.py`` script and the runtime
auto-conversion path in :mod:`libreyolo.models.autoconvert` share one
implementation. LibreYOLO's blocks keep the upstream sublayer names
(``conv1``, ``bottleneck``, ``anchor_conv`` …), so only the layer prefixes
and the detection-head layout change.

The conversion is structural only — it renames keys, keeps the PGI
auxiliary-branch weights for training (yolo9-t/s: layers 23/26/29/30,
yolo9-m/c: layers 23-38, both → ``aux.*`` / ``aux_head.*``), and drops the
``anc2vec`` weights that LibreYOLO derives internally. Class count is taken from the upstream detection head, so
fine-tuned checkpoints with a non-COCO ``nc`` convert correctly.

It also owns :func:`upgrade_legacy_key` / :func:`upgrade_legacy_state_dict`,
which rename the key layout of LibreYOLO checkpoints written before the
yolo9 blocks switched to the upstream sublayer names.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import torch

# =============================================================================
# Layer Index Mapping (YOLO layer index -> LibreYOLO prefix)
# =============================================================================

# Common layers across all variants
COMMON_LAYERS = {
    0: "backbone.conv0",  # Conv 3->X
    1: "backbone.conv1",  # Conv X->Y
}

# PGI auxiliary branches (MultimediaTechLab ``auxiliary`` sections).
# Training-only; inference never consumes these modules. Old LibreYOLO
# conversions dropped them; keeping them is additive. The two size groups
# have different upstream topologies and therefore different maps.

# v9-t/v9-s: top-down branch, LibreYOLO ``AuxNeck``. Layers 24/25 and 27/28
# are parameter-free UpSample/Concat.
YOLO9_AUX_LAYER_MAP = {
    23: "aux.spp",  # SPPELAN on B5 → A5
    26: "aux.elan_a4",  # RepNCSPELAN after upsample+concat B4
    29: "aux.elan_a3",  # RepNCSPELAN after upsample+concat B3
    30: "aux_head",  # MultiheadDetection on [A3, A4, A5]
}

# v9-m/v9-c: CBLinear taps plus a second backbone, LibreYOLO ``AuxBackbone``.
# Layers 30/33/36 are the parameter-free CBFuse sums.
YOLO9_MC_AUX_LAYER_MAP = {
    23: "aux.cblinear3",  # CBLinear on B3 (R3)
    24: "aux.cblinear4",  # CBLinear on B4 (R4)
    25: "aux.cblinear5",  # CBLinear on B5 (R5)
    26: "aux.conv0",  # Conv 3->X on the image
    27: "aux.conv1",  # Conv X->Y
    28: "aux.elan1",  # RepNCSPELAN
    29: "aux.down2",  # AConv (m) / ADown (c)
    31: "aux.elan2",  # RepNCSPELAN after CBFuse (A3)
    32: "aux.down3",  # AConv / ADown
    34: "aux.elan3",  # RepNCSPELAN after CBFuse (A4)
    35: "aux.down4",  # AConv / ADown
    37: "aux.elan4",  # RepNCSPELAN after CBFuse (A5)
    38: "aux_head",  # MultiheadDetection on [A3, A4, A5]
}

AUX_LAYER_MAPS = {
    "t": YOLO9_AUX_LAYER_MAP,
    "s": YOLO9_AUX_LAYER_MAP,
    "m": YOLO9_MC_AUX_LAYER_MAP,
    "c": YOLO9_MC_AUX_LAYER_MAP,
}

# yolo9-t and yolo9-s: ELAN first block, AConv downsampling
YOLO9_TS_LAYER_MAP = {
    **COMMON_LAYERS,
    2: "backbone.elan1",  # ELAN
    3: "backbone.down2",  # AConv
    4: "backbone.elan2",  # RepNCSPELAN
    5: "backbone.down3",  # AConv
    6: "backbone.elan3",  # RepNCSPELAN
    7: "backbone.down4",  # AConv
    8: "backbone.elan4",  # RepNCSPELAN
    9: "backbone.spp",  # SPPELAN
    # Neck
    12: "neck.elan_up1",  # RepNCSPELAN (N4)
    15: "neck.elan_up2",  # RepNCSPELAN (P3)
    16: "neck.down1",  # AConv
    18: "neck.elan_down1",  # RepNCSPELAN (P4)
    19: "neck.down2",  # AConv
    21: "neck.elan_down2",  # RepNCSPELAN (P5)
    # Detection head
    22: "head",  # MultiheadDetection
    **YOLO9_AUX_LAYER_MAP,
}

# yolo9-m: RepNCSPELAN first block, AConv downsampling
YOLO9_M_LAYER_MAP = {
    **COMMON_LAYERS,
    2: "backbone.elan1",  # RepNCSPELAN
    3: "backbone.down2",  # AConv
    4: "backbone.elan2",  # RepNCSPELAN
    5: "backbone.down3",  # AConv
    6: "backbone.elan3",  # RepNCSPELAN
    7: "backbone.down4",  # AConv
    8: "backbone.elan4",  # RepNCSPELAN
    9: "backbone.spp",  # SPPELAN
    # Neck
    12: "neck.elan_up1",  # RepNCSPELAN (N4)
    15: "neck.elan_up2",  # RepNCSPELAN (P3)
    16: "neck.down1",  # AConv
    18: "neck.elan_down1",  # RepNCSPELAN (P4)
    19: "neck.down2",  # AConv
    21: "neck.elan_down2",  # RepNCSPELAN (P5)
    # Detection head
    22: "head",  # MultiheadDetection
    **YOLO9_MC_AUX_LAYER_MAP,
}

# yolo9-c: RepNCSPELAN first block, ADown downsampling
YOLO9_C_LAYER_MAP = {
    **COMMON_LAYERS,
    2: "backbone.elan1",  # RepNCSPELAN
    3: "backbone.down2",  # ADown
    4: "backbone.elan2",  # RepNCSPELAN
    5: "backbone.down3",  # ADown
    6: "backbone.elan3",  # RepNCSPELAN
    7: "backbone.down4",  # ADown
    8: "backbone.elan4",  # RepNCSPELAN
    9: "backbone.spp",  # SPPELAN
    # Neck
    12: "neck.elan_up1",  # RepNCSPELAN (N4)
    15: "neck.elan_up2",  # RepNCSPELAN (P3)
    16: "neck.down1",  # ADown
    18: "neck.elan_down1",  # RepNCSPELAN (P4)
    19: "neck.down2",  # ADown
    21: "neck.elan_down2",  # RepNCSPELAN (P5)
    # Detection head
    22: "head",  # MultiheadDetection
    **YOLO9_MC_AUX_LAYER_MAP,
}

LAYER_MAPS = {
    "t": YOLO9_TS_LAYER_MAP,
    "s": YOLO9_TS_LAYER_MAP,
    "m": YOLO9_M_LAYER_MAP,
    "c": YOLO9_C_LAYER_MAP,
}

SUPPORTED_CONFIGS = ("t", "s", "m", "c")


# =============================================================================
# Sublayer Name Mapping
# =============================================================================
#
# LibreYOLO's blocks (``libreyolo.models.yolo9.nn``) are ported from
# MultimediaTechLab/YOLO with the upstream attribute names, so block sublayer
# keys map one-to-one. Only the detection head is regrouped: the upstream
# per-level ``heads.<i>`` modules become the ``anchor_convs`` / ``class_convs``
# lists of ``YOLO9Head``.


def map_conv_keys(yolo_suffix: str) -> str:
    """Map Conv layer keys (identical naming)."""
    return yolo_suffix


def map_aconv_keys(yolo_suffix: str) -> str:
    """Map AConv keys (identical naming: ``conv.{conv,bn}``)."""
    return yolo_suffix


def map_adown_keys(yolo_suffix: str) -> str:
    """Map ADown keys (identical naming: ``conv1`` / ``conv2``)."""
    return yolo_suffix


def map_elan_keys(yolo_suffix: str) -> str:
    """Map ELAN keys (identical naming: ``conv1..conv4``)."""
    return yolo_suffix


def map_repncspelan_keys(yolo_suffix: str) -> str:
    """Map RepNCSPELAN keys (identical naming, nested RepNCSP/bottleneck)."""
    return yolo_suffix


def map_sppelan_keys(yolo_suffix: str) -> str:
    """Map SPPELAN keys (identical naming: ``conv1`` / ``conv5``)."""
    return yolo_suffix


def map_cblinear_keys(yolo_suffix: str) -> str:
    """Map CBLinear keys (identical naming: ``conv.{weight,bias}``)."""
    return yolo_suffix


def map_detection_keys(yolo_suffix: str) -> Optional[str]:
    """Map MultiheadDetection keys onto ``YOLO9Head``.

    ``heads.N.anchor_conv`` -> ``anchor_convs.N`` (box) and
    ``heads.N.class_conv`` -> ``class_convs.N`` (class). ``anc2vec`` is
    skipped (LibreYOLO keeps the DFL bins as a non-persistent buffer).
    """
    if "anc2vec" in yolo_suffix:
        return None
    result = re.sub(r"^heads\.(\d+)\.anchor_conv\.", r"anchor_convs.\1.", yolo_suffix)
    result = re.sub(r"^heads\.(\d+)\.class_conv\.", r"class_convs.\1.", result)
    return result


# =============================================================================
# Layer Type Detection
# =============================================================================


def get_layer_type(layer_idx: int, config: str) -> str:
    """Determine the layer type based on layer index and config."""
    if layer_idx in (0, 1):
        return "conv"
    if layer_idx == 2:
        return "elan" if config in ("t", "s") else "repncspelan"
    if layer_idx in (3, 5, 7, 16, 19):
        return "adown" if config == "c" else "aconv"
    if layer_idx in (4, 6, 8, 12, 15, 18, 21):
        return "repncspelan"
    if layer_idx == 9:
        return "sppelan"
    if layer_idx == 22:
        return "detection"
    if config in ("t", "s"):
        if layer_idx == 23:
            return "sppelan"
        if layer_idx in (26, 29):
            return "repncspelan"
        if layer_idx == 30:
            return "detection"
        return "unknown"
    # v9-m/v9-c auxiliary branch
    if layer_idx in (23, 24, 25):
        return "cblinear"
    if layer_idx in (26, 27):
        return "conv"
    if layer_idx in (29, 32, 35):
        return "adown" if config == "c" else "aconv"
    if layer_idx in (28, 31, 34, 37):
        return "repncspelan"
    if layer_idx == 38:
        return "detection"
    return "unknown"


_SUBLAYER_MAPPERS = {
    "conv": map_conv_keys,
    "aconv": map_aconv_keys,
    "adown": map_adown_keys,
    "elan": map_elan_keys,
    "repncspelan": map_repncspelan_keys,
    "sppelan": map_sppelan_keys,
    "cblinear": map_cblinear_keys,
    "detection": map_detection_keys,
}


# =============================================================================
# Conversion
# =============================================================================


def convert_key(yolo_key: str, config: str) -> Tuple[str, bool]:
    """Convert a single upstream YOLO9 key to LibreYOLO format.

    Returns ``(converted_key, success)``.
    """
    layer_map = LAYER_MAPS[config]

    parts = yolo_key.split(".", 1)
    if len(parts) < 2:
        return yolo_key, False

    layer_idx_str, suffix = parts
    if not layer_idx_str.isdigit():
        return yolo_key, False

    layer_idx = int(layer_idx_str)
    if layer_idx not in layer_map:
        return yolo_key, False

    libre_prefix = layer_map[layer_idx]
    layer_type = get_layer_type(layer_idx, config)
    mapper = _SUBLAYER_MAPPERS.get(layer_type)
    if mapper is None:
        return yolo_key, False

    libre_suffix = mapper(suffix)
    if libre_suffix is None:
        return yolo_key, False

    return f"{libre_prefix}.{libre_suffix}", True


def convert_state_dict(
    state_dict: Dict[str, torch.Tensor],
    config: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Convert an upstream YOLO9 ``state_dict`` to LibreYOLO key layout.

    Args:
        state_dict: Upstream tensor state dict (numbered-index keys).
        config: Model config, one of ``t``/``s``/``m``/``c``.

    Returns:
        ``(converted_state_dict, stats)`` where ``stats`` has ``converted``,
        ``skipped`` (unmapped auxiliary leftovers, layers >= 23: the
        ``anc2vec`` weights of the auxiliary head) and ``failed`` counts
        (unmapped main-path keys, which include the main head's ``anc2vec``).
    """
    if config not in LAYER_MAPS:
        raise ValueError(
            f"Unknown YOLO9 config {config!r}; expected one of {SUPPORTED_CONFIGS}."
        )

    converted: Dict[str, torch.Tensor] = {}
    skipped = 0
    failed = 0

    for yolo_key, value in state_dict.items():
        libre_key, success = convert_key(yolo_key, config)
        if success:
            converted[libre_key] = value
            continue
        head = yolo_key.split(".", 1)[0]
        if head.isdigit() and int(head) >= 23:
            skipped += 1  # auxiliary leftover (anc2vec) — LibreYOLO derives it
        else:
            failed += 1

    return converted, {"converted": len(converted), "skipped": skipped, "failed": failed}


# =============================================================================
# Upstream detection + metadata inference
# =============================================================================

_UPSTREAM_HEAD_RE = re.compile(r"^\d+\.heads\.\d+\.(class_conv|anchor_conv)\.")


def is_upstream_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Return True for an upstream MultimediaTechLab/YOLO YOLO9 ``state_dict``.

    Identified by the numbered detection-head signature
    (``<idx>.heads.<n>.class_conv`` / ``anchor_conv``), which is absent from
    LibreYOLO's semantic key layout.
    """
    return any(_UPSTREAM_HEAD_RE.match(k) for k in state_dict)


def infer_config(state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
    """Infer the YOLO9 config (t/s/m/c) from upstream stem/first-block widths."""
    stem = state_dict.get("0.conv.weight")
    if stem is None:
        return None
    first_channel = int(stem.shape[0])
    if first_channel == 16:
        return "t"
    if first_channel == 64:
        return "c"
    if first_channel == 32:
        block = state_dict.get("2.conv1.conv.weight")
        if block is not None:
            mid = int(block.shape[0])
            if mid == 64:
                return "s"
            if mid == 128:
                return "m"
    return None


def infer_nb_classes(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    """Infer class count from the upstream detection head (``class_conv.*.2``)."""
    best: Optional[int] = None
    for key, tensor in state_dict.items():
        m = re.match(r"\d+\.heads\.(\d+)\.class_conv\.2\.weight$", key)
        if m and tensor.ndim >= 1:
            best = int(tensor.shape[0])
            if m.group(1) == "0":  # prefer the first (P3) head
                return best
    return best


# =============================================================================
# Legacy LibreYOLO key layout
# =============================================================================
#
# LibreYOLO checkpoints written before the yolo9 blocks took the upstream
# sublayer names use ``cv1``/``cv2``/... for block convolutions, ``m`` for the
# RepNCSP bottleneck stack, ``cv`` for the AConv convolution and
# ``cv2``/``cv3`` (``one2one_cv2``/``one2one_cv3`` for E2E) for the box/class
# towers of the detection head. The rename is purely structural: tensors and
# module order are unchanged, so every legacy file loads after the upgrade.

_LEGACY_BLOCK_PREFIXES = ("backbone.", "neck.", "aux.")
_LEGACY_HEAD_TOWER_RE = re.compile(
    r"^(head|aux_head)\.(one2one_cv2|one2one_cv3|cv2|cv3)\.(\d+)\."
)
_LEGACY_HEAD_TOWER_PREFIX_RE = re.compile(
    r"^(head|aux_head)\.(one2one_cv2|one2one_cv3|cv2|cv3)\."
)
_LEGACY_HEAD_TOWERS = {
    "cv2": "anchor_convs",
    "cv3": "class_convs",
    "one2one_cv2": "one_to_one_anchor_convs",
    "one2one_cv3": "one_to_one_class_convs",
}
# Legacy DFL conv weight (v1.1.x files) and derived head state that is never
# loaded (anchor/stride caches).
_LEGACY_DROPPED_RE = re.compile(r"^(head|aux_head)\.(dfl(\..*)?|stride|strides|anchors)$")


def upgrade_legacy_key(key: str) -> Optional[str]:
    """Map one legacy LibreYOLO yolo9-family key to the current layout.

    Returns ``None`` for keys that are dropped. Current-layout keys and keys
    outside the yolo9 modules pass through unchanged, so the function is
    idempotent.
    """
    if key.startswith("detect."):
        key = "head." + key[len("detect."):]
    if _LEGACY_DROPPED_RE.match(key):
        return None
    if key.startswith(_LEGACY_BLOCK_PREFIXES):
        key = re.sub(r"\.cv(\d)(?=\.)", r".conv\1", key)
        key = re.sub(r"\.m(?=\.)", ".bottleneck", key)
        key = re.sub(r"(\.down\d)\.cv(?=\.)", r"\1.conv", key)
        return key
    match = _LEGACY_HEAD_TOWER_RE.match(key)
    if match:
        prefix, tower, index = match.groups()
        key = f"{prefix}.{_LEGACY_HEAD_TOWERS[tower]}.{index}.{key[match.end():]}"
    return key


def upgrade_legacy_module_name(name: str) -> str:
    """Map a legacy module name or name prefix to the current layout.

    Counterpart of :func:`upgrade_legacy_key` for name-bearing metadata such
    as quantization exclusions (``"backbone.elan1.cv1."``), which name modules
    rather than tensors and may or may not end with a dot. Names that are
    already current, or that match nothing, come back unchanged.
    """
    trailing_dot = name.endswith(".")
    probe = name if trailing_dot else name + "."
    if probe.startswith("detect."):
        probe = "head." + probe[len("detect."):]
    if probe.startswith(_LEGACY_BLOCK_PREFIXES):
        probe = upgrade_legacy_key(probe) or probe
    else:
        match = _LEGACY_HEAD_TOWER_PREFIX_RE.match(probe)
        if match:
            prefix, tower = match.groups()
            probe = f"{prefix}.{_LEGACY_HEAD_TOWERS[tower]}.{probe[match.end():]}"
    return probe if trailing_dot else probe[:-1]


def upgrade_legacy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return ``state_dict`` in the current key layout (order-preserving).

    Applies :func:`upgrade_legacy_key` to every key and drops the keys it
    maps to ``None``. Idempotent: current-layout dicts come back unchanged.
    """
    upgraded: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = upgrade_legacy_key(key) if isinstance(key, str) else key
        if new_key is not None:
            upgraded[new_key] = value
    return upgraded
