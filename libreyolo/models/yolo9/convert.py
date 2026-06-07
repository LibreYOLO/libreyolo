"""Convert upstream YOLO9 weights to LibreYOLO key layout.

The upstream YOLO9 release (MultimediaTechLab/YOLO, MIT) ships plain
``state_dict`` checkpoints that use numbered layer indices (``0.``, ``1.``,
``2.`` …) while LibreYOLO uses semantic module names (``backbone.conv0``,
``neck.elan_up1`` …). This module owns the index/sublayer remapping so both the
offline ``weights/convert_yolo9_weights.py`` script and the runtime
auto-conversion path in :mod:`libreyolo.models.autoconvert` share one
implementation.

The conversion is structural only — it renames keys and drops the
auxiliary-detection-head weights (layers >= 23) and the ``anc2vec`` buffers that
LibreYOLO derives internally. Class count is taken from the upstream detection
head, so fine-tuned checkpoints with a non-COCO ``nc`` convert correctly.
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


def map_conv_keys(yolo_suffix: str) -> str:
    """Map Conv layer keys. YOLO and LibreYOLO use same naming."""
    return yolo_suffix


def map_aconv_keys(yolo_suffix: str) -> str:
    """Map AConv keys. YOLO ``conv.{conv,bn}`` -> LibreYOLO ``cv.{conv,bn}``."""
    return re.sub(r"^conv\.", "cv.", yolo_suffix)


def map_adown_keys(yolo_suffix: str) -> str:
    """Map ADown keys. YOLO ``conv1/conv2`` -> LibreYOLO ``cv1/cv2``."""
    return yolo_suffix.replace("conv1", "cv1").replace("conv2", "cv2")


def map_elan_keys(yolo_suffix: str) -> str:
    """Map ELAN keys (yolo9-t/s first block). ``conv{1..4}`` -> ``cv{1..4}``."""
    return re.sub(r"^conv([1234])\.", r"cv\1.", yolo_suffix)


def map_repncspelan_keys(yolo_suffix: str) -> str:
    """Map RepNCSPELAN keys (nested bottleneck structure)."""
    result = yolo_suffix
    # Map main conv names: conv1/2/3/4 -> cv1/2/3/4
    result = re.sub(r"^conv([1234])\.", r"cv\1.", result)
    # Map RepNCSP internal names (inside cv2.0 and cv3.0)
    result = re.sub(r"\.conv([123])\.", r".cv\1.", result)
    # Map bottleneck -> m
    result = result.replace(".bottleneck.", ".m.")
    # Inside RepNCSP Bottleneck, YOLO conv1/conv2 -> LibreYOLO cv1/cv2
    result = re.sub(r"\.m\.(\d+)\.conv([12])\.", r".m.\1.cv\2.", result)
    return result


def map_sppelan_keys(yolo_suffix: str) -> str:
    """Map SPPELAN keys. ``conv1/conv5`` -> ``cv1/cv5``."""
    result = yolo_suffix.replace("conv1.", "cv1.").replace("conv5.", "cv5.")
    return result


def map_detection_keys(yolo_suffix: str) -> Optional[str]:
    """Map MultiheadDetection keys.

    YOLO ``heads.N.anchor_conv`` -> ``cv2.N`` (box), ``heads.N.class_conv`` ->
    ``cv3.N`` (class). ``anc2vec`` is skipped (LibreYOLO derives DFL internally).
    """
    result = yolo_suffix
    result = re.sub(r"^heads\.(\d+)\.anchor_conv\.", r"cv2.\1.", result)
    result = re.sub(r"^heads\.(\d+)\.class_conv\.", r"cv3.\1.", result)
    if "anc2vec" in result:
        return None
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
    return "unknown"


_SUBLAYER_MAPPERS = {
    "conv": map_conv_keys,
    "aconv": map_aconv_keys,
    "adown": map_adown_keys,
    "elan": map_elan_keys,
    "repncspelan": map_repncspelan_keys,
    "sppelan": map_sppelan_keys,
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
        ``skipped`` (auxiliary head, layers >= 23) and ``failed`` counts.
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
            skipped += 1  # auxiliary detection head — not used at inference
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
        m = re.match(r"\d+\.heads\.(\d+)\.class_conv\.\d+\.weight", key)
        if m and tensor.ndim >= 1:
            best = int(tensor.shape[0])
            if m.group(1) == "0":  # prefer the first (P3) head
                return best
    return best
