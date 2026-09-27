"""PP-YOLOE upstream checkpoint recognition and unwrapping.

Shared by the runtime auto-converter (``libreyolo/models/autoconvert.py``) and
the offline ``weights/convert_ppyoloe_weights.py`` script.

Released PP-YOLOE checkpoints from the source CDN are a mapping with a single
top-level ``net`` key whose entries all carry a leading ``module.`` prefix
(DDP-wrapped at save time). Attribute names below that prefix already match
this port, so conversion is a strip, not a remap.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

# Keys every PP-YOLOE state dict must have once ``module.`` is stripped. The
# conjunction is deliberately narrow: YOLO-NAS shares the PP-YOLOE loss but
# names its head ``heads.head1.cls_pred``, and PicoDet uses ``head.gfl_cls``,
# so neither can satisfy all six of these.
REQUIRED_KEYS = (
    "backbone.stem.conv1.seq.conv.weight",
    "head.pred_cls.0.weight",
    "head.pred_cls.1.weight",
    "head.pred_cls.2.weight",
    "head.pred_reg.0.weight",
    "head.pred_reg.1.weight",
    "head.pred_reg.2.weight",
)

# 4 * (reg_max + 1) with reg_max=16.
REG_OUT_CHANNELS = 68

# Neck output widths per size, in head order (stride 32, 16, 8). Read off
# ``head.pred_cls.<level>.weight.shape[1]``; three independent widths per size
# make this a far stronger size signature than the filename.
_HEAD_WIDTHS_TO_SIZE = {
    (384, 192, 96): "s",
    (576, 288, 144): "m",
    (768, 384, 192): "l",
    (960, 480, 240): "x",
}

# One depth-specific key per size: backbone stage 1 has ``max(round(6 * d), 1)``
# blocks, so the highest existing block index separates s/m (2/4) from l/x
# (6/8), and the neck's per-stage block count separates l from x.
_STAGE1_BLOCKS_TO_SIZES = {2: ("s",), 4: ("m",), 6: ("l",), 8: ("x",)}


def _strip_module_prefix(state_dict: Mapping) -> Dict:
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def unwrap_ppyoloe_checkpoint(checkpoint):
    """Pull the model state out of a released PP-YOLOE checkpoint layout."""
    if not isinstance(checkpoint, Mapping):
        return checkpoint
    for key in ("ema_net", "net", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def count_backbone_stage_blocks(state_dict: Mapping, stage: int) -> int:
    """Number of ``CSPResNetBasicBlock``s in a backbone stage."""
    prefix = f"backbone.stages.{stage}.blocks."
    indices = set()
    for key in state_dict:
        if key.startswith(prefix):
            head = key[len(prefix):].split(".", 1)[0]
            if head.isdigit():
                indices.add(int(head))
    return len(indices)


def detect_size_from_state(state_dict: Mapping) -> Optional[str]:
    """Infer the released size from three head widths plus a depth signature.

    Returns ``None`` when the width signature is unknown or when the depth
    signature contradicts it, rather than guessing a nearby size.
    """
    widths = []
    for level in range(3):
        tensor = state_dict.get(f"head.pred_cls.{level}.weight")
        if tensor is None or tensor.ndim != 4:
            return None
        widths.append(int(tensor.shape[1]))
    size = _HEAD_WIDTHS_TO_SIZE.get(tuple(widths))
    if size is None:
        return None

    blocks = count_backbone_stage_blocks(state_dict, stage=1)
    expected = _STAGE1_BLOCKS_TO_SIZES.get(blocks)
    if expected is not None and size not in expected:
        return None
    return size


def detect_nb_classes_from_state(state_dict: Mapping) -> Optional[int]:
    """Class count from the head, confirmed identical across all three levels."""
    counts = set()
    for level in range(3):
        tensor = state_dict.get(f"head.pred_cls.{level}.weight")
        if tensor is None or tensor.ndim != 4:
            return None
        counts.add(int(tensor.shape[0]))
    if len(counts) != 1:
        return None
    nc = counts.pop()
    return nc if nc > 0 else None


def is_ppyoloe_state_dict(state_dict: Mapping) -> bool:
    """True only for PP-YOLOE tensors (``module.`` prefix already stripped)."""
    if not all(key in state_dict for key in REQUIRED_KEYS):
        return False
    for level in range(3):
        reg = state_dict.get(f"head.pred_reg.{level}.weight")
        if reg is None or reg.ndim != 4 or int(reg.shape[0]) != REG_OUT_CHANNELS:
            return False
    return True


def is_upstream_state_dict(state_dict: Mapping) -> bool:
    """True for a released PP-YOLOE state dict, prefixed or not."""
    return is_ppyoloe_state_dict(_strip_module_prefix(state_dict))


def convert_upstream(state_dict: Mapping) -> Dict:
    """Strip the leading ``module.`` prefix; reject anything else unexpected.

    Numerics are untouched: this is a rename-only conversion.
    """
    converted = _strip_module_prefix(state_dict)
    if not is_ppyoloe_state_dict(converted):
        raise ValueError(
            "State dict does not look like a PP-YOLOE checkpoint: expected "
            f"{REQUIRED_KEYS} with {REG_OUT_CHANNELS} regression channels."
        )
    return converted
