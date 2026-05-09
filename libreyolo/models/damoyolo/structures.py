"""Per-size architecture metadata for DAMO-YOLO TinyNAS backbones.

These dicts are the authoritative architecture specs upstream stores in
``damo/base_models/backbones/nas_backbones/tinynas_*.txt``. Inlined here so
LibreYOLO doesn't ship pretty-printed text files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---- TinyNAS_res structures (used by N / T / S / L) ----------------------

# tinynas_L20_k1kx.txt → DAMO-YOLO-T
TINYNAS_L20_K1KX = [
    {"class": "ConvKXBNRELU", "in": 3, "k": 3, "out": 24, "s": 1},
    {"L": 2, "btn": 24, "class": "SuperResConvK1KX", "in": 24, "k": 3, "out": 64, "s": 2},
    {"L": 2, "btn": 64, "class": "SuperResConvK1KX", "in": 64, "k": 3, "out": 96, "s": 2},
    {"L": 2, "btn": 96, "class": "SuperResConvK1KX", "in": 96, "k": 3, "out": 192, "s": 2},
    {"L": 2, "btn": 152, "class": "SuperResConvK1KX", "in": 192, "k": 3, "out": 192, "s": 1},
    {"L": 1, "btn": 192, "class": "SuperResConvK1KX", "in": 192, "k": 3, "out": 384, "s": 2},
]


# tinynas_L25_k1kx.txt → DAMO-YOLO-S
TINYNAS_L25_K1KX = [
    {"class": "ConvKXBNRELU", "in": 3, "k": 3, "out": 32, "s": 1},
    {"L": 1, "btn": 24, "class": "SuperResConvK1KX", "in": 32, "k": 3, "out": 128, "s": 2},
    {"L": 5, "btn": 88, "class": "SuperResConvK1KX", "in": 128, "k": 3, "out": 128, "s": 2},
    {"L": 3, "btn": 128, "class": "SuperResConvK1KX", "in": 128, "k": 3, "out": 256, "s": 2},
    {"L": 2, "btn": 120, "class": "SuperResConvK1KX", "in": 256, "k": 3, "out": 256, "s": 1},
    {"L": 1, "btn": 144, "class": "SuperResConvK1KX", "in": 256, "k": 3, "out": 512, "s": 2},
]


# tinynas_L35_kxkx.txt → DAMO-YOLO-M (uses TinyNAS_csp backbone, kxkx blocks)
TINYNAS_L35_KXKX = [
    {"class": "ConvKXBNRELU", "in": 3, "k": 3, "out": 32, "s": 1},
    {"L": 2, "btn": 64, "class": "SuperResConvKXKX", "in": 32, "k": 3, "out": 128, "s": 2},
    {"L": 4, "btn": 64, "class": "SuperResConvKXKX", "in": 128, "k": 3, "out": 128, "s": 2},
    {"L": 4, "btn": 256, "class": "SuperResConvKXKX", "in": 128, "k": 3, "out": 256, "s": 2},
    {"L": 4, "btn": 256, "class": "SuperResConvKXKX", "in": 256, "k": 3, "out": 256, "s": 1},
    {"L": 3, "btn": 256, "class": "SuperResConvKXKX", "in": 256, "k": 3, "out": 512, "s": 2},
]


# tinynas_L45_kxkx.txt → DAMO-YOLO-L (TinyNAS_csp, kxkx). Pretrained
# weights are not currently downloadable (Aliyun bucket is 404 and
# ModelScope hosts only T/S/M). Structure is registered for
# from-scratch training; download routing falls through to the user.
TINYNAS_L45_KXKX = [
    {"class": "ConvKXBNRELU", "in": 3, "k": 3, "out": 32, "s": 1},
    {"L": 3, "btn": 96, "class": "SuperResConvKXKX", "in": 32, "k": 3, "out": 128, "s": 2},
    {"L": 5, "btn": 96, "class": "SuperResConvKXKX", "in": 128, "k": 3, "out": 128, "s": 2},
    {"L": 5, "btn": 384, "class": "SuperResConvKXKX", "in": 128, "k": 3, "out": 256, "s": 2},
    {"L": 5, "btn": 384, "class": "SuperResConvKXKX", "in": 256, "k": 3, "out": 256, "s": 1},
    {"L": 4, "btn": 384, "class": "SuperResConvKXKX", "in": 256, "k": 3, "out": 512, "s": 2},
]


@dataclass(frozen=True)
class FamilyConfig:
    """A complete DAMO-YOLO family member spec."""

    structure: List[Dict]
    backbone_class: str   # "tinynas_res", "tinynas_csp", or "tinynas_mob"
    backbone_with_spp: bool
    backbone_use_focus: bool
    backbone_act: str
    backbone_reparam: bool
    backbone_out_indices: Tuple[int, int, int]
    neck_in_channels: Tuple[int, int, int]
    neck_out_channels: Tuple[int, int, int]
    neck_depth: float
    neck_hidden_ratio: float
    neck_act: str
    neck_spp: bool
    head_in_channels: Tuple[int, int, int]
    head_stacked_convs: int
    head_reg_max: int
    head_act: str
    head_legacy: bool
    head_feat_channels: int = 256
    head_last_kernel_size: int = 3
    backbone_depthwise: bool = False
    backbone_use_se: bool = False
    neck_depthwise: bool = False


# ---- DAMO-YOLO-T (42.0 mAP, target for first-pass parity) ----------------

DAMOYOLO_T = FamilyConfig(
    structure=TINYNAS_L20_K1KX,
    backbone_class="tinynas_res",
    backbone_with_spp=True,
    backbone_use_focus=True,
    backbone_act="relu",
    backbone_reparam=True,
    backbone_out_indices=(2, 4, 5),
    neck_in_channels=(96, 192, 384),
    neck_out_channels=(64, 128, 256),
    neck_depth=1.0,
    neck_hidden_ratio=1.0,
    neck_act="relu",
    neck_spp=False,
    head_in_channels=(64, 128, 256),
    head_stacked_convs=0,
    head_reg_max=16,
    head_act="silu",
    # The publicly hosted ModelScope `damoyolo_tinynasL20_T.pt` was trained
    # with legacy=True (cls head emits num_classes + 1 channels — extra
    # channel is unused). The newer `_420.pth` release used legacy=False
    # but Aliyun took those down, so we mirror the actually-loadable
    # checkpoint's layout.
    head_legacy=True,
)


DAMOYOLO_S = FamilyConfig(
    structure=TINYNAS_L25_K1KX,
    backbone_class="tinynas_res",
    backbone_with_spp=True,
    backbone_use_focus=True,
    backbone_act="relu",
    backbone_reparam=True,
    backbone_out_indices=(2, 4, 5),
    neck_in_channels=(128, 256, 512),
    neck_out_channels=(128, 256, 512),
    neck_depth=1.0,
    neck_hidden_ratio=0.75,
    neck_act="relu",
    neck_spp=False,
    head_in_channels=(128, 256, 512),
    head_stacked_convs=0,
    head_reg_max=16,
    head_act="silu",
    head_legacy=True,
)


DAMOYOLO_M = FamilyConfig(
    structure=TINYNAS_L35_KXKX,
    backbone_class="tinynas_csp",
    backbone_with_spp=True,
    backbone_use_focus=True,
    backbone_act="silu",
    backbone_reparam=True,
    # CSP backbone outputs from csp_stage indices (5-element list,
    # not the 6-element raw block list).
    backbone_out_indices=(2, 3, 4),
    neck_in_channels=(128, 256, 512),
    neck_out_channels=(128, 256, 512),
    neck_depth=1.5,
    neck_hidden_ratio=1.0,
    neck_act="silu",
    neck_spp=False,
    head_in_channels=(128, 256, 512),
    head_stacked_convs=0,
    head_reg_max=16,
    head_act="silu",
    head_legacy=True,
)


# tinynas_nano_*.txt → DAMO-YOLO Nano variants (TinyNAS_mob backbone, depthwise everywhere)
TINYNAS_NANO_SMALL = [
    {"class": "ConvKXBNRELU", "in": 3, "k": 3, "out": 16, "s": 1},
    {"L": 1, "btn": 24, "class": "SuperResConvK1KX", "in": 16, "k": 3, "out": 24, "s": 2},
    {"L": 2, "btn": 64, "class": "SuperResConvK1KX", "in": 24, "k": 3, "out": 40, "s": 2},
    {"L": 2, "btn": 40, "class": "SuperResConvK1KX", "in": 40, "k": 3, "out": 64, "s": 2},
    {"L": 2, "btn": 152, "class": "SuperResConvK1KX", "in": 64, "k": 3, "out": 80, "s": 1},
    {"L": 2, "btn": 192, "class": "SuperResConvK1KX", "in": 80, "k": 3, "out": 160, "s": 2},
]

TINYNAS_NANO_MIDDLE = [
    {"class": "ConvKXBNRELU", "in": 3, "k": 3, "out": 16, "s": 1},
    {"L": 2, "btn": 24, "class": "SuperResConvK1KX", "in": 16, "k": 3, "out": 40, "s": 2},
    {"L": 2, "btn": 64, "class": "SuperResConvK1KX", "in": 40, "k": 3, "out": 64, "s": 2},
    {"L": 2, "btn": 40, "class": "SuperResConvK1KX", "in": 64, "k": 3, "out": 112, "s": 2},
    {"L": 2, "btn": 152, "class": "SuperResConvK1KX", "in": 112, "k": 3, "out": 128, "s": 1},
    {"L": 1, "btn": 192, "class": "SuperResConvK1KX", "in": 128, "k": 3, "out": 256, "s": 2},
]

TINYNAS_NANO_LARGE = [
    {"class": "ConvKXBNRELU", "in": 3, "k": 3, "out": 24, "s": 1},
    {"L": 1, "btn": 24, "class": "SuperResConvK1KX", "in": 24, "k": 3, "out": 48, "s": 2},
    {"L": 2, "btn": 64, "class": "SuperResConvK1KX", "in": 48, "k": 3, "out": 80, "s": 2},
    {"L": 2, "btn": 40, "class": "SuperResConvK1KX", "in": 80, "k": 3, "out": 160, "s": 2},
    {"L": 3, "btn": 152, "class": "SuperResConvK1KX", "in": 160, "k": 3, "out": 160, "s": 1},
    {"L": 2, "btn": 192, "class": "SuperResConvK1KX", "in": 160, "k": 3, "out": 320, "s": 2},
]


def _make_nano_config(structure, neck_io: Tuple[int, int, int]) -> "FamilyConfig":
    return FamilyConfig(
        structure=structure,
        backbone_class="tinynas_mob",
        backbone_with_spp=True,
        backbone_use_focus=False,
        backbone_act="silu",
        backbone_reparam=False,
        backbone_out_indices=(2, 4, 5),
        backbone_depthwise=True,
        backbone_use_se=False,
        neck_in_channels=neck_io,
        neck_out_channels=neck_io,
        neck_depth=0.5,
        neck_hidden_ratio=0.5,
        neck_act="silu",
        neck_spp=False,
        neck_depthwise=True,
        head_in_channels=neck_io,
        head_stacked_convs=0,
        head_reg_max=7,
        head_act="silu",
        head_legacy=False,
        head_last_kernel_size=1,
    )


DAMOYOLO_NS = _make_nano_config(TINYNAS_NANO_SMALL, (40, 80, 160))
DAMOYOLO_NM = _make_nano_config(TINYNAS_NANO_MIDDLE, (64, 128, 256))
DAMOYOLO_NL = _make_nano_config(TINYNAS_NANO_LARGE, (80, 160, 320))


DAMOYOLO_L = FamilyConfig(
    structure=TINYNAS_L45_KXKX,
    backbone_class="tinynas_csp",
    backbone_with_spp=True,
    backbone_use_focus=True,
    backbone_act="silu",
    backbone_reparam=True,
    backbone_out_indices=(2, 3, 4),
    neck_in_channels=(128, 256, 512),
    neck_out_channels=(128, 256, 512),
    neck_depth=2.0,        # upstream: depth 2.0 for L
    neck_hidden_ratio=1.0,
    neck_act="silu",
    neck_spp=False,
    head_in_channels=(128, 256, 512),
    head_stacked_convs=0,
    head_reg_max=16,
    head_act="silu",
    head_legacy=True,
)


SIZES: Dict[str, FamilyConfig] = {
    "ns": DAMOYOLO_NS,
    "nm": DAMOYOLO_NM,
    "nl": DAMOYOLO_NL,
    "t": DAMOYOLO_T,
    "s": DAMOYOLO_S,
    "m": DAMOYOLO_M,
    "l": DAMOYOLO_L,
}
