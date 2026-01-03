"""
Convert YOLOv7 weights from official YOLO repo format to LibreYOLO format.

Usage:
    python weights/convert_yolo7_weights.py weights/v7.pt weights/libreyolo7.pt --verify

The YOLO repo uses numbered layer indices (0., 1., 2., etc.) while LibreYOLO
uses semantic naming (backbone.conv0, neck.conv52, etc.).
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch


# =============================================================================
# Layer Index to LibreYOLO Name Mapping
# =============================================================================

# Backbone layers (0-50)
BACKBONE_MAP = {
    0: "backbone.conv0",
    1: "backbone.conv1",
    2: "backbone.conv2",
    3: "backbone.conv3",
    4: "backbone.conv4",
    5: "backbone.conv5",
    6: "backbone.conv6",
    7: "backbone.conv7",
    8: "backbone.conv8",
    9: "backbone.conv9",
    11: "backbone.conv11",
    13: "backbone.conv13",
    14: "backbone.conv14",
    15: "backbone.conv15",
    17: "backbone.conv17",
    18: "backbone.conv18",
    19: "backbone.conv19",
    20: "backbone.conv20",
    21: "backbone.conv21",
    22: "backbone.conv22",
    24: "backbone.conv24",
    26: "backbone.conv26",
    27: "backbone.conv27",
    28: "backbone.conv28",
    30: "backbone.conv30",
    31: "backbone.conv31",
    32: "backbone.conv32",
    33: "backbone.conv33",
    34: "backbone.conv34",
    35: "backbone.conv35",
    37: "backbone.conv37",
    39: "backbone.conv39",
    40: "backbone.conv40",
    41: "backbone.conv41",
    43: "backbone.conv43",
    44: "backbone.conv44",
    45: "backbone.conv45",
    46: "backbone.conv46",
    47: "backbone.conv47",
    48: "backbone.conv48",
    50: "backbone.conv50",
}

# Neck layers (51-101)
NECK_MAP = {
    51: "neck.sppcspc",  # Special handling
    52: "neck.conv52",
    54: "neck.conv54",
    56: "neck.conv56",
    57: "neck.conv57",
    58: "neck.conv58",
    59: "neck.conv59",
    60: "neck.conv60",
    61: "neck.conv61",
    63: "neck.conv63",
    64: "neck.conv64",
    66: "neck.conv66",
    68: "neck.conv68",
    69: "neck.conv69",
    70: "neck.conv70",
    71: "neck.conv71",
    72: "neck.conv72",
    73: "neck.conv73",
    75: "neck.conv75",
    77: "neck.conv77",
    78: "neck.conv78",
    79: "neck.conv79",
    81: "neck.conv81",
    82: "neck.conv82",
    83: "neck.conv83",
    84: "neck.conv84",
    85: "neck.conv85",
    86: "neck.conv86",
    88: "neck.conv88",
    90: "neck.conv90",
    91: "neck.conv91",
    92: "neck.conv92",
    94: "neck.conv94",
    95: "neck.conv95",
    96: "neck.conv96",
    97: "neck.conv97",
    98: "neck.conv98",
    99: "neck.conv99",
    101: "neck.conv101",
}

# RepConv layers (102-104)
REP_MAP = {
    102: "rep102",
    103: "rep103",
    104: "rep104",
}

# Detection head (105)
DETECT_MAP = {
    105: "detect",
}


# =============================================================================
# SPPCSPC Key Mapping
# =============================================================================

SPPCSPC_KEY_MAP = {
    "pre_conv": "pre_conv",
    "short_conv": "short_conv",
    "merge_conv": "merge_conv",
    "post_conv": "post_conv",
    # Note: pools are not in state_dict (no learnable params)
}


def map_sppcspc_keys(suffix: str) -> Optional[str]:
    """Map SPPCSPC layer keys."""
    # Original keys: pre_conv.*, short_conv.*, merge_conv.*, post_conv.*
    parts = suffix.split('.', 1)
    if len(parts) < 2:
        return None

    sublayer = parts[0]
    rest = parts[1]

    if sublayer in SPPCSPC_KEY_MAP:
        return f"{SPPCSPC_KEY_MAP[sublayer]}.{rest}"

    # Handle cv5, cv6, cv7 (not in original, but checking just in case)
    return None


# =============================================================================
# RepConv Key Mapping
# =============================================================================

def map_repconv_keys(suffix: str) -> Optional[str]:
    """Map RepConv layer keys.

    Original: conv1.*, conv2.*
    LibreYOLO: conv1.*, conv2.* (same naming)
    """
    # Both use the same naming (conv1, conv2)
    if suffix.startswith("conv1.") or suffix.startswith("conv2."):
        return suffix
    return None


# =============================================================================
# Detection Head Key Mapping
# =============================================================================

def map_detect_keys(suffix: str) -> Optional[str]:
    """Map detection head keys.

    Original: heads.N.head_conv.*, heads.N.implicit_a.*, heads.N.implicit_m.*
    LibreYOLO: m.N.*, ia.N.*, im.N.*
    """
    # heads.0.head_conv.weight -> m.0.weight
    match = re.match(r'^heads\.(\d+)\.head_conv\.(.+)$', suffix)
    if match:
        idx, rest = match.groups()
        return f"m.{idx}.{rest}"

    # heads.0.implicit_a.implicit -> ia.0.implicit
    match = re.match(r'^heads\.(\d+)\.implicit_a\.(.+)$', suffix)
    if match:
        idx, rest = match.groups()
        return f"ia.{idx}.{rest}"

    # heads.0.implicit_m.implicit -> im.0.implicit
    match = re.match(r'^heads\.(\d+)\.implicit_m\.(.+)$', suffix)
    if match:
        idx, rest = match.groups()
        return f"im.{idx}.{rest}"

    return None


# =============================================================================
# Main Conversion Logic
# =============================================================================

def convert_key(yolo_key: str) -> Tuple[str, bool]:
    """Convert a single YOLO key to LibreYOLO format."""
    parts = yolo_key.split('.', 1)
    if len(parts) < 2:
        return yolo_key, False

    layer_idx_str, suffix = parts
    if not layer_idx_str.isdigit():
        return yolo_key, False

    layer_idx = int(layer_idx_str)

    # Check backbone
    if layer_idx in BACKBONE_MAP:
        return f"{BACKBONE_MAP[layer_idx]}.{suffix}", True

    # Check neck (regular convs)
    if layer_idx in NECK_MAP and layer_idx != 51:
        return f"{NECK_MAP[layer_idx]}.{suffix}", True

    # Check SPPCSPC (layer 51)
    if layer_idx == 51:
        mapped = map_sppcspc_keys(suffix)
        if mapped:
            return f"neck.sppcspc.{mapped}", True
        return yolo_key, False

    # Check RepConv
    if layer_idx in REP_MAP:
        mapped = map_repconv_keys(suffix)
        if mapped:
            return f"{REP_MAP[layer_idx]}.{mapped}", True
        return yolo_key, False

    # Check detection head
    if layer_idx in DETECT_MAP:
        mapped = map_detect_keys(suffix)
        if mapped:
            return f"detect.{mapped}", True
        return yolo_key, False

    return yolo_key, False


def convert_weights(input_path: str, output_path: str, verbose: bool = False) -> Dict[str, torch.Tensor]:
    """Convert YOLOv7 weights to LibreYOLO format."""
    print(f"Loading weights from {input_path}")
    weights = torch.load(input_path, map_location='cpu', weights_only=False)

    # Handle different formats
    if isinstance(weights, dict):
        if 'state_dict' in weights:
            state_dict = weights['state_dict']
        elif 'model' in weights:
            state_dict = weights['model']
        else:
            state_dict = weights
    else:
        state_dict = weights

    print(f"Found {len(state_dict)} keys in original weights")

    converted = {}
    skipped = []

    for yolo_key, value in state_dict.items():
        libre_key, success = convert_key(yolo_key)

        if success:
            converted[libre_key] = value
            if verbose:
                print(f"  {yolo_key} -> {libre_key}")
        else:
            skipped.append(yolo_key)

    print(f"\nConversion summary:")
    print(f"  Converted: {len(converted)} keys")
    print(f"  Skipped: {len(skipped)} keys")

    if skipped and verbose:
        print(f"\nSkipped keys:")
        for k in skipped[:20]:
            print(f"  {k}")

    # Save
    print(f"\nSaving converted weights to {output_path}")
    torch.save(converted, output_path)

    return converted


def verify_conversion(converted_path: str) -> bool:
    """Verify converted weights can be loaded."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from libreyolo.v7.nn import LibreYOLO7Model

    print(f"\nVerifying weights can be loaded...")

    converted = torch.load(converted_path, map_location='cpu', weights_only=False)
    model = LibreYOLO7Model(config='base', nb_classes=80)

    model_keys = set(model.state_dict().keys())
    converted_keys = set(converted.keys())

    matched = model_keys & converted_keys
    missing = model_keys - converted_keys
    extra = converted_keys - model_keys

    print(f"Model has {len(model_keys)} parameters")
    print(f"Converted weights have {len(converted_keys)} parameters")
    print(f"Matched: {len(matched)} ({100*len(matched)/len(model_keys):.1f}%)")

    if missing:
        print(f"\nMissing in converted ({len(missing)}):")
        for k in sorted(missing)[:10]:
            print(f"  {k}")

    if extra:
        print(f"\nExtra in converted ({len(extra)}):")
        for k in sorted(extra)[:10]:
            print(f"  {k}")

    try:
        result = model.load_state_dict(converted, strict=False)
        print(f"\nLoad result:")
        print(f"  Missing keys: {len(result.missing_keys)}")
        print(f"  Unexpected keys: {len(result.unexpected_keys)}")

        # Test forward pass
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 640, 640)
            out = model(x)
        print(f"\nForward pass OK! Output shape: {out['predictions'].shape}")

        return len(result.missing_keys) == 0
    except Exception as e:
        print(f"\nLoad failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLOv7 weights to LibreYOLO format")
    parser.add_argument("input", help="Path to YOLO weights")
    parser.add_argument("output", help="Path to save converted weights")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--verify", action="store_true")

    args = parser.parse_args()

    converted = convert_weights(args.input, args.output, args.verbose)

    if args.verify:
        verify_conversion(args.output)
