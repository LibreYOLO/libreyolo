"""
Convert YOLOv9 weights from official YOLO repo format to LibreYOLO format.

Usage:
    python weights/convert_yolo9_weights.py weights/v9-t.pt weights/libreyolo9t.pt --config t
    python weights/convert_yolo9_weights.py weights/v9-s.pt weights/libreyolo9s.pt --config s
    python weights/convert_yolo9_weights.py weights/v9-m.pt weights/libreyolo9m.pt --config m
    python weights/convert_yolo9_weights.py weights/v9-c.pt weights/libreyolo9c.pt --config c

The YOLO repo uses numbered layer indices (0., 1., 2., etc.) while LibreYOLO
uses semantic naming (backbone.conv0, neck.elan_up1, etc.).

This script handles the mapping between the two naming conventions for all variants.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch


# =============================================================================
# Layer Index Mapping (YOLO layer index -> LibreYOLO prefix)
# =============================================================================

# Common layers across all variants
COMMON_LAYERS = {
    0: "backbone.conv0",      # Conv 3->X
    1: "backbone.conv1",      # Conv X->Y
}

# v9-t and v9-s: ELAN first block, AConv downsampling
V9_TS_LAYER_MAP = {
    **COMMON_LAYERS,
    2: "backbone.elan1",      # ELAN
    3: "backbone.down2",      # AConv
    4: "backbone.elan2",      # RepNCSPELAN
    5: "backbone.down3",      # AConv
    6: "backbone.elan3",      # RepNCSPELAN
    7: "backbone.down4",      # AConv
    8: "backbone.elan4",      # RepNCSPELAN
    9: "backbone.spp",        # SPPELAN
    # Neck
    12: "neck.elan_up1",      # RepNCSPELAN (N4)
    15: "neck.elan_up2",      # RepNCSPELAN (P3)
    16: "neck.down1",         # AConv
    18: "neck.elan_down1",    # RepNCSPELAN (P4)
    19: "neck.down2",         # AConv
    21: "neck.elan_down2",    # RepNCSPELAN (P5)
    # Detection head
    22: "detect",             # MultiheadDetection
}

# v9-m: RepNCSPELAN first block, AConv downsampling
V9_M_LAYER_MAP = {
    **COMMON_LAYERS,
    2: "backbone.elan1",      # RepNCSPELAN
    3: "backbone.down2",      # AConv
    4: "backbone.elan2",      # RepNCSPELAN
    5: "backbone.down3",      # AConv
    6: "backbone.elan3",      # RepNCSPELAN
    7: "backbone.down4",      # AConv
    8: "backbone.elan4",      # RepNCSPELAN
    9: "backbone.spp",        # SPPELAN
    # Neck
    12: "neck.elan_up1",      # RepNCSPELAN (N4)
    15: "neck.elan_up2",      # RepNCSPELAN (P3)
    16: "neck.down1",         # AConv
    18: "neck.elan_down1",    # RepNCSPELAN (P4)
    19: "neck.down2",         # AConv
    21: "neck.elan_down2",    # RepNCSPELAN (P5)
    # Detection head
    22: "detect",             # MultiheadDetection
}

# v9-c: RepNCSPELAN first block, ADown downsampling
V9_C_LAYER_MAP = {
    **COMMON_LAYERS,
    2: "backbone.elan1",      # RepNCSPELAN
    3: "backbone.down2",      # ADown
    4: "backbone.elan2",      # RepNCSPELAN
    5: "backbone.down3",      # ADown
    6: "backbone.elan3",      # RepNCSPELAN
    7: "backbone.down4",      # ADown
    8: "backbone.elan4",      # RepNCSPELAN
    9: "backbone.spp",        # SPPELAN
    # Neck
    12: "neck.elan_up1",      # RepNCSPELAN (N4)
    15: "neck.elan_up2",      # RepNCSPELAN (P3)
    16: "neck.down1",         # ADown
    18: "neck.elan_down1",    # RepNCSPELAN (P4)
    19: "neck.down2",         # ADown
    21: "neck.elan_down2",    # RepNCSPELAN (P5)
    # Detection head
    22: "detect",             # MultiheadDetection
}

LAYER_MAPS = {
    't': V9_TS_LAYER_MAP,
    's': V9_TS_LAYER_MAP,
    'm': V9_M_LAYER_MAP,
    'c': V9_C_LAYER_MAP,
}


# =============================================================================
# Sublayer Name Mapping
# =============================================================================

def map_conv_keys(yolo_suffix: str) -> str:
    """Map Conv layer keys. YOLO and LibreYOLO use same naming."""
    return yolo_suffix


def map_aconv_keys(yolo_suffix: str) -> str:
    """Map AConv layer keys.

    YOLO AConv: conv.conv, conv.bn
    LibreYOLO AConv: cv.conv, cv.bn

    Only replace the first 'conv.' with 'cv.'
    """
    return re.sub(r'^conv\.', 'cv.', yolo_suffix)


def map_adown_keys(yolo_suffix: str) -> str:
    """Map ADown layer keys.

    YOLO: conv1, conv2
    LibreYOLO: cv1, cv2
    """
    return yolo_suffix.replace("conv1", "cv1").replace("conv2", "cv2")


def map_elan_keys(yolo_suffix: str) -> str:
    """Map ELAN layer keys (for v9-t/s first block).

    YOLO ELAN: conv1, conv2, conv3, conv4
    LibreYOLO ELAN: cv1, cv2, cv3, cv4
    """
    result = yolo_suffix
    result = re.sub(r'^conv([1234])\.', r'cv\1.', result)
    return result


def map_repncspelan_keys(yolo_suffix: str) -> str:
    """Map RepNCSPELAN layer keys.

    YOLO: conv1, conv2, conv3, conv4 with nested bottleneck structure
    LibreYOLO: cv1, cv2, cv3, cv4 with nested m (bottleneck) structure
    """
    result = yolo_suffix

    # Map main conv names: conv1/2/3/4 -> cv1/2/3/4
    result = re.sub(r'^conv([1234])\.', r'cv\1.', result)

    # Map RepNCSP internal names (inside cv2.0 and cv3.0)
    result = re.sub(r'\.conv([123])\.', r'.cv\1.', result)

    # Map bottleneck -> m
    result = result.replace('.bottleneck.', '.m.')

    # Inside RepNCSP Bottleneck, YOLO uses conv1/conv2
    # LibreYOLO's RepNBottleneck uses cv1/cv2
    result = re.sub(r'\.m\.(\d+)\.conv([12])\.', r'.m.\1.cv\2.', result)

    return result


def map_sppelan_keys(yolo_suffix: str) -> str:
    """Map SPPELAN layer keys.

    YOLO: conv1, pools.0/1/2, conv5
    LibreYOLO: cv1, pools.0/1/2, cv5
    """
    result = yolo_suffix
    result = result.replace("conv1.", "cv1.")
    result = result.replace("conv5.", "cv5.")
    return result


def map_detection_keys(yolo_suffix: str) -> Optional[str]:
    """Map MultiheadDetection layer keys.

    YOLO: heads.N.anchor_conv, heads.N.class_conv, heads.N.anc2vec
    LibreYOLO: cv2.N (box), cv3.N (class), dfl
    """
    result = yolo_suffix

    # Map heads.N.anchor_conv -> cv2.N
    result = re.sub(r'^heads\.(\d+)\.anchor_conv\.', r'cv2.\1.', result)

    # Map heads.N.class_conv -> cv3.N
    result = re.sub(r'^heads\.(\d+)\.class_conv\.', r'cv3.\1.', result)

    # Skip anc2vec (incompatible shapes with DFL)
    if 'anc2vec' in result:
        return None

    return result


# =============================================================================
# Layer Type Detection
# =============================================================================

def get_layer_type(layer_idx: int, config: str) -> str:
    """Determine the layer type based on layer index and config."""
    if layer_idx in [0, 1]:
        return "conv"

    # First block (layer 2)
    if layer_idx == 2:
        if config in ['t', 's']:
            return "elan"
        else:
            return "repncspelan"

    # Downsampling layers
    if layer_idx in [3, 5, 7, 16, 19]:
        if config == 'c':
            return "adown"
        else:
            return "aconv"

    # RepNCSPELAN layers
    if layer_idx in [4, 6, 8, 12, 15, 18, 21]:
        return "repncspelan"

    # SPPELAN
    if layer_idx == 9:
        return "sppelan"

    # Detection head
    if layer_idx == 22:
        return "detection"

    return "unknown"


# =============================================================================
# Main Conversion Logic
# =============================================================================

def convert_key(yolo_key: str, config: str) -> Tuple[str, bool]:
    """Convert a single YOLO key to LibreYOLO format.

    Returns:
        Tuple of (converted_key, success)
    """
    layer_map = LAYER_MAPS[config]

    # Parse layer index
    parts = yolo_key.split('.', 1)
    if len(parts) < 2:
        return yolo_key, False

    layer_idx_str, suffix = parts
    if not layer_idx_str.isdigit():
        return yolo_key, False

    layer_idx = int(layer_idx_str)

    # Check if this layer is in our mapping
    if layer_idx not in layer_map:
        return yolo_key, False

    # Get the LibreYOLO prefix
    libre_prefix = layer_map[layer_idx]

    # Map the suffix based on layer type
    layer_type = get_layer_type(layer_idx, config)

    if layer_type == "conv":
        libre_suffix = map_conv_keys(suffix)
    elif layer_type == "aconv":
        libre_suffix = map_aconv_keys(suffix)
    elif layer_type == "adown":
        libre_suffix = map_adown_keys(suffix)
    elif layer_type == "elan":
        libre_suffix = map_elan_keys(suffix)
    elif layer_type == "repncspelan":
        libre_suffix = map_repncspelan_keys(suffix)
    elif layer_type == "sppelan":
        libre_suffix = map_sppelan_keys(suffix)
    elif layer_type == "detection":
        libre_suffix = map_detection_keys(suffix)
        if libre_suffix is None:
            return yolo_key, False
    else:
        return yolo_key, False

    return f"{libre_prefix}.{libre_suffix}", True


def convert_weights(input_path: str, output_path: str, config: str,
                    verbose: bool = False, reg_max: int = 16) -> Dict[str, torch.Tensor]:
    """Convert YOLO v9 weights to LibreYOLO format.

    Args:
        input_path: Path to YOLO weights (.pt file)
        output_path: Path to save converted weights
        config: Model config ('t', 's', 'm', 'c')
        verbose: Print detailed conversion info
        reg_max: Regression max value for DFL layer

    Returns:
        Converted state dict
    """
    print(f"Loading weights from {input_path}")
    weights = torch.load(input_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(weights, dict):
        if 'state_dict' in weights:
            state_dict = {k.replace('model.model.', ''): v
                         for k, v in weights['state_dict'].items()}
        elif 'model' in weights:
            state_dict = weights['model']
        else:
            state_dict = weights
    else:
        state_dict = weights

    print(f"Found {len(state_dict)} keys in original weights")
    print(f"Converting for config: v9-{config}")

    # Convert keys
    converted = {}
    skipped = []
    failed = []

    for yolo_key, value in state_dict.items():
        libre_key, success = convert_key(yolo_key, config)

        if success:
            converted[libre_key] = value
            if verbose:
                print(f"  {yolo_key} -> {libre_key}")
        else:
            # Check if it's an auxiliary head (layer 23+)
            parts = yolo_key.split('.', 1)
            if parts[0].isdigit() and int(parts[0]) >= 23:
                skipped.append(yolo_key)
            else:
                failed.append(yolo_key)

    print(f"\nConversion summary:")
    print(f"  Converted: {len(converted)} keys")
    print(f"  Skipped (auxiliary head): {len(skipped)} keys")
    print(f"  Failed: {len(failed)} keys")

    if failed and verbose:
        print(f"\nFailed keys:")
        for k in failed[:20]:
            print(f"  {k}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

    # Add DFL fixed weights
    dfl_weight = torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1)
    converted['detect.dfl.conv.weight'] = dfl_weight
    print(f"  Added DFL fixed weights (detect.dfl.conv.weight)")

    # Save converted weights
    print(f"\nSaving converted weights to {output_path}")
    torch.save(converted, output_path)

    return converted


def verify_conversion(converted_path: str, config: str) -> bool:
    """Verify converted weights can be loaded into LibreYOLO model."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from libreyolo.v9.nn import LibreYOLO9Model

    print(f"\nVerifying weights can be loaded into v9-{config} model...")

    # Load converted weights
    converted = torch.load(converted_path, map_location='cpu', weights_only=False)

    # Create model
    model = LibreYOLO9Model(config=config, reg_max=16, nb_classes=80)
    model_keys = set(model.state_dict().keys())
    converted_keys = set(converted.keys())

    # Check coverage
    matched = model_keys & converted_keys
    missing_in_converted = model_keys - converted_keys
    extra_in_converted = converted_keys - model_keys

    print(f"Model has {len(model_keys)} parameters")
    print(f"Converted weights have {len(converted_keys)} parameters")
    print(f"Matched: {len(matched)} ({100*len(matched)/len(model_keys):.1f}%)")

    if missing_in_converted:
        print(f"\nMissing in converted weights ({len(missing_in_converted)}):")
        for k in sorted(missing_in_converted)[:10]:
            print(f"  {k}")
        if len(missing_in_converted) > 10:
            print(f"  ... and {len(missing_in_converted) - 10} more")

    if extra_in_converted:
        print(f"\nExtra in converted weights ({len(extra_in_converted)}):")
        for k in sorted(extra_in_converted)[:10]:
            print(f"  {k}")
        if len(extra_in_converted) > 10:
            print(f"  ... and {len(extra_in_converted) - 10} more")

    # Try loading
    try:
        result = model.load_state_dict(converted, strict=False)
        print(f"\nLoad result:")
        print(f"  Missing keys: {len(result.missing_keys)}")
        print(f"  Unexpected keys: {len(result.unexpected_keys)}")

        if result.missing_keys:
            print(f"\n  Sample missing keys:")
            for k in result.missing_keys[:5]:
                print(f"    {k}")

        return len(result.missing_keys) == 0
    except Exception as e:
        print(f"\nLoad failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLOv9 weights to LibreYOLO format")
    parser.add_argument("input", help="Path to YOLO weights (.pt)")
    parser.add_argument("output", help="Path to save converted weights")
    parser.add_argument("--config", required=True, choices=['t', 's', 'm', 'c'],
                       help="Model config (required)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed info")
    parser.add_argument("--verify", action="store_true", help="Verify converted weights")

    args = parser.parse_args()

    converted = convert_weights(args.input, args.output, args.config, args.verbose)

    if args.verify:
        verify_conversion(args.output, args.config)
