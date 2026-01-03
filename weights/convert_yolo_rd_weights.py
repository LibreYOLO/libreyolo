"""
Convert YOLO-RD weights from official YOLO repo format to LibreYOLO format.

Usage:
    python weights/convert_yolo_rd_weights.py weights/rd-9c.pt weights/libreyolo_rd.pt --verify

YOLO-RD is based on YOLOv9-c with DConv (Dynamic Convolution) at the B3 position.
The auxiliary path (layers 23-38) is skipped for inference.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch


# =============================================================================
# Layer Index to LibreYOLO Name Mapping
# =============================================================================

# Main detection path layers (0-22)
LAYER_MAP = {
    0: "backbone.conv0",      # Conv 3->64
    1: "backbone.conv1",      # Conv 64->128
    2: "backbone.elan1",      # RepNCSPELAN
    3: "backbone.down2",      # ADown
    4: "backbone.elan2",      # RepNCSPELAND (with DConv) - B3
    5: "backbone.down3",      # ADown
    6: "backbone.elan3",      # RepNCSPELAN - B4
    7: "backbone.down4",      # ADown
    8: "backbone.elan4",      # RepNCSPELAN - B5
    9: "backbone.spp",        # SPPELAN - N3
    # 10-11: Upsample, Concat (no params)
    12: "neck.elan_up1",      # RepNCSPELAN - N4
    # 13-14: Upsample, Concat (no params)
    15: "neck.elan_up2",      # RepNCSPELAN
    16: "neck.down1",         # ADown
    # 17: Concat (no params)
    18: "neck.elan_down1",    # RepNCSPELAN - P4
    19: "neck.down2",         # ADown
    # 20: Concat (no params)
    21: "neck.elan_down2",    # RepNCSPELAN - P5
    22: "detect",             # MultiheadDetection
}

# Auxiliary path layers (23-38) - skip for inference
AUXILIARY_LAYERS = set(range(23, 39))


# =============================================================================
# Sublayer Name Mapping
# =============================================================================

def map_conv_keys(yolo_suffix: str) -> str:
    """Map Conv layer keys. YOLO and LibreYOLO use same naming."""
    return yolo_suffix


def map_adown_keys(yolo_suffix: str) -> str:
    """Map ADown layer keys."""
    return yolo_suffix.replace("conv1", "cv1").replace("conv2", "cv2")


def map_repncspelan_keys(yolo_suffix: str) -> str:
    """Map RepNCSPELAN layer keys."""
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


def map_repncspelan_d_keys(yolo_suffix: str) -> str:
    """Map RepNCSPELAND layer keys (with DConv)."""
    # First apply RepNCSPELAN mapping
    result = map_repncspelan_keys(yolo_suffix)

    # DConv uses same naming (CG, GIE, D) - pass through
    return result


def map_sppelan_keys(yolo_suffix: str) -> str:
    """Map SPPELAN layer keys."""
    result = yolo_suffix
    result = result.replace("conv1.", "cv1.")
    result = result.replace("conv5.", "cv5.")
    return result


def map_detection_keys(yolo_suffix: str) -> Optional[str]:
    """Map MultiheadDetection layer keys."""
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

def get_layer_type(layer_idx: int) -> str:
    """Determine the layer type based on layer index."""
    if layer_idx in [0, 1]:
        return "conv"

    # RepNCSPELAN layers (no DConv)
    if layer_idx in [2, 6, 8, 12, 15, 18, 21]:
        return "repncspelan"

    # RepNCSPELAND (with DConv) - layer 4 only
    if layer_idx == 4:
        return "repncspelan_d"

    # Downsampling layers (ADown)
    if layer_idx in [3, 5, 7, 16, 19]:
        return "adown"

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

def convert_key(yolo_key: str) -> Tuple[str, bool]:
    """Convert a single YOLO key to LibreYOLO format."""
    # Parse layer index
    parts = yolo_key.split('.', 1)
    if len(parts) < 2:
        return yolo_key, False

    layer_idx_str, suffix = parts
    if not layer_idx_str.isdigit():
        return yolo_key, False

    layer_idx = int(layer_idx_str)

    # Skip auxiliary path layers
    if layer_idx in AUXILIARY_LAYERS:
        return yolo_key, False

    # Check if this layer is in our mapping
    if layer_idx not in LAYER_MAP:
        return yolo_key, False

    # Get the LibreYOLO prefix
    libre_prefix = LAYER_MAP[layer_idx]

    # Map the suffix based on layer type
    layer_type = get_layer_type(layer_idx)

    if layer_type == "conv":
        libre_suffix = map_conv_keys(suffix)
    elif layer_type == "adown":
        libre_suffix = map_adown_keys(suffix)
    elif layer_type == "repncspelan":
        libre_suffix = map_repncspelan_keys(suffix)
    elif layer_type == "repncspelan_d":
        libre_suffix = map_repncspelan_d_keys(suffix)
    elif layer_type == "sppelan":
        libre_suffix = map_sppelan_keys(suffix)
    elif layer_type == "detection":
        libre_suffix = map_detection_keys(suffix)
        if libre_suffix is None:
            return yolo_key, False
    else:
        return yolo_key, False

    return f"{libre_prefix}.{libre_suffix}", True


def convert_weights(input_path: str, output_path: str,
                    verbose: bool = False, reg_max: int = 16) -> Dict[str, torch.Tensor]:
    """Convert YOLO-RD weights to LibreYOLO format."""
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

    # Convert keys
    converted = {}
    skipped_aux = []
    failed = []

    for yolo_key, value in state_dict.items():
        libre_key, success = convert_key(yolo_key)

        if success:
            converted[libre_key] = value
            if verbose:
                print(f"  {yolo_key} -> {libre_key}")
        else:
            # Check if it's an auxiliary path layer
            parts = yolo_key.split('.', 1)
            if parts[0].isdigit() and int(parts[0]) in AUXILIARY_LAYERS:
                skipped_aux.append(yolo_key)
            else:
                failed.append(yolo_key)

    print(f"\nConversion summary:")
    print(f"  Converted: {len(converted)} keys")
    print(f"  Skipped (auxiliary path): {len(skipped_aux)} keys")
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


def verify_conversion(converted_path: str, atoms: int = 512) -> bool:
    """Verify converted weights can be loaded into LibreYOLO model."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from libreyolo.rd.nn import LibreYOLORDModel

    print(f"\nVerifying weights can be loaded (atoms={atoms})...")

    # Load converted weights
    converted = torch.load(converted_path, map_location='cpu', weights_only=False)

    # Create model with appropriate atoms
    model = LibreYOLORDModel(config='c', reg_max=16, nb_classes=80, atoms=atoms)
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
    parser = argparse.ArgumentParser(description="Convert YOLO-RD weights to LibreYOLO format")
    parser.add_argument("input", help="Path to YOLO weights (.pt)")
    parser.add_argument("output", help="Path to save converted weights")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed info")
    parser.add_argument("--verify", action="store_true", help="Verify converted weights")
    parser.add_argument("--atoms", type=int, default=512, help="DConv atoms (512 for rd-9c, 4096 for rd-9c-4096)")

    args = parser.parse_args()

    converted = convert_weights(args.input, args.output, args.verbose)

    if args.verify:
        verify_conversion(args.output, atoms=args.atoms)
