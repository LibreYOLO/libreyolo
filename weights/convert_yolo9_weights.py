"""
Convert YOLOv9 weights from the upstream YOLO repo format to LibreYOLO format.

Usage:
    python weights/convert_yolo9_weights.py weights/v9-t.pt weights/LibreYOLO9t.pt --config t
    python weights/convert_yolo9_weights.py weights/v9-s.pt weights/LibreYOLO9s.pt
    python weights/convert_yolo9_weights.py weights/v9-m.pt weights/LibreYOLO9m.pt
    python weights/convert_yolo9_weights.py weights/v9-c.pt weights/LibreYOLO9c.pt

The upstream repo uses numbered layer indices (0., 1., 2., etc.) while LibreYOLO
uses semantic naming (backbone.conv0, neck.elan_up1, etc.). The shared remapping
logic lives in ``libreyolo.models.yolo9.convert`` so this script and the runtime
auto-conversion path stay in sync. ``--config`` is inferred from the checkpoint
when omitted, and the class count is read from the upstream detection head so
fine-tuned (non-COCO) checkpoints convert correctly.
"""

import argparse

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    strip_state_dict_prefix,
    wrap_libreyolo_checkpoint,
)

add_repo_root_to_path()

from libreyolo.models.yolo9.convert import (  # noqa: E402
    SUPPORTED_CONFIGS,
    convert_state_dict,
    infer_config,
    infer_nb_classes,
)


def _extract_upstream_state_dict(weights) -> dict:
    """Return the upstream tensor state dict across known checkpoint layouts."""
    if isinstance(weights, dict) and "state_dict" in weights:
        return strip_state_dict_prefix(
            extract_state_dict(
                weights, state_dict_keys=("state_dict",), prefer_ema=False
            ),
            "model.model.",
        )
    return extract_state_dict(weights, state_dict_keys=("model",), prefer_ema=False)


def _extract_names(weights, nc: int) -> object | None:
    """Return class names from top-level or args metadata, when present."""
    if not isinstance(weights, dict):
        return None
    names = weights.get("names")
    if names is not None:
        return names

    args = weights.get("args") or weights.get("hyper_parameters") or {}
    class_names = (
        args.get("class_names")
        if isinstance(args, dict)
        else getattr(args, "class_names", None)
    )
    if class_names is None:
        return None
    if isinstance(class_names, dict):
        names = {int(key): str(value) for key, value in class_names.items()}
        return {key: value for key, value in names.items() if key < nc}

    return list(class_names)[:nc]


def convert_weights(
    input_path: str,
    output_path: str,
    config: str | None = None,
    verbose: bool = False,
) -> dict:
    """Convert upstream YOLO9 weights to a LibreYOLO v1.0 checkpoint on disk."""
    print(f"Loading weights from {input_path}")
    weights = load_checkpoint(input_path)
    state_dict = _extract_upstream_state_dict(weights)
    print(f"Found {len(state_dict)} keys in original weights")

    if config is None:
        config = infer_config(state_dict)
        if config is None:
            raise ValueError(
                "Could not infer YOLO9 config from the checkpoint; "
                f"pass --config explicitly (one of {SUPPORTED_CONFIGS})."
            )
        print(f"Inferred config: yolo9-{config}")
    print(f"Converting for config: yolo9-{config}")

    converted, stats = convert_state_dict(state_dict, config)

    print("\nConversion summary:")
    print(f"  Converted: {stats['converted']} keys")
    print(f"  Skipped (auxiliary head): {stats['skipped']} keys")
    print(f"  Failed: {stats['failed']} keys")
    print("  DFL projection is model-derived; no fixed DFL weights added")

    nc = infer_nb_classes(state_dict) or 80
    names = _extract_names(weights, nc)
    print(f"  Class count (nc): {nc}")

    print(f"\nSaving converted weights to {output_path}")
    wrapped = wrap_libreyolo_checkpoint(
        converted, model_family="yolo9", size=config, nc=nc, names=names,
    )
    save_checkpoint(wrapped, output_path)

    return converted


def verify_conversion(converted_path: str, config: str) -> bool:
    """Verify converted weights can be loaded into the LibreYOLO model."""
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    print(f"\nVerifying weights can be loaded into yolo9-{config} model...")

    raw = load_checkpoint(converted_path)
    converted = extract_state_dict(raw)
    # Read nc from the wrapped metadata when present.
    nc = raw.get("nc", 80) if isinstance(raw, dict) else 80

    model = LibreYOLO9Model(config=config, reg_max=16, nb_classes=nc)
    model_keys = set(model.state_dict().keys())
    converted_keys = set(converted.keys())

    matched = model_keys & converted_keys
    missing_in_converted = model_keys - converted_keys
    extra_in_converted = converted_keys - model_keys

    print(f"Model has {len(model_keys)} parameters")
    print(f"Converted weights have {len(converted_keys)} parameters")
    print(f"Matched: {len(matched)} ({100 * len(matched) / len(model_keys):.1f}%)")

    if missing_in_converted:
        print(f"\nMissing in converted weights ({len(missing_in_converted)}):")
        for k in sorted(missing_in_converted)[:10]:
            print(f"  {k}")
    if extra_in_converted:
        print(f"\nExtra in converted weights ({len(extra_in_converted)}):")
        for k in sorted(extra_in_converted)[:10]:
            print(f"  {k}")

    result = model.load_state_dict(converted, strict=False)
    print("\nLoad result:")
    print(f"  Missing keys: {len(result.missing_keys)}")
    print(f"  Unexpected keys: {len(result.unexpected_keys)}")
    return len(result.missing_keys) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert YOLOv9 weights to LibreYOLO format"
    )
    parser.add_argument("input", help="Path to YOLO weights (.pt)")
    parser.add_argument("output", help="Path to save converted weights")
    parser.add_argument(
        "--config",
        choices=list(SUPPORTED_CONFIGS),
        default=None,
        help="Model config (inferred from the checkpoint when omitted)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print detailed info"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify converted weights"
    )

    args = parser.parse_args()

    convert_weights(args.input, args.output, args.config, args.verbose)

    if args.verify:
        # Re-read the config we actually used (inferred or explicit).
        used_config = args.config or infer_config(
            _extract_upstream_state_dict(load_checkpoint(args.input))
        )
        verify_conversion(args.output, used_config)
