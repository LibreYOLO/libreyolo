"""Mode-aware alias resolution for CLI parameter names.

Translates user-facing ultralytics names to internal library field names.
This is the single source of truth for name translation.
"""

TRAIN_ALIASES: dict[str, str] = {
    "mosaic": "mosaic_prob",
    "mixup": "mixup_prob",
}

VAL_ALIASES: dict[str, str] = {
    "batch": "batch_size",
    "conf": "conf_thres",
    "iou": "iou_thres",
    "workers": "num_workers",
}

# Predict and export use native parameter names — no aliases needed.

_MODE_ALIASES: dict[str, dict[str, str]] = {
    "train": TRAIN_ALIASES,
    "val": VAL_ALIASES,
}


def resolve_aliases(overrides: dict, mode: str) -> dict:
    """Translate CLI-facing keys to internal config field names.

    Args:
        overrides: Dict of CLI parameter names and values.
        mode: Command mode ("train", "val", "predict", "export").

    Returns:
        Dict with internal field names.
    """
    aliases = _MODE_ALIASES.get(mode, {})
    resolved = {}
    for key, value in overrides.items():
        resolved[aliases.get(key, key)] = value
    return resolved
