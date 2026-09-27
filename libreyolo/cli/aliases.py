"""Mode-aware alias resolution for CLI parameter names.

Translates user-facing shorthand names to internal library field names.
This is the single source of truth for name translation.
"""

TRAIN_ALIASES: dict[str, str] = {
    "mosaic": "mosaic_prob",
    "mixup": "mixup_prob",
}

# Classification models train through the ImageFolder pipeline, where the
# ecosystem ``mixup`` knob is the batch-MixUp field of the same name (soft
# labels), not the detection ``mixup_prob``. Task-aware lookup lives in
# :func:`train_aliases`; consumers that map CLI names to TrainConfig fields
# go through it rather than reading TRAIN_ALIASES directly.
CLASSIFY_TRAIN_ALIASES: dict[str, str] = {
    "mosaic": "mosaic_prob",
}


def train_aliases(task: str | None = None) -> dict[str, str]:
    """Return the CLI-name -> TrainConfig-field alias table for ``task``."""
    if task == "classify":
        return CLASSIFY_TRAIN_ALIASES
    return TRAIN_ALIASES

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
