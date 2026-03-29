"""Family-specific default overrides and config building for the CLI."""

from typing import Any, Optional

import click

# Family-specific defaults using CLI-facing names (ultralytics vocabulary).
# These override base TrainConfig defaults when the user doesn't explicitly
# set a parameter.

FAMILY_TRAIN_DEFAULTS: dict[str, dict[str, Any]] = {
    "yolox": {
        "momentum": 0.9,
        "warmup_epochs": 5,
        "warmup_lr_start": 0.0,
        "min_lr_ratio": 0.05,
        "degrees": 10.0,
        "shear": 2.0,
        "mosaic_scale": (0.1, 2.0),
        "mosaic": 1.0,
        "mixup": 1.0,
        "ema_decay": 0.9998,
        "name": "exp",
    },
    "yolo9": {
        "scheduler": "linear",
        "warmup_epochs": 3,
        "warmup_lr_start": 0.0001,
        "min_lr_ratio": 0.01,
        "degrees": 0.0,
        "shear": 0.0,
        "mosaic_scale": (0.5, 1.5),
        "mixup": 0.0,
        "ema_decay": 0.9999,
        "name": "yolo9_exp",
        "workers": 8,
    },
    "rfdetr": {
        "epochs": 100,
        "batch": 4,
        "lr0": 0.0001,
        "optimizer": "adamw",
        "name": "rfdetr_exp",
    },
}

# RF-DETR does not support these augmentation/scheduler parameters.
# They are warned and ignored rather than errored.
RFDETR_UNSUPPORTED_PARAMS: set[str] = {
    "mosaic",
    "mixup",
    "degrees",
    "shear",
    "scheduler",
    "warmup_epochs",
    "warmup_lr_start",
    "min_lr_ratio",
    "mosaic_scale",
    "mixup_scale",
    "no_aug_epochs",
    "momentum",
    "nesterov",
    "ema",
    "ema_decay",
    "hsv_prob",
    "flip_prob",
    "translate",
}


# Maps CLI model names (e.g. "yolox-s") to weight filenames (e.g. "LibreYOLOXs.pt").
_CLI_NAME_TO_WEIGHTS: dict[str, str] = {}


def _build_name_map() -> None:
    """Populate CLI name → weight filename mapping from model registry."""
    if _CLI_NAME_TO_WEIGHTS:
        return
    from libreyolo.models.base.model import BaseModel

    for cls in BaseModel._registry:
        for size_code in cls.INPUT_SIZES:
            cli_name = f"{cls.FAMILY}-{size_code}"
            filename = f"{cls.FILENAME_PREFIX}{size_code}{cls.WEIGHT_EXT}"
            _CLI_NAME_TO_WEIGHTS[cli_name] = filename

    # Also try RF-DETR (lazily registered)
    try:
        from libreyolo.models.rfdetr.model import LibreYOLORFDETR as rfcls

        for size_code in rfcls.INPUT_SIZES:
            cli_name = f"{rfcls.FAMILY}-{size_code}"
            filename = f"{rfcls.FILENAME_PREFIX}{size_code}{rfcls.WEIGHT_EXT}"
            _CLI_NAME_TO_WEIGHTS[cli_name] = filename
    except ImportError:
        pass


def resolve_model_name(model: str) -> str:
    """Resolve a CLI model name to a weight filename or passthrough.

    ``yolox-s`` → ``LibreYOLOXs.pt``
    ``best.pt`` → ``best.pt`` (unchanged)
    """
    _build_name_map()
    return _CLI_NAME_TO_WEIGHTS.get(model.lower(), model)


def detect_family_from_name(model_name: str) -> Optional[str]:
    """Detect model family from a CLI model name like 'yolox-s' or 'yolo9-m'."""
    lower = model_name.lower()
    if lower.startswith("yolox-"):
        return "yolox"
    if lower.startswith("yolo9-"):
        return "yolo9"
    if lower.startswith("rfdetr-"):
        return "rfdetr"
    return None


def is_user_provided(param_name: str) -> bool:
    """Check if a parameter was explicitly provided by the user (not defaulted)."""
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return False
    source = ctx.get_parameter_source(param_name)
    return source == click.core.ParameterSource.COMMANDLINE


def apply_family_defaults(
    params: dict[str, Any], family: str, mode: str
) -> dict[str, Any]:
    """Apply family-specific defaults to parameters that weren't explicitly set.

    Only overrides values that came from Typer defaults (not user input).
    """
    if mode != "train":
        return params

    overrides = FAMILY_TRAIN_DEFAULTS.get(family, {})
    result = dict(params)
    for key, default_value in overrides.items():
        if key in result and not is_user_provided(key):
            result[key] = default_value
    return result
