"""Family-specific config discovery and model name resolution for the CLI.

Config defaults come from the dataclass source of truth (TrainConfig subclasses).
The CLI discovers them via BaseModel._registry → TRAIN_CONFIG, so adding a new
model family requires zero CLI changes.
"""

from dataclasses import MISSING, fields
from typing import Any, Optional

import click


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


# =========================================================================
# Model name resolution
# =========================================================================

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
    from libreyolo.models import try_ensure_rfdetr

    rfcls = try_ensure_rfdetr()
    if rfcls is not None:
        for size_code in rfcls.INPUT_SIZES:
            cli_name = f"{rfcls.FAMILY}-{size_code}"
            filename = f"{rfcls.FILENAME_PREFIX}{size_code}{rfcls.WEIGHT_EXT}"
            _CLI_NAME_TO_WEIGHTS[cli_name] = filename


def resolve_model_name(model: str) -> str:
    """Resolve a CLI model name to a weight filename or passthrough.

    ``yolox-s`` → ``LibreYOLOXs.pt``
    ``best.pt`` → ``best.pt`` (unchanged)
    """
    _build_name_map()
    return _CLI_NAME_TO_WEIGHTS.get(model.lower(), model)


def detect_family_from_name(model_name: str) -> Optional[str]:
    """Detect model family from a CLI model name like 'yolox-s' or 'yolo9-m'."""
    _build_name_map()
    lower = model_name.lower()
    # Check against all registered families (auto-discovered)
    for cli_name in _CLI_NAME_TO_WEIGHTS:
        family = cli_name.rsplit("-", 1)[0]
        if lower.startswith(f"{family}-"):
            return family
    return None


# =========================================================================
# Config class discovery via model registry
# =========================================================================


def get_train_config_class(family: str) -> type:
    """Look up the TrainConfig subclass for a model family from the registry.

    Returns the base TrainConfig if the family has no specific config.
    """
    from libreyolo.models.base.model import BaseModel
    from libreyolo.training.config import TrainConfig

    for cls in BaseModel._registry:
        if cls.FAMILY == family and cls.TRAIN_CONFIG is not None:
            return cls.TRAIN_CONFIG

    # Check RF-DETR (lazily registered)
    from libreyolo.models import try_ensure_rfdetr

    rfcls = try_ensure_rfdetr()
    if rfcls is not None and rfcls.FAMILY == family and rfcls.TRAIN_CONFIG is not None:
        return rfcls.TRAIN_CONFIG

    return TrainConfig


def get_family_defaults(family: str) -> dict[str, Any]:
    """Get family-specific training defaults from the config dataclass.

    Returns a dict of {field_name: default_value} for fields where the
    family config differs from the base TrainConfig.
    """
    from libreyolo.training.config import TrainConfig

    config_cls = get_train_config_class(family)
    if config_cls is TrainConfig:
        return {}

    base = TrainConfig()
    family_cfg = config_cls()

    # Find fields where the family default differs from the base default
    diffs = {}
    for f in fields(config_cls):
        base_val = getattr(base, f.name)
        family_val = getattr(family_cfg, f.name)
        if base_val != family_val:
            diffs[f.name] = family_val
    return diffs


# =========================================================================
# Parameter source detection
# =========================================================================


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

    Discovers defaults from the model's TRAIN_CONFIG dataclass — no hardcoded
    dicts. Only overrides values that came from Typer defaults (not user input).
    """
    if mode != "train":
        return params

    family_diffs = get_family_defaults(family)
    if not family_diffs:
        return params

    # Reverse alias map: internal name → CLI name (for is_user_provided check)
    from .aliases import TRAIN_ALIASES

    internal_to_cli = {v: k for k, v in TRAIN_ALIASES.items()}

    result = dict(params)
    for internal_name, default_value in family_diffs.items():
        # The params dict uses CLI-facing names, so check both
        cli_name = internal_to_cli.get(internal_name, internal_name)
        if cli_name in result and not is_user_provided(cli_name):
            result[cli_name] = default_value
    return result


# =========================================================================
# Cfg defaults (auto-discovered from config dataclasses)
# =========================================================================


def _to_json_safe(val: Any) -> Any:
    """Convert tuples to lists for JSON serialization."""
    return list(val) if isinstance(val, tuple) else val


def get_cfg_defaults() -> dict[str, Any]:
    """Build configuration defaults from dataclasses for the cfg command.

    All values are derived from TrainConfig and ValidationConfig — nothing
    hardcoded.  Family overrides are auto-discovered from the model registry.
    """
    from libreyolo.models.base.model import BaseModel
    from libreyolo.models import try_ensure_rfdetr
    from libreyolo.training.config import TrainConfig
    from libreyolo.validation.config import ValidationConfig
    from .aliases import TRAIN_ALIASES, VAL_ALIASES

    train_internal_to_cli = {v: k for k, v in TRAIN_ALIASES.items()}
    val_internal_to_cli = {v: k for k, v in VAL_ALIASES.items()}

    # -- Train defaults from TrainConfig() -----------------------------------
    train_exclude = {"size", "num_classes", "data", "data_dir", "device"}
    base = TrainConfig()
    train_defaults = {}
    for f in fields(TrainConfig):
        if f.name in train_exclude:
            continue
        cli_name = train_internal_to_cli.get(f.name, f.name)
        train_defaults[cli_name] = _to_json_safe(getattr(base, f.name))

    # -- Val defaults from ValidationConfig field defaults --------------------
    #    (Can't instantiate — __post_init__ requires data.)
    val_exclude = {"data", "data_dir", "device", "save_dir", "iou_thresholds"}
    val_defaults = {}
    for f in fields(ValidationConfig):
        if f.name in val_exclude or f.default is MISSING:
            continue
        cli_name = val_internal_to_cli.get(f.name, f.name)
        val_defaults[cli_name] = _to_json_safe(f.default)

    # -- Predict defaults (no backing dataclass) -----------------------------
    predict_defaults: dict[str, Any] = {
        "conf": 0.25,
        "iou": 0.45,
        "batch": 1,
        "imgsz": None,
    }

    # -- Family overrides (auto-discovered from registry) --------------------
    family_overrides: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for cls in BaseModel._registry:
        if cls.FAMILY in seen:
            continue
        seen.add(cls.FAMILY)
        diffs = get_family_defaults(cls.FAMILY)
        if diffs:
            family_overrides[cls.FAMILY] = {
                train_internal_to_cli.get(k, k): _to_json_safe(v)
                for k, v in diffs.items()
            }

    rfcls = try_ensure_rfdetr()
    if rfcls is not None and rfcls.FAMILY not in seen:
        diffs = get_family_defaults(rfcls.FAMILY)
        if diffs:
            family_overrides[rfcls.FAMILY] = {
                train_internal_to_cli.get(k, k): _to_json_safe(v)
                for k, v in diffs.items()
            }

    return {
        "train_defaults": train_defaults,
        "val_defaults": val_defaults,
        "predict_defaults": predict_defaults,
        "family_overrides": family_overrides,
    }
