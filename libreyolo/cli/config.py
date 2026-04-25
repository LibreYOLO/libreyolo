"""Family-specific config discovery and model name resolution for the CLI.

Config defaults come from the dataclass source of truth (TrainConfig subclasses).
The CLI discovers them via BaseModel._registry → TRAIN_CONFIG, so adding a new
model family requires zero CLI changes.
"""

from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Optional

import click


# RF-DETR does not support these augmentation/scheduler parameters.
# They are warned and ignored rather than errored.
RFDETR_UNSUPPORTED_PARAMS: set[str] = {
    "imgsz",
    "mosaic",
    "mixup",
    "degrees",
    "shear",
    "scheduler",
    "warmup_lr_start",
    "min_lr_ratio",
    "mosaic_scale",
    "mixup_scale",
    "no_aug_epochs",
    "optimizer",
    "momentum",
    "nesterov",
    "hsv_prob",
    "flip_prob",
    "translate",
    "amp",
    "pretrained",
    "log_interval",
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


def get_all_cli_names() -> list[str]:
    """Return all valid CLI model names (e.g. ['yolox-n', 'yolox-s', ...])."""
    _build_name_map()
    return list(_CLI_NAME_TO_WEIGHTS.keys())


def is_known_weight_filename(model: str) -> bool:
    """Return whether a path or filename matches a known packaged weight name."""
    _build_name_map()
    filename = Path(model).name.lower()
    return any(Path(weight).name.lower() == filename for weight in _CLI_NAME_TO_WEIGHTS.values())


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
        if not hasattr(base, f.name):
            continue
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
# Train kwargs builder (auto-discovered from TrainConfig)
# =========================================================================


def build_train_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """Build training kwargs from CLI params, keyed by TrainConfig field names.

    Iterates TrainConfig fields and maps CLI-facing parameter names to
    internal field names using TRAIN_ALIASES.  Adding a new field to
    TrainConfig automatically makes it available — no manual dict needed.
    """
    from .aliases import TRAIN_ALIASES
    from libreyolo.training.config import TrainConfig

    internal_to_cli = {v: k for k, v in TRAIN_ALIASES.items()}
    excluded = {"size", "num_classes", "data", "data_dir"}

    kwargs = {}
    for f in fields(TrainConfig):
        if f.name in excluded:
            continue
        cli_name = internal_to_cli.get(f.name, f.name)
        if cli_name in params:
            kwargs[f.name] = params[cli_name]
    return kwargs


def _build_rfdetr_train_kwargs(
    params: dict[str, Any], *, model_path: str | None = None
) -> dict[str, Any]:
    """Build RF-DETR kwargs without forcing unrelated generic CLI defaults.

    RF-DETR uses a different training API than the YOLO-family wrappers. The CLI
    should translate only the parameters it intentionally supports, and leave the
    rest to RF-DETR's own defaults instead of pushing generic TrainConfig values.
    """
    from libreyolo.utils.general import increment_path

    output_dir = increment_path(
        Path(params["project"]) / params["name"],
        exist_ok=params["exist_ok"],
        mkdir=True,
    )

    kwargs: dict[str, Any] = {"output_dir": str(output_dir)}

    direct_mappings = {
        "epochs": "epochs",
        "batch": "batch_size",
        "lr0": "lr",
        "workers": "num_workers",
        "weight_decay": "weight_decay",
        "eval_interval": "eval_interval",
        "warmup_epochs": "warmup_epochs",
        "ema": "use_ema",
        "ema_decay": "ema_decay",
        "save_period": "checkpoint_interval",
        "seed": "seed",
        "device": "device",
    }

    for cli_name, target_name in direct_mappings.items():
        if is_user_provided(cli_name):
            kwargs[target_name] = params[cli_name]

    if is_user_provided("patience"):
        kwargs["early_stopping"] = params["patience"] > 0
        kwargs["early_stopping_patience"] = params["patience"]

    if is_user_provided("resume"):
        resume = params["resume"]
        if resume is True:
            resume = model_path
        elif not resume:
            resume = None
        kwargs["resume"] = resume

    return kwargs


def build_family_train_kwargs(
    params: dict[str, Any],
    family: str | None,
    *,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Build train kwargs, translating family-specific CLI/API mismatches."""
    if family == "rfdetr":
        return _build_rfdetr_train_kwargs(params, model_path=model_path)
    return build_train_kwargs(params)


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
