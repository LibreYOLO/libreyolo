"""Family-specific config discovery and model name resolution for the CLI.

Config defaults come from the dataclass source of truth (TrainConfig subclasses).
The CLI discovers them via BaseModel._registry → TRAIN_CONFIG, so adding a new
model family requires zero CLI changes.
"""

from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Optional


def get_unsupported_train_params(family: str | None) -> set[str]:
    """Return the set of CLI parameters ignored by a model family's trainer."""
    if family == "rfdetr":
        from libreyolo.models import try_ensure_rfdetr

        rfcls = try_ensure_rfdetr()
        if rfcls is not None:
            return getattr(rfcls, "UNSUPPORTED_TRAIN_PARAMS", set())
    return set()


# =========================================================================
# Model name resolution
# =========================================================================

# Maps CLI model names (e.g. "yolox-s") to weight filenames (e.g. "LibreYOLOXs.pt").
_CLI_NAME_TO_WEIGHTS: dict[str, str] = {}


def _weight_filename_for_cli(cls, size_code: str) -> str:
    formatter = getattr(cls, "format_weight_filename", None)
    if callable(formatter):
        return formatter(size_code)
    return f"{cls.FILENAME_PREFIX}{size_code}{cls.WEIGHT_EXT}"


def _build_name_map() -> None:
    """Populate CLI name → weight filename mapping from model registry."""
    if _CLI_NAME_TO_WEIGHTS:
        return
    from libreyolo.models.base.model import BaseModel

    for cls in BaseModel._registry:
        for size_code in cls.INPUT_SIZES:
            cli_name = f"{cls.FAMILY}-{size_code}"
            _CLI_NAME_TO_WEIGHTS[cli_name] = _weight_filename_for_cli(cls, size_code)

    # Also try RF-DETR (lazily registered)
    from libreyolo.models import try_ensure_rfdetr

    rfcls = try_ensure_rfdetr()
    if rfcls is not None:
        for size_code in rfcls.INPUT_SIZES:
            cli_name = f"{rfcls.FAMILY}-{size_code}"
            _CLI_NAME_TO_WEIGHTS[cli_name] = _weight_filename_for_cli(
                rfcls, size_code
            )


def get_all_cli_names() -> list[str]:
    """Return all valid CLI model names (e.g. ['yolox-n', 'yolox-s', ...])."""
    _build_name_map()
    return list(_CLI_NAME_TO_WEIGHTS.keys())


def is_known_weight_filename(model: str) -> bool:
    """Return whether a path or filename matches a known packaged weight name."""
    _build_name_map()
    filename = Path(model).name.lower()
    return any(
        Path(weight).name.lower() == filename
        for weight in _CLI_NAME_TO_WEIGHTS.values()
    )


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


def _iter_model_classes():
    """Yield registered model classes, including lazy optional families."""
    from libreyolo.models import try_ensure_rfdetr
    from libreyolo.models.base.model import BaseModel

    seen: set[type] = set()
    for cls in BaseModel._registry:
        if cls not in seen:
            seen.add(cls)
            yield cls

    rfcls = try_ensure_rfdetr()
    if rfcls is not None and rfcls not in seen:
        yield rfcls


def detect_family_from_weight_filename(model: str) -> Optional[str]:
    """Detect model family from a known LibreYOLO weight filename or path."""
    filename = Path(model).name
    for cls in _iter_model_classes():
        if cls.detect_size_from_filename(filename) is not None:
            return cls.FAMILY
    return None


def _unwrap_checkpoint_state(checkpoint: Any) -> Any:
    """Extract a state dict from common trainer checkpoint layouts."""
    if not isinstance(checkpoint, dict):
        return checkpoint
    if "ema" in checkpoint and isinstance(checkpoint.get("ema"), dict):
        ema_data = checkpoint["ema"]
        return ema_data.get("module", ema_data)
    if "model" in checkpoint and isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"]
    return checkpoint


def detect_family_from_checkpoint(model_path: str) -> Optional[str]:
    """Detect model family from an existing PyTorch checkpoint when possible."""
    path = Path(model_path)
    if not path.exists() or path.suffix.lower() not in {".pt", ".pth"}:
        return None

    try:
        from libreyolo.utils.serialization import load_untrusted_torch_file

        checkpoint = load_untrusted_torch_file(
            path,
            map_location="cpu",
            context="CLI model family inspection",
        )
    except Exception:
        return None

    if isinstance(checkpoint, dict):
        family = checkpoint.get("model_family")
        if isinstance(family, str) and family:
            return family

    state = _unwrap_checkpoint_state(checkpoint)
    if not isinstance(state, dict):
        return None

    for cls in _iter_model_classes():
        try:
            if cls.can_load(state):
                return cls.FAMILY
        except Exception:
            continue
    return None


def detect_family_from_model_ref(
    model: str,
    resolved_model_path: str | None = None,
    *,
    inspect_checkpoint: bool = False,
) -> Optional[str]:
    """Detect model family from a CLI model reference and optional resolved path.

    The CLI accepts aliases (``yolox-s``), packaged weight filenames
    (``LibreYOLOXs.pt``), and filesystem paths. Keep this detection registry-based
    so family-specific defaults follow model-class metadata instead of hardcoded
    CLI switches.
    """
    refs = [model]
    if resolved_model_path is not None and resolved_model_path != model:
        refs.append(resolved_model_path)

    for ref in refs:
        family = detect_family_from_name(ref)
        if family is not None:
            return family
        family = detect_family_from_weight_filename(ref)
        if family is not None:
            return family

    if inspect_checkpoint:
        for ref in refs:
            family = detect_family_from_checkpoint(ref)
            if family is not None:
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


def apply_family_defaults(
    params: dict[str, Any],
    family: str,
    mode: str,
    *,
    user_provided: set[str] | None = None,
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

    # Reverse alias map: internal name → CLI name (for user_provided check)
    from .aliases import TRAIN_ALIASES

    internal_to_cli = {v: k for k, v in TRAIN_ALIASES.items()}

    provided = user_provided or set()
    result = dict(params)
    for internal_name, default_value in family_diffs.items():
        # The params dict uses CLI-facing names, so check both
        cli_name = internal_to_cli.get(internal_name, internal_name)
        if cli_name in result and cli_name not in provided:
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
    params: dict[str, Any],
    *,
    model_path: str | None = None,
    user_provided: set[str] | None = None,
) -> dict[str, Any]:
    """Build RF-DETR kwargs from resolved family-aware CLI params.

    RF-DETR uses a different training API than the YOLO-family wrappers. The CLI
    translates only the parameters it intentionally supports; unsupported YOLO
    family options stay out of the upstream adapter call.
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
        if cli_name in params:
            kwargs[target_name] = params[cli_name]

    provided = user_provided or set()
    if "patience" in params:
        kwargs["early_stopping"] = params["patience"] > 0
        kwargs["early_stopping_patience"] = params["patience"]

    if "resume" in provided:
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
    user_provided: set[str] | None = None,
) -> dict[str, Any]:
    """Build train kwargs, translating family-specific CLI/API mismatches."""
    if family == "rfdetr":
        return _build_rfdetr_train_kwargs(
            params, model_path=model_path, user_provided=user_provided
        )
    if family == "deimv2":
        kwargs = build_train_kwargs(params)
        provided = user_provided or set()
        from .aliases import TRAIN_ALIASES

        internal_to_cli = {v: k for k, v in TRAIN_ALIASES.items()}
        size_defaulted = {
            "epochs",
            "batch",
            "imgsz",
            "optimizer",
            "lr0",
            "weight_decay",
            "scheduler",
            "warmup_epochs",
            "warmup_lr_start",
            "no_aug_epochs",
            "min_lr_ratio",
            "mosaic_prob",
            "mixup_prob",
            "hsv_prob",
            "flip_prob",
            "degrees",
            "translate",
            "shear",
            "mosaic_scale",
            "mixup_scale",
            "ema",
            "ema_decay",
            "amp",
            "name",
        }
        for internal_name in size_defaulted:
            cli_name = internal_to_cli.get(internal_name, internal_name)
            if cli_name not in provided:
                kwargs.pop(internal_name, None)
        return kwargs
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
                if k not in train_exclude
            }

    rfcls = try_ensure_rfdetr()
    if rfcls is not None and rfcls.FAMILY not in seen:
        diffs = get_family_defaults(rfcls.FAMILY)
        if diffs:
            family_overrides[rfcls.FAMILY] = {
                train_internal_to_cli.get(k, k): _to_json_safe(v)
                for k, v in diffs.items()
                if k not in train_exclude
            }

    return {
        "train_defaults": train_defaults,
        "val_defaults": val_defaults,
        "predict_defaults": predict_defaults,
        "family_overrides": family_overrides,
    }
