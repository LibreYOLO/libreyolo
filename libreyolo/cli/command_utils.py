"""Shared helpers for CLI command implementations."""

import json
from pathlib import Path
from typing import Any, NoReturn, Optional, Set, Tuple, Union

import click
import typer

from libreyolo.utils.image_size import normalize_imgsz

from .errors import CLIError
from .output import OutputHandler


def exit_with_error(
    out: OutputHandler,
    code: str,
    message: str,
    *,
    suggestion: Optional[str] = None,
) -> NoReturn:
    """Emit a structured CLI error and terminate the command."""
    err = CLIError(code, message, suggestion=suggestion)
    out.error(err)
    raise typer.Exit(code=err.exit_code)


# Inclusive (low, high) bounds; None leaves that side open.
_VALUE_RANGES: dict[str, tuple[Optional[float], Optional[float]]] = {
    "conf": (0.0, 1.0),
    "iou": (0.0, 1.0),
    "max_det": (1, None),
    "batch": (1, None),
    "epochs": (1, None),
}


def exit_if_out_of_range(
    out: OutputHandler, *, autobatch: bool = False, **values: Any
) -> None:
    """Reject option values outside their valid range with config_range_error.

    ``autobatch`` also accepts ``batch=-1`` (train's AutoBatch request).
    """
    for name, value in values.items():
        if value is None or (autobatch and name == "batch" and value == -1):
            continue
        low, high = _VALUE_RANGES[name]
        if (low is not None and not value >= low) or (
            high is not None and not value <= high
        ):
            bound = f">= {low}" if high is None else f"in [{low}, {high}]"
            if autobatch and name == "batch":
                bound += ", or -1 for AutoBatch"
            exit_with_error(
                out, "config_range_error", f"{name} must be {bound}, got {value}"
            )


def load_model_or_exit(
    out: OutputHandler,
    *,
    model: str,
    model_path: str,
    device: str,
    task: str | None = None,
) -> Any:
    """Load a model with consistent CLI error handling."""
    from libreyolo import LibreYOLO

    out.progress(f"Loading {model}...")
    try:
        return LibreYOLO(model_path, device=device, task=task)
    except Exception as exc:
        missing = _missing_accelerator(device)
        if missing is not None:
            exit_with_error(
                out,
                "device_not_available",
                f"device={device} was requested but {missing} is not available "
                "on this machine.",
                suggestion="Use device=cpu, or device=auto to pick the best "
                "available device.",
            )
        exit_with_error(
            out,
            "model_load_failed",
            f"Failed to load model '{model}': {exc}",
        )


def _missing_accelerator(device: Any) -> Optional[str]:
    """Name the accelerator ``device`` asks for when this machine lacks it."""
    import torch

    name = str(device).strip().lower()
    if name.startswith("cuda") or name.replace(",", "").isdigit():
        return None if torch.cuda.is_available() else "CUDA"
    if name.startswith("mps"):
        mps = getattr(torch.backends, "mps", None)
        return None if mps is not None and mps.is_available() else "MPS"
    return None


def get_loaded_model_family(loaded_model: Any) -> Optional[str]:
    """Return the model family for native wrappers or exported-runtime backends."""
    family = getattr(loaded_model, "FAMILY", None)
    if family:
        return str(family)

    family = getattr(loaded_model, "model_family", None)
    if family:
        return str(family)

    get_model_name = getattr(loaded_model, "_get_model_name", None)
    if callable(get_model_name):
        try:
            family = get_model_name()
        except Exception:
            family = None
        if family:
            return str(family)

    return None


ImageSize = Union[int, Tuple[int, int]]


def _coerce_input_size(value: Any) -> ImageSize:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"input size must be int or (height, width), got {value}")
        h, w = int(value[0]), int(value[1])
        return h if h == w else (h, w)
    return int(value)


def parse_imgsz_str(imgsz: str | int | None) -> int | tuple[int, int] | None:
    """Parse an imgsz value from CLI string format.

    Accepts:
        "640"      -> 640
        "480x640"  -> (480, 640)
        "480X640"  -> (480, 640)
        "480,640"  -> (480, 640), legacy alias
        None       -> None
        int        -> int (pass-through)
    """
    if imgsz is None:
        return None
    if isinstance(imgsz, str) and not imgsz.strip():
        return None
    return normalize_imgsz(
        imgsz,
        name="imgsz",
        allow_string=True,
        allow_comma=True,
    )


def get_loaded_model_input_size(
    loaded_model: Any,
    *,
    imgsz: Optional[ImageSize] = None,
    default: int = 640,
) -> ImageSize:
    """Return the effective input size for wrapper or backend output."""
    if imgsz is not None:
        return _coerce_input_size(imgsz)

    get_input_size = getattr(loaded_model, "_get_input_size", None)
    if callable(get_input_size):
        try:
            return _coerce_input_size(get_input_size())
        except Exception:
            pass

    input_size = getattr(loaded_model, "input_size", None)
    if input_size is not None:
        return _coerce_input_size(input_size)

    backend_imgsz = getattr(loaded_model, "imgsz", None)
    if backend_imgsz is not None:
        return _coerce_input_size(backend_imgsz)

    input_sizes = getattr(loaded_model, "INPUT_SIZES", None)
    size = getattr(loaded_model, "size", None)
    if isinstance(input_sizes, dict) and size is not None:
        return int(input_sizes.get(size, default))

    return default


def resolve_model_or_exit(out: OutputHandler, model: str) -> str:
    """Resolve a model reference or fail with a consistent CLI error."""
    from libreyolo.backends.triton import is_triton_model_url
    from .config import get_all_cli_names, is_known_weight_filename, resolve_model_name
    from .errors import suggest_key

    model_path = resolve_model_name(model)
    if (
        is_triton_model_url(model)
        or model_path != model
        or Path(model).exists()
        or is_known_weight_filename(model)
    ):
        return model_path

    all_names = get_all_cli_names()
    suggestion = suggest_key(model, all_names)
    hint = f" Did you mean '{suggestion}'?" if suggestion else ""
    exit_with_error(
        out,
        "model_not_found",
        f"Unknown model '{model}'.{hint}",
        suggestion=f"Available: {', '.join(all_names)}",
    )


def is_device_op_error(exc: BaseException | str) -> bool:
    """True for torch's "operator ... not implemented for the <X> device"."""
    return isinstance(exc, NotImplementedError) and (
        "not currently implemented for the" in str(exc)
    )


def exit_stage_error(
    out: OutputHandler,
    *,
    stage: str,
    detail: Exception | str,
    code: str = "io_error",
    suggestion: Optional[str] = None,
) -> NoReturn:
    """Emit a stage-specific runtime error and terminate the command."""
    if is_device_op_error(detail):
        code = "device_not_available"
        suggestion = suggestion or "Run on another device, for example device=cpu."
    exit_with_error(
        out,
        code,
        f"{stage} failed: {detail}",
        suggestion=suggestion,
    )


def model_call_error_code(exc: BaseException) -> str:
    """CLI error code for an exception raised inside a model call.

    Argument and configuration errors are usage errors (exit 2), not I/O
    failures: ``ValueError`` is a bad value, ``TypeError`` a bad or unknown
    argument, ``NotImplementedError`` a configuration the model does not
    support.
    """
    message = str(exc)
    if is_device_op_error(exc):
        return "device_not_available"
    if "CUDA out of memory" in message or type(exc).__name__ == "OutOfMemoryError":
        return "cuda_oom"
    if isinstance(exc, NotImplementedError):
        return "config_unsupported"
    if isinstance(exc, TypeError):
        if "unexpected keyword argument" in message or message.startswith(
            "Unsupported"
        ):
            return "config_unknown_key"
        return "config_type_error"
    if isinstance(exc, ValueError):
        return "config_type_error"
    return "io_error"


# What torch raises when an input size does not tile a model's feature maps.
_IMGSZ_SHAPE_ERRORS = ("Sizes of tensors must match", "must match the size of tensor")


def is_imgsz_error(exc: BaseException) -> bool:
    """Whether a model call failed because it cannot run at the requested imgsz."""
    message = str(exc)
    if isinstance(exc, ValueError):
        return "imgsz" in message
    return isinstance(exc, RuntimeError) and any(
        marker in message for marker in _IMGSZ_SHAPE_ERRORS
    )


def exit_imgsz_error(out: OutputHandler, exc: BaseException) -> NoReturn:
    """Report an imgsz the model cannot take as a usage error."""
    if "exported with a fixed" in str(exc):
        suggestion = "Use the exported imgsz, or re-export at the size you need."
    else:
        suggestion = (
            "Use a multiple of the model stride (e.g. 320 or 640), "
            "or omit imgsz for the model's native size."
        )
    exit_with_error(
        out,
        "invalid_imgsz",
        f"The model cannot run at this imgsz: {exc}",
        suggestion=suggestion,
    )


def get_user_provided_params() -> Set[str]:
    """Return parameter names explicitly provided on the current command line."""
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        # typer >= 0.26 vendors its own click as typer._click with a separate
        # context stack — fall back to it so the test runner sees the same ctx
        # that KeyValueCommand.parse_args wrote user_provided into.
        try:
            from typer._click.globals import get_current_context as _typer_get_ctx

            ctx = _typer_get_ctx(silent=True)
        except ImportError:
            pass
    if ctx is None:
        return set()
    # KeyValueCommand.parse_args stores the set in ctx.meta for reliable access
    # regardless of click version or test-runner context behaviour.
    if "user_provided" in ctx.meta:
        return ctx.meta["user_provided"]
    # Fallback for plain click commands not using KeyValueCommand.
    return {
        p.name
        for p in ctx.command.params
        if ctx.get_parameter_source(p.name) == click.core.ParameterSource.COMMANDLINE
    }


def help_json_callback(
    ctx: typer.Context,
    param: typer.CallbackParam,
    value: bool,
) -> None:
    """Eager callback for --help-json: dump command schema and exit."""
    del param
    if not value:
        return

    params = []
    flags = []
    for p in ctx.command.params:
        if p.name in ("help_json", "help"):
            continue
        info: dict[str, Any] = {"name": p.name, "type": p.type.name}
        if p.default is not None:
            info["default"] = p.default
        if p.required:
            info["required"] = True
        if p.help:
            info["help"] = p.help
        params.append(info)
        if getattr(p, "is_flag", False):
            for opt in (*p.opts, *p.secondary_opts):
                if opt.startswith("--"):
                    flags.append(opt)

    schema = {
        "schema_version": 1,
        "command": ctx.info_name,
        "parameters": params,
        "flags": sorted(set(flags)),
    }
    print(json.dumps(schema, default=str))
    ctx.exit()
