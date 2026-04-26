"""Shared helpers for CLI command implementations."""

import json
from pathlib import Path
from typing import Any, NoReturn, Optional

import typer

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


def load_model_or_exit(
    out: OutputHandler,
    *,
    model: str,
    model_path: str,
    device: str,
) -> Any:
    """Load a model with consistent CLI error handling."""
    from libreyolo import LibreYOLO

    out.progress(f"Loading {model}...")
    try:
        return LibreYOLO(model_path, device=device)
    except Exception as exc:
        exit_with_error(
            out,
            "model_load_failed",
            f"Failed to load model '{model}': {exc}",
        )


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


def get_loaded_model_input_size(
    loaded_model: Any,
    *,
    imgsz: Optional[int] = None,
    default: int = 640,
) -> int:
    """Return the effective square input size for wrapper or backend output."""
    if imgsz is not None:
        return int(imgsz)

    get_input_size = getattr(loaded_model, "_get_input_size", None)
    if callable(get_input_size):
        try:
            return int(get_input_size())
        except Exception:
            pass

    input_size = getattr(loaded_model, "input_size", None)
    if input_size is not None:
        return int(input_size)

    backend_imgsz = getattr(loaded_model, "imgsz", None)
    if backend_imgsz is not None:
        return int(backend_imgsz)

    input_sizes = getattr(loaded_model, "INPUT_SIZES", None)
    size = getattr(loaded_model, "size", None)
    if isinstance(input_sizes, dict) and size is not None:
        return int(input_sizes.get(size, default))

    return default


def resolve_model_or_exit(out: OutputHandler, model: str) -> str:
    """Resolve a model reference or fail with a consistent CLI error."""
    from .config import get_all_cli_names, is_known_weight_filename, resolve_model_name
    from .errors import suggest_key

    model_path = resolve_model_name(model)
    if model_path != model or Path(model).exists() or is_known_weight_filename(model):
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


def exit_stage_error(
    out: OutputHandler,
    *,
    stage: str,
    detail: Exception | str,
    code: str = "io_error",
    suggestion: Optional[str] = None,
) -> NoReturn:
    """Emit a stage-specific runtime error and terminate the command."""
    exit_with_error(
        out,
        code,
        f"{stage} failed: {detail}",
        suggestion=suggestion,
    )


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
