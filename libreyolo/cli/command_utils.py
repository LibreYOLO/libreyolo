"""Shared helpers for CLI command implementations."""

import json

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
