"""Shared helpers for CLI command implementations."""

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
