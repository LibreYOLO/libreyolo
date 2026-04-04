"""LibreYOLO CLI — ultralytics-compatible command-line interface.

Entry point registered in pyproject.toml as ``libreyolo``.
"""

import sys

import typer

app = typer.Typer(
    name="libreyolo",
    help="LibreYOLO — open source YOLO detection toolkit.",
    add_completion=False,
    no_args_is_help=True,
)


def _strip_task_prefix() -> None:
    """Strip optional 'detect' task prefix from argv.

    ``libreyolo detect predict ...`` becomes ``libreyolo predict ...``.
    """
    known_tasks = {"detect"}
    args = sys.argv[1:]
    if args and args[0] in known_tasks:
        sys.argv = [sys.argv[0]] + args[1:]


def _setup_logging_from_argv() -> None:
    """Configure logging early, before Typer parses args.

    Peeks at sys.argv for --quiet/--verbose so the logger is ready
    before any command code runs.
    """
    from ..utils.logging import setup_logging

    args = sys.argv[1:]
    quiet = "--quiet" in args
    verbose = "--verbose" in args
    setup_logging(quiet=quiet, verbose=verbose)


def entrypoint() -> None:
    """CLI entry point registered in pyproject.toml."""
    import warnings

    warnings.filterwarnings("ignore")

    _strip_task_prefix()
    _setup_logging_from_argv()

    from .commands import special, predict, train, val, export  # noqa: F401
    from .parsing import KeyValueCommand

    # Special commands
    for cmd_name in ("version", "checks", "models", "formats", "cfg", "info"):
        app.command(cmd_name, cls=KeyValueCommand)(
            getattr(special, f"{cmd_name}_cmd")
        )

    # Core mode commands
    app.command("predict", cls=KeyValueCommand)(predict.predict_cmd)
    app.command("train", cls=KeyValueCommand)(train.train_cmd)
    app.command("val", cls=KeyValueCommand)(val.val_cmd)
    app.command("export", cls=KeyValueCommand)(export.export_cmd)

    app()
