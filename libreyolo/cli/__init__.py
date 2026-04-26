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


def _configure_warning_filters() -> None:
    """Suppress only known high-noise dependency deprecations."""
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.jit\.script` is deprecated\..*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"rfdetr\.util\.box_ops is deprecated;.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"rfdetr\.util\.logger is deprecated;.*",
        category=DeprecationWarning,
    )


def _strip_task_prefix(argv: list[str]) -> list[str]:
    """Strip optional 'detect' task prefix from argv.

    ``libreyolo detect predict ...`` becomes ``libreyolo predict ...``.
    """
    known_tasks = {"detect"}
    args = argv[1:]
    if args and args[0] in known_tasks:
        return [argv[0]] + args[1:]
    return argv


def _setup_logging_from_argv(argv: list[str]) -> None:
    """Configure logging early, before Typer parses args.

    Peeks at argv for --quiet/--verbose so the logger is ready
    before any command code runs.
    """
    from ..utils.logging import setup_logging

    args = argv[1:]
    quiet = "--quiet" in args
    verbose = "--verbose" in args
    setup_logging(quiet=quiet, verbose=verbose)


def _normalize_logging_flags(argv: list[str]) -> list[str]:
    """Normalize key=value bool syntax for flags that affect early logging."""
    from .parsing import rewrite_known_bool_flags

    args = rewrite_known_bool_flags(argv[1:], {"quiet", "verbose"})
    return [argv[0]] + args


def entrypoint() -> None:
    """CLI entry point registered in pyproject.toml."""
    _configure_warning_filters()
    argv = list(sys.argv)
    argv = _strip_task_prefix(argv)
    argv = _normalize_logging_flags(argv)
    _setup_logging_from_argv(argv)

    from .commands import special, predict, train, val, export  # noqa: F401
    from .parsing import KeyValueCommand

    # Special commands
    for cmd_name in ("version", "checks", "models", "formats", "cfg", "info"):
        app.command(cmd_name, cls=KeyValueCommand)(getattr(special, f"{cmd_name}_cmd"))

    # Core mode commands
    app.command("predict", cls=KeyValueCommand)(predict.predict_cmd)
    app.command("train", cls=KeyValueCommand)(train.train_cmd)
    app.command("val", cls=KeyValueCommand)(val.val_cmd)
    app.command("export", cls=KeyValueCommand)(export.export_cmd)

    app(args=argv[1:])
