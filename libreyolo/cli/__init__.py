"""LibreYOLO command-line interface.

Entry point registered in pyproject.toml as ``libreyolo``.
"""

import sys

import typer

app = typer.Typer(
    name="libreyolo",
    help="LibreYOLO: open source YOLO detection toolkit.",
    add_completion=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    """Print the version and exit for the root ``--version`` flag."""
    if value:
        from libreyolo import __version__

        typer.echo(__version__)
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show LibreYOLO version and exit.",
    ),
) -> None:
    """LibreYOLO: open source YOLO detection toolkit."""


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
    # Normalize key=value bool syntax ONLY for the early logging peek. The args
    # handed to ``app()`` stay in their raw key=value form so each command's
    # ``KeyValueCommand`` does the per-command rewrite (it knows whether a flag
    # has a real ``--no-<flag>`` form). Emitting ``--no-verbose`` here would break
    # commands whose ``--verbose`` is one-way (e.g. predict), see issue #490 #41.
    logging_argv = _normalize_logging_flags(argv)
    _setup_logging_from_argv(logging_argv)

    from .commands import special, predict, train, val, export, quantize, ui, doctor, label, profile, monitor  # noqa: F401
    from .parsing import KeyValueCommand

    # Special commands
    for cmd_name in ("version", "checks", "models", "formats", "cfg", "info", "metadata"):
        app.command(cmd_name, cls=KeyValueCommand)(getattr(special, f"{cmd_name}_cmd"))

    # Face-embedding verification (facial-recognition): compare two images.
    app.command("compare", cls=KeyValueCommand)(special.compare_cmd)
    app.command("verify", cls=KeyValueCommand)(special.compare_cmd)
    # Face identification: build a gallery from a folder-per-person tree.
    app.command("enroll", cls=KeyValueCommand)(special.enroll_cmd)

    from .commands.fcos3d import fcos3d_cmd

    app.command("fcos3d", cls=KeyValueCommand)(fcos3d_cmd)
    from .commands.wilddet3d import wilddet3d_cmd

    app.command("wilddet3d", cls=KeyValueCommand)(wilddet3d_cmd)

    from .commands.mood3d import mood3d_cmd

    app.command("3dmood", cls=KeyValueCommand)(mood3d_cmd)

    # Core mode commands
    app.command("predict", cls=KeyValueCommand)(predict.predict_cmd)
    app.command("train", cls=KeyValueCommand)(train.train_cmd)
    app.command("val", cls=KeyValueCommand)(val.val_cmd)
    app.command("export", cls=KeyValueCommand)(export.export_cmd)
    app.command("quantize", cls=KeyValueCommand)(quantize.quantize_cmd)
    app.command("ui", cls=KeyValueCommand)(ui.ui_cmd)
    app.command("monitor", cls=KeyValueCommand)(monitor.monitor_cmd)
    app.command("label", cls=KeyValueCommand)(label.label_cmd)
    app.command("doctor", cls=KeyValueCommand)(doctor.doctor_cmd)

    # Profiler analysis command group (agent-friendly: every subcommand --json).
    app.add_typer(profile.profile_app, name="profile")

    app(args=argv[1:])
