"""Custom Typer command class that supports key=value argument syntax.

Subclasses TyperCommand and rewrites key=value tokens into --key value
before Click's parser sees them. This preserves all Typer features
(auto-help, type validation, shell completion) while supporting key=value
syntax.
"""

import ast
import re
from typing import Any, Optional

import click
from typer.core import TyperCommand


_TRUE_VALUES = {"true", "1"}
_FALSE_VALUES = {"false", "0"}


def rewrite_known_bool_flags(
    args: list[str],
    bool_flags: set[str],
    negatable: Optional[set[str]] = None,
    value_options: Optional[set[str]] = None,
) -> list[str]:
    """Rewrite known bool flags from key=value or bare-word syntax.

    This is used both by the CLI parser and by the entry point's early logging
    setup so flags like ``quiet=true`` behave the same as ``--quiet``.

    ``negatable`` is the subset of ``bool_flags`` (CLI names, dash form) that
    actually expose a ``--no-<flag>`` form. For one-way flags (e.g. ``--json``,
    ``--quiet``) that have no negative form, ``key=false`` drops the flag
    entirely instead of emitting a nonexistent ``--no-<flag>`` (which Click
    would reject with "No such option").

    ``value_options`` are ``--key`` options that take a value: the token after
    one is that value and is passed through untouched (``--name show``).
    """
    if negatable is None:
        negatable = set()
    value_options = value_options or set()
    new_args: list[str] = []
    takes_value = False
    for arg in args:
        if takes_value:
            new_args.append(arg)
            takes_value = False
            continue
        if arg in value_options:
            new_args.append(arg)
            takes_value = True
            continue
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_-]*)=(.*)$", arg)
        if m:
            key, value = m.group(1), m.group(2)
            cli_key = key.replace("_", "-")
            lower_value = value.lower()
            if cli_key in bool_flags and lower_value in _TRUE_VALUES | _FALSE_VALUES:
                if lower_value in _TRUE_VALUES:
                    new_args.append(f"--{cli_key}")
                elif cli_key in negatable:
                    new_args.append(f"--no-{cli_key}")
                # One-way flag set to false: omit it entirely.
            else:
                new_args.append(arg)
        elif arg.replace("_", "-") in bool_flags:
            new_args.append(f"--{arg.replace('_', '-')}")
        else:
            new_args.append(arg)
    return new_args


def _usage_error_types() -> tuple[type[Exception], ...]:
    """Click's UsageError, plus the copy newer Typer versions vendor."""
    types: list[type[Exception]] = [click.exceptions.UsageError]
    try:
        from typer._click.exceptions import UsageError as VendoredUsageError
    except ImportError:
        pass
    else:
        types.append(VendoredUsageError)
    return tuple(types)


def _usage_error_code(exc: Exception) -> str:
    names = {cls.__name__ for cls in type(exc).__mro__}
    if "NoSuchOption" in names or "unexpected extra argument" in str(exc):
        return "config_unknown_key"
    if "MissingParameter" in names:
        return "config_required_key"
    return "config_type_error"


class KeyValueCommand(TyperCommand):
    """Typer command that accepts both key=value and --key value syntax."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # Build bool_flags and a CLI-name → param-name reverse map.
        # Use getattr throughout to be robust across typer/click versions
        # (TyperOption may not be a direct subclass of click.Option in all envs).
        bool_flags: set[str] = set()
        negatable: set[str] = set()
        value_options: set[str] = set()
        cli_to_param: dict[str, str] = {}
        for param in self.params:
            if not getattr(param, "opts", None):
                continue
            param_name = getattr(param, "name", None)
            if not getattr(param, "is_flag", False):
                value_options.update(o for o in param.opts if o.startswith("--"))
            for opt in param.opts:
                if opt.startswith("--"):
                    cli_key = opt.lstrip("-").replace("-", "_")
                    if param_name:
                        cli_to_param[cli_key] = param_name
            for opt in getattr(param, "secondary_opts", []):
                if opt.startswith("--"):
                    cli_key = opt.lstrip("-").replace("-", "_")
                    if param_name:
                        cli_to_param[cli_key] = param_name
            if getattr(param, "is_flag", False):
                primary_names: list[str] = []
                for opt in param.opts:
                    if opt.startswith("--"):
                        name = opt.lstrip("-")
                        bool_flags.add(name)
                        primary_names.append(name)
                has_negative = False
                for opt in getattr(param, "secondary_opts", []):
                    if opt.startswith("--"):
                        name = opt.lstrip("-")
                        bool_flags.add(name)
                        if name.startswith("no-"):
                            has_negative = True
                if has_negative:
                    negatable.update(primary_names)

        # Compute which params the user explicitly provided from the raw args,
        # and store them in ctx.meta so get_user_provided_params() can retrieve
        # them without relying on click's ParameterSource tracking (which can
        # be unreliable inside typer's test runner on some Python versions).
        user_provided: set[str] = set()
        takes_value = False
        for arg in args:
            if takes_value:
                takes_value = False
                continue
            if arg.startswith("--"):
                takes_value = arg in value_options
                raw = arg.lstrip("-").split("=")[0].replace("-", "_")
                user_provided.add(cli_to_param.get(raw, raw))
            elif re.match(r"^[a-zA-Z_][a-zA-Z0-9_-]*=", arg):
                key = arg.split("=")[0].replace("-", "_")
                user_provided.add(cli_to_param.get(key, key))
            elif re.match(r"^[a-zA-Z_][a-zA-Z0-9_-]*$", arg):
                # Bare word: only counts if it's a known bool flag
                key = arg.replace("-", "_")
                if arg.replace("_", "-") in bool_flags or arg in bool_flags:
                    user_provided.add(cli_to_param.get(key, key))
        ctx.meta["user_provided"] = user_provided

        new_args = rewrite_known_bool_flags(
            args, bool_flags, negatable, value_options
        )
        parsed_args: list[str] = []
        takes_value = False
        for arg in new_args:
            # The value of a preceding ``--key`` is never rewritten.
            if takes_value or arg in value_options:
                parsed_args.append(arg)
                takes_value = not takes_value
                continue
            # Match key=value pattern (key must start with letter or underscore)
            m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_-]*)=(.*)$", arg)
            if m:
                key, value = m.group(1), m.group(2)
                cli_key = key.replace("_", "-")

                # Boolean flag with explicit value: half=true → --half
                parsed_args.append(f"--{cli_key}")
                parsed_args.append(value)
            else:
                parsed_args.append(arg)

        # Click's parser consumes the list, so check for --json up front.
        json_requested = "--json" in parsed_args
        try:
            return super().parse_args(ctx, parsed_args)
        except _usage_error_types() as exc:
            if not json_requested:
                raise
            # --json promises a machine-readable error, usage errors included.
            from .errors import CLIError
            from .output import OutputHandler

            possibilities = getattr(exc, "possibilities", None) or []
            if possibilities:
                suggestion = f"Did you mean '{possibilities[0].lstrip('-').replace('-', '_')}'?"
            else:
                suggestion = f"Run 'libreyolo {ctx.info_name} --help' for valid options."
            err = CLIError(_usage_error_code(exc), exc.format_message(), suggestion)
            OutputHandler(json_mode=True).error(err)
            ctx.exit(err.exit_code)

    def invoke(self, ctx: click.Context) -> Any:
        if ctx.params.get("json_output"):
            from .output import stdout_reserved_for_json

            with stdout_reserved_for_json():
                return super().invoke(ctx)
        return super().invoke(ctx)


class PythonLiteral(click.ParamType):
    """Click param type that parses Python literals (lists, tuples) via ast.literal_eval."""

    name = "literal"

    def __init__(self, expected_type: Optional[type] = None) -> None:
        self.expected_type = expected_type

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Any:
        if isinstance(value, (list, tuple)):
            return value
        try:
            result = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            self.fail(f"Could not parse '{value}' as a Python literal.", param, ctx)
        if self.expected_type and not isinstance(result, self.expected_type):
            self.fail(
                f"Expected {self.expected_type.__name__}, got {type(result).__name__}.",
                param,
                ctx,
            )
        return result
