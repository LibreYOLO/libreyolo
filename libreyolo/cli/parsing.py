"""Custom Typer command class that supports key=value argument syntax.

Subclasses TyperCommand and rewrites key=value tokens into --key value
before Click's parser sees them. This preserves all Typer features
(auto-help, type validation, shell completion) while supporting
ultralytics-compatible syntax.
"""

import ast
import re
from typing import Any, Optional

import click
from typer.core import TyperCommand


_TRUE_VALUES = {"true", "1"}
_FALSE_VALUES = {"false", "0"}


def rewrite_known_bool_flags(args: list[str], bool_flags: set[str]) -> list[str]:
    """Rewrite known bool flags from key=value or bare-word syntax.

    This is used both by the CLI parser and by the entry point's early logging
    setup so flags like ``quiet=true`` behave the same as ``--quiet``.
    """
    new_args: list[str] = []
    for arg in args:
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_-]*)=(.*)$", arg)
        if m:
            key, value = m.group(1), m.group(2)
            cli_key = key.replace("_", "-")
            lower_value = value.lower()
            if cli_key in bool_flags and lower_value in _TRUE_VALUES | _FALSE_VALUES:
                if lower_value in _TRUE_VALUES:
                    new_args.append(f"--{cli_key}")
                else:
                    new_args.append(f"--no-{cli_key}")
            else:
                new_args.append(arg)
        elif arg.replace("_", "-") in bool_flags:
            new_args.append(f"--{arg.replace('_', '-')}")
        else:
            new_args.append(arg)
    return new_args


class KeyValueCommand(TyperCommand):
    """Typer command that accepts both key=value and --key value syntax."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # Identify boolean flag parameter names from the command definition
        bool_flags: set[str] = set()
        for param in self.params:
            if isinstance(param, click.Option) and param.is_flag:
                for opt in param.opts:
                    bool_flags.add(opt.lstrip("-"))
                for opt in param.secondary_opts:
                    bool_flags.add(opt.lstrip("-"))

        new_args = rewrite_known_bool_flags(args, bool_flags)
        parsed_args: list[str] = []
        for arg in new_args:
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

        return super().parse_args(ctx, parsed_args)


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
