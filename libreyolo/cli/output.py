"""Output routing for the LibreYOLO CLI.

stdout is the API (results only). stderr is for humans (progress, logs).
"""

import json
import sys
from typing import Any

from .errors import CLIError


class OutputHandler:
    """Routes output to stdout (results) and stderr (progress/errors)."""

    def __init__(
        self, *, json_mode: bool = False, quiet: bool = False
    ) -> None:
        self.json_mode = json_mode
        self.quiet = quiet
        self.is_tty = sys.stdout.isatty()

    def result(self, data: dict[str, Any]) -> None:
        """Write result to stdout. In JSON mode, adds schema_version."""
        if self.json_mode:
            data["schema_version"] = 1
            print(json.dumps(data, default=str))
        else:
            self._print_human(data)

    def progress(self, message: str) -> None:
        """Write progress info to stderr. Suppressed by --quiet."""
        if not self.quiet:
            print(message, file=sys.stderr)

    def error(self, err: CLIError) -> None:
        """Write error. With --json: JSON to stdout. Without: text to stderr."""
        if self.json_mode:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "error": err.code,
                        "message": err.message,
                        "suggestion": err.suggestion,
                    },
                    default=str,
                )
            )
        else:
            print(f"Error [{err.code}]: {err.message}", file=sys.stderr)
            if err.suggestion:
                print(f"  Suggestion: {err.suggestion}", file=sys.stderr)

    def _print_human(self, data: dict[str, Any]) -> None:
        """Format data as human-readable text to stdout."""
        # Commands provide their own human formatting via _human_text key.
        # Fall back to simple key: value display.
        if "_human_text" in data:
            print(data["_human_text"])
        else:
            for key, value in data.items():
                if not key.startswith("_"):
                    print(f"  {key}: {value}")
