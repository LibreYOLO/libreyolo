"""Output routing for the LibreYOLO CLI.

stdout is the API (results only). stderr is for humans (progress, logs).
"""

import json
import logging
import sys
from typing import Any

from pathlib import Path

from .errors import CLIError

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """Strict JSON default: only allow Path → str. Everything else is an error."""
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class OutputHandler:
    """Routes output to stdout (results) and stderr (progress/errors)."""

    def __init__(self, *, json_mode: bool = False, quiet: bool = False) -> None:
        self.json_mode = json_mode
        self.quiet = quiet
        self.is_tty = sys.stdout.isatty()

    def result(self, data: dict[str, Any]) -> None:
        """Write result to stdout. In JSON mode, adds schema_version."""
        if self.json_mode:
            public_data = {
                key: value for key, value in data.items() if not key.startswith("_")
            }
            public_data["schema_version"] = 1
            print(json.dumps(public_data, default=_json_default))
        else:
            self._print_human(data)

    def progress(self, message: str) -> None:
        """Write progress info to stderr via logger. Respects --quiet."""
        logger.info(message)

    def warning(self, message: str) -> None:
        """Write warnings to stderr."""
        logger.warning(message)

    def error(self, err: CLIError) -> None:
        """Write error. With --json: JSON to stdout. Without: log to stderr."""
        if self.json_mode:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "error": err.code,
                        "message": err.message,
                        "suggestion": err.suggestion,
                    },
                    default=_json_default,
                )
            )
        else:
            logger.error("Error [%s]: %s", err.code, err.message)
            if err.suggestion:
                logger.info("  Suggestion: %s", err.suggestion)

    def _print_human(self, data: dict[str, Any]) -> None:
        """Format data as human-readable text to stdout."""
        if "_human_text" in data:
            print(data["_human_text"])
        else:
            for key, value in data.items():
                if not key.startswith("_"):
                    print(f"  {key}: {value}")
