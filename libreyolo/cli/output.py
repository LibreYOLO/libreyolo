"""Output routing for the LibreYOLO CLI.

stdout is the API (results only). stderr is for humans (progress, logs).
"""

import contextlib
import json
import logging
import os
import sys
from typing import Any, Iterator, TextIO

from pathlib import Path

from .errors import CLIError

logger = logging.getLogger(__name__)

# Set while a --json command runs: the JSON document is written here, and
# everything else sent to stdout is diverted to stderr.
_json_stream: TextIO | None = None


def _flush_c_stdio() -> None:
    """Flush C-level stdio buffers (e.g. torch's C++ ONNX log)."""
    try:
        import ctypes

        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass


@contextlib.contextmanager
def stdout_reserved_for_json() -> Iterator[None]:
    """Keep stdout for the JSON document while a --json command runs.

    Third-party output (pycocotools' AP table, torch's ONNX graph log) would
    otherwise land on stdout and break ``json.loads``. It goes to stderr
    instead, at the Python level and, when stdout is a real file descriptor,
    at the descriptor level too.
    """
    global _json_stream
    if _json_stream is not None:
        yield
        return
    original = sys.stdout
    original.flush()
    try:
        fd = original.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        fd = None
    saved_fd = None
    if fd is None:
        _json_stream = original
    else:
        _flush_c_stdio()
        saved_fd = os.dup(fd)
        os.dup2(stderr_fd, fd)
        _json_stream = open(
            saved_fd,
            "w",
            encoding=getattr(original, "encoding", None) or "utf-8",
            closefd=False,
        )
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = original
        stream, _json_stream = _json_stream, None
        if saved_fd is not None:
            original.flush()
            _flush_c_stdio()
            stream.close()
            os.dup2(saved_fd, fd)
            os.close(saved_fd)


def _json_default(obj: Any) -> Any:
    """Strict JSON default: only allow Path -> str. Everything else is an error."""
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _print_json(data: dict[str, Any]) -> None:
    stream = _json_stream if _json_stream is not None else sys.stdout
    print(json.dumps(data, default=_json_default), file=stream, flush=True)


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
            _print_json(public_data)
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
            _print_json(
                {
                    "schema_version": 1,
                    "error": err.code,
                    "message": err.message,
                    "suggestion": err.suggestion,
                }
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
