"""Unified logging for LibreYOLO.

Usage in any module::

    import logging
    logger = logging.getLogger(__name__)
    logger.info("training started")

The CLI entry point calls ``setup_logging()`` once to configure
handlers, formatting, and log level.
"""

import logging
import sys
from typing import ClassVar

# Custom HEADER level — bold green banner for phase separators
HEADER_LEVEL = 25
logging.addLevelName(HEADER_LEVEL, "HEADER")


class ConsoleFormatter(logging.Formatter):
    """Colored log formatter for terminal output."""

    GREEN = "\x1b[32m"
    RESET = "\x1b[0m"

    LEVEL_COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: "\x1b[34m",  # Blue
        logging.INFO: RESET,  # Normal
        HEADER_LEVEL: GREEN + "\x1b[1m",  # Bold Green
        logging.WARNING: "\x1b[33m",  # Yellow
        logging.ERROR: "\x1b[31m",  # Red
        logging.CRITICAL: "\x1b[31;1m",  # Bold Red
    }

    def __init__(self, *, colors: bool = True) -> None:
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")
        self.colors = colors

    def format(self, record: logging.LogRecord) -> str:
        asctime = self.formatTime(record, self.datefmt)
        message = record.getMessage()

        if self.colors:
            color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
            s = (
                f"{self.GREEN}{asctime}{self.RESET} | "
                f"{color}{record.levelname:<8}{self.RESET} | "
                f"{color}{message}{self.RESET}"
            )
        else:
            s = f"{asctime} | {record.levelname:<8} | {message}"

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            s += "\n" + record.exc_text
        return s


def setup_logging(*, quiet: bool = False, verbose: bool = False) -> logging.Logger:
    """Configure the ``libreyolo`` logger hierarchy.

    Call this once at startup (CLI entry point). Every module that uses
    ``logging.getLogger(__name__)`` automatically inherits this config.

    Args:
        quiet: Suppress output below WARNING.
        verbose: Show DEBUG messages.

    Returns:
        The root ``libreyolo`` logger.
    """
    logger = logging.getLogger("libreyolo")

    # Set level
    if quiet:
        logger.setLevel(logging.WARNING)
    elif verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on repeated calls
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler → stderr (stdout is reserved for API results)
    use_colors = sys.stderr.isatty()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ConsoleFormatter(colors=use_colors))
    logger.addHandler(handler)

    # Don't propagate to root logger
    logger.propagate = False

    # Add header() convenience method
    def header(msg, *args, **kwargs):
        if logger.isEnabledFor(HEADER_LEVEL):
            logger._log(HEADER_LEVEL, msg, args, **kwargs)

    if not hasattr(logger, "header"):
        logger.header = header

    return logger


def ensure_default_logging() -> logging.Logger:
    """Install a minimal default logger for Python API usage when unconfigured.

    This intentionally does nothing when either the ``libreyolo`` logger or the
    process root logger already has handlers, so host applications keep control
    of logging when they have configured it themselves.
    """
    logger = logging.getLogger("libreyolo")
    root_logger = logging.getLogger()

    if logger.handlers or root_logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ConsoleFormatter(colors=sys.stderr.isatty()))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
