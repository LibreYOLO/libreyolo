"""Structured error handling for the LibreYOLO CLI."""

import difflib
from typing import Optional


# Exit code mapping: error category → exit code
EXIT_CODES: dict[str, int] = {
    # Runtime errors
    "device_not_available": 1,
    "cuda_oom": 1,
    "training_diverged": 1,
    "download_failed": 1,
    "io_error": 1,
    # Usage errors
    "config_unknown_key": 2,
    "config_type_error": 2,
    "config_range_error": 2,
    "config_required_key": 2,
    "config_conflict": 2,
    "config_unsupported": 2,
    # Data errors
    "source_not_found": 3,
    "data_not_found": 3,
    "data_invalid": 3,
    "data_images_missing": 3,
    # Model errors
    "model_not_found": 4,
    "model_load_failed": 4,
    "model_family_mismatch": 4,
    "checkpoint_not_found": 4,
    # Export errors
    "export_format_unknown": 5,
    "export_dep_missing": 5,
    "format_precision_unsupported": 5,
}


class CLIError(Exception):
    """Structured CLI error with code, message, and optional suggestion."""

    def __init__(
        self, code: str, message: str, suggestion: Optional[str] = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggestion = suggestion
        self.exit_code = EXIT_CODES.get(code, 1)


def suggest_key(unknown: str, valid_keys: list[str]) -> Optional[str]:
    """Find closest match for an unknown key using difflib."""
    matches = difflib.get_close_matches(unknown, valid_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None
