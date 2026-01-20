"""
Data utilities for LibreYOLO.

Provides dataset configuration loading, auto-download, and path resolution.
"""

from .utils import (
    DATASETS_DIR,
    check_dataset,
    load_data_config,
    resolve_dataset_yaml,
    list_builtin_datasets,
    safe_download,
)

__all__ = [
    "DATASETS_DIR",
    "check_dataset",
    "load_data_config",
    "resolve_dataset_yaml",
    "list_builtin_datasets",
    "safe_download",
]
