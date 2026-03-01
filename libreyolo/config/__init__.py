"""
Configuration files for LibreYOLO.

Contains default dataset configurations and model settings.
"""

from pathlib import Path

# Path to built-in datasets directory
DATASETS_DIR = Path(__file__).parent / "datasets"


def list_datasets() -> list:
    """List all built-in dataset configurations."""
    return [f.stem for f in DATASETS_DIR.glob("*.yaml")]
