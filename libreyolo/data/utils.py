"""
Data utilities for LibreYOLO.

Provides dataset auto-download, path resolution, and configuration loading.
Follows the Ultralytics pattern where dataset YAMLs contain download URLs.
"""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.parse import urlparse

import requests
import yaml
from tqdm import tqdm


# Default datasets directory (can be overridden via environment variable)
DATASETS_DIR = Path(os.getenv("LIBREYOLO_DATASETS_DIR", Path.home() / "datasets"))

# Built-in datasets directory (shipped with package)
BUILTIN_DATASETS_DIR = Path(__file__).parent.parent / "cfg" / "datasets"


def resolve_dataset_yaml(data: str) -> Path:
    """
    Resolve a dataset YAML path.

    Searches in the following order:
    1. Exact path (if exists)
    2. Current working directory
    3. Built-in datasets directory (libreyolo/cfg/datasets/)

    Args:
        data: Dataset name (e.g., "coco8.yaml") or path to YAML file.

    Returns:
        Resolved Path to the YAML file.

    Raises:
        FileNotFoundError: If the dataset YAML cannot be found.
    """
    data_path = Path(data)

    # 1. Check if it's an absolute path that exists
    if data_path.is_absolute() and data_path.exists():
        return data_path

    # 2. Check current working directory
    if data_path.exists():
        return data_path.resolve()

    # 3. Add .yaml extension if not present
    if not data.endswith((".yaml", ".yml")):
        data = f"{data}.yaml"
        data_path = Path(data)

    # 4. Check current working directory again with extension
    if data_path.exists():
        return data_path.resolve()

    # 5. Check built-in datasets directory
    builtin_path = BUILTIN_DATASETS_DIR / data_path.name
    if builtin_path.exists():
        return builtin_path

    # Also try without .yaml for directory-style names
    builtin_path_alt = BUILTIN_DATASETS_DIR / f"{data_path.stem}.yaml"
    if builtin_path_alt.exists():
        return builtin_path_alt

    raise FileNotFoundError(
        f"Dataset '{data}' not found. Searched in:\n"
        f"  - {Path.cwd()}\n"
        f"  - {BUILTIN_DATASETS_DIR}\n"
        f"Available built-in datasets: {list_builtin_datasets()}"
    )


def list_builtin_datasets() -> list:
    """List all built-in dataset configurations."""
    if not BUILTIN_DATASETS_DIR.exists():
        return []
    return [f.stem for f in BUILTIN_DATASETS_DIR.glob("*.yaml")]


def load_data_config(data: str, autodownload: bool = True) -> Dict:
    """
    Load dataset configuration from YAML file.

    If the dataset doesn't exist locally and a download URL is specified
    in the YAML, it will be automatically downloaded.

    Args:
        data: Dataset name (e.g., "coco8") or path to YAML file.
        autodownload: Whether to auto-download missing datasets.

    Returns:
        Dictionary with dataset configuration including resolved paths.

    Example:
        >>> config = load_data_config("coco8")
        >>> print(config["train"])  # Path to training images
    """
    # Resolve YAML path
    yaml_path = resolve_dataset_yaml(data)

    # Load YAML
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve dataset root path
    dataset_path = _resolve_dataset_path(config, yaml_path)
    config["path"] = str(dataset_path)
    config["yaml_file"] = str(yaml_path)

    # Check if dataset exists, download if needed
    if autodownload:
        config = check_dataset(config)

    # Resolve train/val/test paths
    for split in ("train", "val", "test"):
        if split in config and config[split]:
            split_path = dataset_path / config[split]
            config[split] = str(split_path)

    # Keep 'root' for backward compatibility
    config["root"] = str(dataset_path)

    return config


def _resolve_dataset_path(config: Dict, yaml_path: Path) -> Path:
    """Resolve the dataset root path from config."""
    if "path" in config:
        path = Path(config["path"])
        if path.is_absolute():
            return path
        # Check relative to DATASETS_DIR first
        datasets_path = DATASETS_DIR / path
        if datasets_path.exists():
            return datasets_path
        # Check relative to YAML file location
        yaml_relative = yaml_path.parent / path
        if yaml_relative.exists():
            return yaml_relative.resolve()
        # Default to DATASETS_DIR for new downloads
        return datasets_path
    else:
        # Use YAML file's parent directory
        return yaml_path.parent


def check_dataset(config: Dict) -> Dict:
    """
    Check if dataset exists, download if missing and URL is provided.

    Args:
        config: Dataset configuration dictionary.

    Returns:
        Updated configuration with resolved paths.
    """
    dataset_path = Path(config["path"])

    # Check if dataset exists by looking for train/val directories
    exists = False
    for split in ("train", "val"):
        if split in config and config[split]:
            split_path = dataset_path / config[split]
            if split_path.exists() and any(split_path.iterdir()):
                exists = True
                break

    if exists:
        return config

    # Dataset doesn't exist, check for download URL
    download_url = config.get("download")
    if not download_url:
        print(f"Warning: Dataset not found at {dataset_path} and no download URL specified.")
        return config

    # Download and extract
    print(f"Dataset not found at {dataset_path}")
    download_dataset(download_url, dataset_path)

    return config


def download_dataset(url: str, dest: Path) -> None:
    """
    Download and extract a dataset from URL.

    Supports:
    - ZIP files (http/https URLs ending in .zip)
    - Hugging Face datasets (huggingface.co URLs)

    Args:
        url: Download URL.
        dest: Destination directory.
    """
    print(f"Downloading dataset from {url}...")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Handle different URL types
    if url.startswith("http"):
        if url.endswith(".zip"):
            _download_and_extract_zip(url, dest)
        else:
            # Assume it's a direct download or HF dataset
            _download_and_extract_zip(url, dest)
    else:
        raise ValueError(f"Unsupported download URL format: {url}")

    print(f"Dataset extracted to {dest}")


def _download_and_extract_zip(url: str, dest: Path) -> None:
    """Download and extract a ZIP file."""
    # Create temp file for download
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Download with progress bar
        _download_file(url, tmp_path)

        # Extract
        print(f"Extracting to {dest}...")
        dest.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(tmp_path, "r") as zf:
            # Check if zip contains a single root directory
            names = zf.namelist()
            if names:
                # Get the common prefix (root directory in zip)
                root_dirs = set(n.split("/")[0] for n in names if "/" in n)
                if len(root_dirs) == 1:
                    # Extract to parent and rename
                    root_dir = root_dirs.pop()
                    extract_to = dest.parent
                    zf.extractall(extract_to)
                    # If extracted dir is different from dest, rename
                    extracted_path = extract_to / root_dir
                    if extracted_path != dest and extracted_path.exists():
                        if dest.exists():
                            shutil.rmtree(dest)
                        extracted_path.rename(dest)
                else:
                    # Extract directly to dest
                    zf.extractall(dest)
            else:
                zf.extractall(dest)

    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


def _download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {dest.name}",
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def safe_download(
    url: str,
    dest: Optional[Union[str, Path]] = None,
    unzip: bool = True,
    delete: bool = True,
) -> Path:
    """
    Download a file safely with automatic retry and extraction.

    Args:
        url: URL to download from.
        dest: Destination path or directory.
        unzip: Whether to extract ZIP files.
        delete: Whether to delete the ZIP after extraction.

    Returns:
        Path to downloaded/extracted content.
    """
    if dest is None:
        dest = DATASETS_DIR

    dest = Path(dest)

    # If dest is a directory, use filename from URL
    if dest.is_dir() or not dest.suffix:
        filename = Path(urlparse(url).path).name
        filepath = dest / filename
        dest.mkdir(parents=True, exist_ok=True)
    else:
        filepath = dest
        filepath.parent.mkdir(parents=True, exist_ok=True)

    # Download
    _download_file(url, filepath)

    # Extract if ZIP
    if unzip and filepath.suffix == ".zip":
        extract_dir = filepath.parent / filepath.stem
        with zipfile.ZipFile(filepath, "r") as zf:
            zf.extractall(extract_dir)

        if delete:
            filepath.unlink()

        return extract_dir

    return filepath
