"""
Data utilities for LibreYOLO.

Provides dataset auto-download, path resolution, and configuration loading.

Supports YAML configs with:
- Directory paths (e.g., val: images/val)
- Text file paths (e.g., val: val2017.txt)
- List paths (e.g., val: [path1, path2])
- Python download scripts
"""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import requests
import yaml
from tqdm import tqdm


# Default datasets directory (can be overridden via environment variable)
DATASETS_DIR = Path(os.getenv("LIBREYOLO_DATASETS_DIR", Path.home() / "datasets"))

# Built-in datasets directory (shipped with package)
BUILTIN_DATASETS_DIR = Path(__file__).parent.parent / "cfg" / "datasets"

# Supported image extensions
IMG_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".pfm"}


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


def get_img_files(path: Union[str, Path, List], prefix: str = "") -> List[Path]:
    """
    Get list of image files from various input formats.

    Supports:
    1. Directory path - recursively finds all images
    2. Text file (.txt) - reads paths line by line
    3. List of paths - processes each element

    Args:
        path: Directory path, .txt file path, or list of paths.
        prefix: Optional prefix to prepend to relative paths.

    Returns:
        List of resolved Path objects to image files.

    Raises:
        FileNotFoundError: If path doesn't exist.
        ValueError: If no valid images found.
    """
    if isinstance(path, list):
        # Handle list of paths recursively
        img_files = []
        for p in path:
            img_files.extend(get_img_files(p, prefix))
        return img_files

    path = Path(path)

    # Apply prefix if path is relative
    if prefix and not path.is_absolute():
        path = Path(prefix) / path

    if path.is_dir():
        # Directory: recursively find all images
        img_files = []
        for ext in IMG_FORMATS:
            img_files.extend(path.rglob(f"*{ext}"))
            img_files.extend(path.rglob(f"*{ext.upper()}"))
        return sorted(set(img_files))

    elif path.suffix.lower() == ".txt":
        # Text file: read paths line by line
        if not path.exists():
            raise FileNotFoundError(f"Image list file not found: {path}")

        img_files = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Handle relative paths in txt file
                    img_path = Path(line)
                    if not img_path.is_absolute():
                        # Relative to txt file's parent directory
                        img_path = path.parent / img_path
                    if img_path.exists() and img_path.suffix.lower() in IMG_FORMATS:
                        img_files.append(img_path.resolve())
        return sorted(img_files)

    elif path.suffix.lower() in IMG_FORMATS:
        # Single image file
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        return [path.resolve()]

    else:
        raise ValueError(f"Unsupported path format: {path}")


def img2label_paths(img_paths: List[Path]) -> List[Path]:
    """
    Convert image paths to corresponding label paths.

    Convention:
    - Replace 'images' with 'labels' in path
    - Change extension to '.txt'

    Args:
        img_paths: List of image file paths.

    Returns:
        List of corresponding label file paths.

    Example:
        >>> img_paths = [Path("/data/images/train/001.jpg")]
        >>> label_paths = img2label_paths(img_paths)
        >>> print(label_paths)  # [Path("/data/labels/train/001.txt")]
    """
    label_paths = []
    for img_path in img_paths:
        # Convert path to string for replacement
        path_str = str(img_path)

        # Replace 'images' with 'labels' (handles various positions)
        # Common patterns: /images/, \images\, /images, images/
        for sep in [os.sep, "/"]:
            path_str = path_str.replace(f"{sep}images{sep}", f"{sep}labels{sep}")
            path_str = path_str.replace(f"{sep}images", f"{sep}labels")

        # Change extension to .txt
        label_path = Path(path_str).with_suffix(".txt")
        label_paths.append(label_path)

    return label_paths


def load_data_config(data: str, autodownload: bool = True, allow_scripts: bool = True) -> Dict:
    """
    Load dataset configuration from YAML file.

    Supports YAML formats:
    - Directory paths: train/val point to directories (e.g., "images/train")
    - File list paths: train/val can be .txt files (e.g., "train2017.txt")

    If the dataset doesn't exist locally and a download URL/script is specified
    in the YAML, it will be automatically downloaded.

    Args:
        data: Dataset name (e.g., "coco8") or path to YAML file.
        autodownload: Whether to auto-download missing datasets.
        allow_scripts: Whether to allow execution of Python download scripts
            embedded in YAML configs. When False, only URL-based downloads are
            permitted and script-based downloads are skipped with a warning.

    Returns:
        Dictionary with dataset configuration including:
        - path: Dataset root path
        - train/val/test: Resolved paths (directory or .txt file)
        - {split}_img_files: List of image paths (if .txt format detected)
        - {split}_label_files: List of label paths (if .txt format detected)
        - names: Class names dict or list
        - nc: Number of classes

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
        config = check_dataset(config, yaml_path, allow_scripts=allow_scripts)

    # Resolve train/val/test paths
    for split in ("train", "val", "test"):
        if split in config and config[split]:
            split_value = config[split]
            split_path = dataset_path / split_value
            config[split] = str(split_path)

            # If it's a .txt file, pre-resolve image files
            if str(split_value).endswith(".txt") and split_path.exists():
                try:
                    img_files = get_img_files(split_path)
                    config[f"{split}_img_files"] = img_files
                    config[f"{split}_label_files"] = img2label_paths(img_files)
                except (FileNotFoundError, ValueError):
                    # File doesn't exist or no valid images found
                    pass

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


def check_dataset(config: Dict, yaml_path: Path = None, allow_scripts: bool = True) -> Dict:
    """
    Check if dataset exists, download if missing and URL/script is provided.

    Supports:
    - URL downloads (ZIP files)
    - Python script execution (multiline download scripts, requires allow_scripts=True)

    Args:
        config: Dataset configuration dictionary.
        yaml_path: Path to the YAML file (for script context).
        allow_scripts: Whether to allow execution of Python download scripts.
            When False, script-based downloads are skipped with a warning.

    Returns:
        Updated configuration with resolved paths.
    """
    dataset_path = Path(config["path"])

    # Check if dataset exists by looking for train/val paths
    exists = False
    for split in ("train", "val"):
        if split in config and config[split]:
            split_path = dataset_path / config[split]
            # Check if it's a directory with contents or a .txt file that exists
            if split_path.is_dir() and any(split_path.iterdir()):
                exists = True
                break
            elif split_path.is_file():
                exists = True
                break

    if exists:
        return config

    # Dataset doesn't exist, check for download URL/script
    download_spec = config.get("download")
    if not download_spec:
        print(f"Warning: Dataset not found at {dataset_path} and no download specified.")
        return config

    print(f"Dataset not found at {dataset_path}")

    # Check if download is a Python script (multiline or contains Python code)
    if _is_python_script(download_spec):
        if not allow_scripts:
            print(
                f"Warning: Dataset YAML contains a Python download script but "
                f"allow_scripts=False. Skipping script execution. "
                f"Pass allow_scripts=True to load_data_config() to enable."
            )
            return config
        import logging
        logging.getLogger(__name__).info(
            "Executing embedded download script from %s", yaml_path or "config"
        )
        _execute_download_script(download_spec, config, yaml_path)
    else:
        # Treat as URL
        download_dataset(download_spec, dataset_path)

    return config


def _is_python_script(download_spec: str) -> bool:
    """Check if download specification is a Python script."""
    # Check for common Python patterns
    python_indicators = [
        "import ",
        "from ",
        "def ",
        "exec(",
        "download(",
        "\n",  # Multiline usually indicates script
    ]
    return any(indicator in download_spec for indicator in python_indicators)


def _execute_download_script(script: str, config: Dict, yaml_path: Path = None) -> None:
    """
    Execute a Python download script.

    The script has access to:
    - yaml: The full config dict
    - path: The dataset root path
    - Path: pathlib.Path class

    Args:
        script: Python code to execute.
        config: Dataset configuration.
        yaml_path: Path to YAML file.
    """
    print("Executing download script...")

    # Auto-replace common import patterns with libreyolo equivalents
    script = script.replace(
        "from ultralytics.utils.downloads import download",
        "from libreyolo.utils.downloads import download"
    )
    script = script.replace(
        "from ultralytics.utils import ASSETS_URL",
        "from libreyolo.utils.downloads import ASSETS_URL"
    )

    # Create execution context with useful variables
    context = {
        "yaml": config,
        "path": Path(config["path"]),
        "yaml_file": yaml_path,
        "Path": Path,
        "os": os,
    }

    try:
        exec(script, context)
        print("Download script completed.")
    except Exception as e:
        print(f"Warning: Download script failed: {e}")
        raise


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
