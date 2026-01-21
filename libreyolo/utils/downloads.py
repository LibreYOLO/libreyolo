"""
Download utilities for LibreYOLO.

Provides download function for dataset auto-download.
"""

import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Union

from libreyolo.data.utils import _download_file

# Assets URL for pre-converted COCO labels
ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"


def download(
    urls: Union[str, List[str]],
    dir: Union[str, Path] = ".",
    unzip: bool = True,
    delete: bool = True,
    threads: int = 1,
) -> None:
    """
    Download files from URLs with optional extraction.

    Args:
        urls: Single URL or list of URLs to download.
        dir: Destination directory.
        unzip: Extract zip files after download.
        delete: Delete zip files after extraction.
        threads: Number of parallel downloads (for multiple URLs).

    Example:
        >>> download(["http://example.com/data.zip"], dir="./datasets")
    """
    if isinstance(urls, str):
        urls = [urls]

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)

    def download_one(url: str) -> None:
        """Download and optionally extract a single URL."""
        filename = Path(url).name
        filepath = dir / filename

        # Download
        print(f"Downloading {filename}...")
        _download_file(url, filepath)

        # Extract if zip
        if unzip and filepath.suffix == ".zip":
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(filepath, "r") as zf:
                zf.extractall(dir)
            if delete:
                filepath.unlink()

    # Download all URLs
    if threads > 1 and len(urls) > 1:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            executor.map(download_one, urls)
    else:
        for url in urls:
            download_one(url)
