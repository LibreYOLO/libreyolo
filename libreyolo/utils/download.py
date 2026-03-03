"""Download helpers for LibreYOLO model weights."""

from pathlib import Path

import requests


def download_weights(model_path: str, size: str):
    """Download weights from Hugging Face if not found locally."""
    path = Path(model_path)
    if path.exists():
        return

    from libreyolo.models.base.model import BaseModel

    url = None
    for cls in BaseModel._registry:
        url = cls.get_download_url(path.name)
        if url:
            break

    if url is None:
        raise ValueError(f"Could not determine download URL for '{path.name}'.")

    print(f"Model weights not found at {model_path}. Attempting download from {url}...")
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int(100 * downloaded / total_size)
                        print(
                            f"\rDownloading: {percent}% ({downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB)",
                            end="",
                            flush=True,
                        )
            print("\nDownload complete.")
    except Exception as e:
        if path.exists():
            path.unlink()
        raise RuntimeError(f"Failed to download weights from {url}: {e}") from e
