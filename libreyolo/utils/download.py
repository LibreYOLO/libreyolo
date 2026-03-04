"""Download helpers for LibreYOLO model weights."""

import os
import re
from pathlib import Path
from typing import Optional

import requests


def _get_hf_token() -> Optional[str]:
    """Get HuggingFace token from env var or cached login."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def _detect_family_from_filename(filename: str) -> Optional[str]:
    """Return model family hint from filename (for download routing only)."""
    fl = filename.lower()
    if re.search(r"librerfdetr", fl):
        return "rfdetr"
    if re.search(r"libreyolox", fl):
        return "yolox"
    if re.search(r"libreyolo9", fl):
        return "yolo9"
    return None



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

    # RF-DETR is lazily registered — try loading it if no match yet
    if url is None:
        try:
            from libreyolo.models import _ensure_rfdetr

            _ensure_rfdetr()
            for cls in BaseModel._registry:
                url = cls.get_download_url(path.name)
                if url:
                    break
        except (ModuleNotFoundError, ImportError):
            pass

    if url is None:
        raise ValueError(f"Could not determine download URL for '{path.name}'.")

    print(f"Model weights not found at {model_path}. Attempting download from {url}...")
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    token = _get_hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        print("Tip: Run `huggingface-cli login` or set HF_TOKEN for faster downloads.")

    try:
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
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
