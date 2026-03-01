"""Download helpers for LibreYOLO model weights."""

import re
from pathlib import Path
from typing import Optional

import requests


def _detect_family_from_filename(filename: str) -> Optional[str]:
    """Return model family hint from filename (for download routing only)."""
    fl = filename.lower()
    if re.search(r"librerfdetr", fl):
        return "rfdetr"
    if re.search(r"libreyolox", fl):
        return "yolox"
    if re.search(r"libreyolo9|yolov?9", fl):
        return "yolo9"
    return None


def download_weights(model_path: str, size: str):
    """Download weights from Hugging Face if not found locally."""
    path = Path(model_path)
    if path.exists():
        return

    filename = path.name

    # RF-DETR
    if re.search(r"librerfdetr(nano|small|medium|large)", filename.lower()):
        m = re.search(r"librerfdetr(nano|small|medium|large)", filename.lower())
        rfdetr_size = m.group(1)
        repo = f"Libre-YOLO/librerfdetr{rfdetr_size}"
        if rfdetr_size == "large":
            actual_filename = "rf-detr-large-2026.pth"
        else:
            actual_filename = f"rf-detr-{rfdetr_size}.pth"
        url = f"https://huggingface.co/{repo}/resolve/main/{actual_filename}"
    # YOLOX
    elif re.search(r"libreyolox(nano|tiny|s|m|l|x)", filename.lower()):
        yolox_match = re.search(r"libreyolox(nano|tiny|s|m|l|x)", filename.lower())
        yolox_size = yolox_match.group(1)
        repo = f"Libre-YOLO/libreyoloX{yolox_size}"
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    # YOLOv9
    elif re.search(r"libreyolo9|yolov?9", filename.lower()):
        repo = f"Libre-YOLO/libreyolo9{size}"
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    else:
        raise ValueError(
            f"Could not determine model version from filename '{filename}' for auto-download."
        )

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
                            f"\rDownloading: {percent}% ({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)",
                            end="", flush=True,
                        )
            print("\nDownload complete.")
    except Exception as e:
        if path.exists():
            path.unlink()
        raise RuntimeError(f"Failed to download weights from {url}: {e}") from e
