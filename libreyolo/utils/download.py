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
    if re.search(r"libreyolo9", fl):
        return "yolo9"
    return None


def download_weights(model_path: str, size: str):
    """Download weights from Hugging Face if not found locally."""
    path = Path(model_path)
    if path.exists():
        return

    filename = path.name
    fl = filename.lower()

    # RF-DETR: LibreRFDETR(n|s|m|l).pth
    m = re.search(r"librerfdetr([nsml])\.pth", fl)
    if m:
        letter = m.group(1)
        repo = f"LibreYOLO/LibreRFDETR{letter}"
        actual_filename = f"LibreRFDETR{letter}.pth"
        url = f"https://huggingface.co/{repo}/resolve/main/{actual_filename}"
    # YOLOX: LibreYOLOX(n|t|s|m|l|x).pt
    elif re.search(r"libreyolox([ntslmx])\.pt", fl):
        yolox_m = re.search(r"libreyolox([ntslmx])\.pt", fl)
        letter = yolox_m.group(1)
        # HF repos use full names for n/t variants
        _YOLOX_HF_NAMES = {"n": "nano", "t": "tiny"}
        hf_letter = _YOLOX_HF_NAMES.get(letter, letter)
        repo = f"LibreYOLO/LibreYOLOX{hf_letter}"
        actual_filename = f"LibreYOLOX{letter}.pt"
        url = f"https://huggingface.co/{repo}/resolve/main/{actual_filename}"
    # YOLOv9: LibreYOLO9(t|s|m|c).pt
    elif re.search(r"libreyolo9([tsmc])\.pt", fl):
        yolo9_m = re.search(r"libreyolo9([tsmc])\.pt", fl)
        letter = yolo9_m.group(1)
        repo = f"LibreYOLO/LibreYOLO9{letter}"
        actual_filename = f"LibreYOLO9{letter}.pt"
        url = f"https://huggingface.co/{repo}/resolve/main/{actual_filename}"
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
                            f"\rDownloading: {percent}% ({downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB)",
                            end="",
                            flush=True,
                        )
            print("\nDownload complete.")
    except Exception as e:
        if path.exists():
            path.unlink()
        raise RuntimeError(f"Failed to download weights from {url}: {e}") from e
