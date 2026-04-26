"""Download helpers for LibreYOLO model weights."""

import logging
import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

_YOLONAS_LICENSE_NOTICE_SHOWN = False
logger = logging.getLogger(__name__)


def _get_hf_token() -> Optional[str]:
    """Get HuggingFace token from env var or cached login."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def _notify_yolonas_license_once() -> None:
    """Print Deci's YOLO-NAS license terms once per process before download."""
    global _YOLONAS_LICENSE_NOTICE_SHOWN
    if _YOLONAS_LICENSE_NOTICE_SHOWN:
        return
    _YOLONAS_LICENSE_NOTICE_SHOWN = True
    print(
        "\n"
        "─────────────────────────────────────────────────────────────────────\n"
        "YOLO-NAS weights are distributed by Deci.AI under a proprietary\n"
        "license (non-commercial, no redistribution, no production use\n"
        "without a separate agreement). By downloading, you accept those\n"
        "terms. Full license text:\n"
        "  https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md\n"
        "─────────────────────────────────────────────────────────────────────\n"
    )


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

    logger.info(
        "Model weights not found at %s. Attempting download from %s...",
        model_path,
        url,
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    host = urlparse(url).netloc
    is_hf = host.endswith("huggingface.co")

    if "cloudfront.net" in host or host.endswith("deci.ai"):
        _notify_yolonas_license_once()

    headers = {}
    token = _get_hf_token()
    if token and is_hf:
        # Only attach the HF token to HF URLs — never leak it to third parties.
        headers["Authorization"] = f"Bearer {token}"
    elif is_hf and not token:
        logger.info("Tip: Run `huggingface-cli login` or set HF_TOKEN for faster downloads.")

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
                        if percent % 25 == 0:
                            logger.info(
                                "Downloading: %d%% (%.1f/%.1f MB)",
                                percent,
                                downloaded / 1024 / 1024,
                                total_size / 1024 / 1024,
                            )
            logger.info("Download complete.")
    except Exception as e:
        if path.exists():
            path.unlink()
        raise RuntimeError(f"Failed to download weights from {url}: {e}") from e
