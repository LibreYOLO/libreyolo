"""Download helpers for LibreYOLO model weights."""

import json
import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

_YOLONAS_LICENSE_NOTICE_SHOWN = False
_DOWNLOAD_RETRIES = 3
_DOWNLOAD_BACKOFF_SECONDS = 1.0
_DOWNLOAD_TIMEOUT = (10, 60)
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_DOWNLOAD_LOCK_TIMEOUT = 6 * 60 * 60
_DOWNLOAD_LOCK_POLL_SECONDS = 0.1
# Statuses that mean the object is not published (Hugging Face answers 401 for
# a repo that does not exist or is private). Retrying cannot fix them.
_UNPUBLISHED_STATUS_CODES = frozenset({401, 403, 404, 410})
logger = logging.getLogger(__name__)


class WeightsNotPublishedError(FileNotFoundError):
    """Raised when a weight URL answers with a permanent 'not there' status."""

    def __init__(self, url: str, status_code: int, *, used_token: bool = False):
        self.url = url
        self.status_code = status_code
        token_hint = (
            " The request sent your Hugging Face token; an invalid HF_TOKEN "
            "also causes this."
            if used_token and status_code == 401
            else ""
        )
        super().__init__(
            f"No weights are published at {url} (HTTP {status_code}): the "
            "file does not exist or is not public. Pass the path to a local "
            "checkpoint instead, or pick a published weight name "
            f"(`libreyolo models` lists the families).{token_hint}"
        )


def _get_hf_token() -> Optional[str]:
    """Get HuggingFace token from env var or cached login."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def _notify_deci_cdn_license_once(url: str = "") -> None:
    """Print the governing weight-license notice once per process.

    Deci's CDN serves artifacts under more than one set of terms. The YOLO-NAS
    checkpoints carry Deci's proprietary weight license; the DEKR checkpoint
    comes from the Apache-2.0 SuperGradients repository but has no per-artifact
    redistribution grant of its own, so it gets its own honest wording rather
    than borrowing YOLO-NAS's proprietary text.
    """
    global _YOLONAS_LICENSE_NOTICE_SHOWN
    if _YOLONAS_LICENSE_NOTICE_SHOWN:
        return
    _YOLONAS_LICENSE_NOTICE_SHOWN = True
    rule = "─" * 69
    if "dekr_" in url.lower():
        print(
            "\n"
            f"{rule}\n"
            "DEKR weights are downloaded from Deci.AI's public CDN. The source\n"
            "code is Apache-2.0, but no per-artifact redistribution grant was\n"
            "found for this checkpoint, so LibreYOLO links to it rather than\n"
            "mirroring it. Review the source repository's terms before\n"
            "redistributing the file or anything derived from it:\n"
            "  https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.md\n"
            f"{rule}\n"
        )
        return
    is_rotated = "yolo_nas_r_" in url.lower()
    name = "YOLO-NAS-R (rotated)" if is_rotated else "YOLO-NAS"
    license_file = "LICENSE.YOLONAS-R.md" if is_rotated else "LICENSE.YOLONAS.md"
    rule = "─" * 69
    print(
        "\n"
        f"{rule}\n"
        f"{name} weights are distributed by Deci.AI under a proprietary\n"
        "license (non-commercial, no redistribution, no production use\n"
        "without a separate agreement). By downloading, you accept those\n"
        "terms. Full license text:\n"
        f"  https://github.com/Deci-AI/super-gradients/blob/master/{license_file}\n"
        f"{rule}\n"
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


def _response_total_size(response, offset: int) -> int:
    """Return the complete object size for a full or ranged response."""
    content_range = response.headers.get("content-range", "")
    match = re.match(r"bytes\s+\d+-\d+/(\d+|\*)", content_range, re.IGNORECASE)
    if match and match.group(1) != "*":
        return int(match.group(1))

    content_length = int(response.headers.get("content-length", 0))
    if content_length <= 0:
        return 0
    if response.status_code == 206:
        return offset + content_length
    return content_length


def _partial_metadata_path(partial: Path) -> Path:
    return partial.with_name(partial.name + ".json")


def _reset_partial(partial: Path) -> None:
    partial.unlink(missing_ok=True)
    _partial_metadata_path(partial).unlink(missing_ok=True)


def _response_validator(response) -> str | None:
    """Return a validator suitable for an If-Range request."""
    etag = response.headers.get("etag")
    if etag and not etag.lstrip().lower().startswith("w/"):
        return etag
    return response.headers.get("last-modified")


def _load_partial_validator(partial: Path, url: str) -> str | None:
    metadata = _partial_metadata_path(partial)
    try:
        state = json.loads(metadata.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    if state.get("url") != url:
        return None
    validator = state.get("validator")
    return validator if isinstance(validator, str) and validator else None


def _save_partial_validator(partial: Path, url: str, validator: str | None) -> None:
    metadata = _partial_metadata_path(partial)
    if validator is None:
        metadata.unlink(missing_ok=True)
        return
    temporary = metadata.with_name(metadata.name + ".tmp")
    temporary.write_text(
        json.dumps({"url": url, "validator": validator}),
        encoding="utf-8",
    )
    os.replace(temporary, metadata)


def _try_lock_file(lock_file) -> None:
    lock_file.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(lock_file) -> None:
    lock_file.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _download_lock(path: Path):
    """Serialize downloads targeting the same final weights path."""
    lock_path = path.with_name(path.name + ".download.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as lock_file:
        if lock_file.seek(0, os.SEEK_END) == 0:
            lock_file.write(b"\0")
            lock_file.flush()

        deadline = time.monotonic() + _DOWNLOAD_LOCK_TIMEOUT
        while True:
            try:
                _try_lock_file(lock_file)
                break
            except OSError as e:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for download lock: {lock_path}"
                    ) from e
                time.sleep(_DOWNLOAD_LOCK_POLL_SECONDS)

        try:
            yield
        finally:
            _unlock_file(lock_file)


def _download_once(url: str, partial: Path, headers: dict[str, str]) -> None:
    """Stream one request, resuming a partial file when the server permits it."""
    offset = partial.stat().st_size if partial.exists() else 0
    validator = _load_partial_validator(partial, url) if offset else None
    if offset and validator is None:
        logger.warning(
            "Partial download has no matching HTTP validator; restarting from byte 0."
        )
        _reset_partial(partial)
        offset = 0

    request_headers = dict(headers)
    if offset:
        assert validator is not None
        request_headers["Range"] = f"bytes={offset}-"
        request_headers["If-Range"] = validator

    response = requests.get(
        url,
        stream=True,
        headers=request_headers,
        timeout=_DOWNLOAD_TIMEOUT,
    )
    try:
        if response.status_code == 416 and offset:
            response_validator = _response_validator(response)
            complete_range = re.match(
                r"bytes\s+\*/(\d+)",
                response.headers.get("content-range", ""),
                re.IGNORECASE,
            )
            same_object = response_validator == validator
            if (
                same_object
                and complete_range
                and int(complete_range.group(1)) == offset
            ):
                return

            # The saved offset is invalid for this object. Remove only this
            # download's temporary file so the retry starts cleanly.
            _reset_partial(partial)

        if response.status_code in _UNPUBLISHED_STATUS_CODES:
            raise WeightsNotPublishedError(
                url,
                response.status_code,
                used_token="Authorization" in request_headers,
            )
        response.raise_for_status()

        append = offset > 0 and response.status_code == 206
        response_validator = _response_validator(response)
        if append and response_validator not in {None, validator}:
            _reset_partial(partial)
            raise IOError("Download object changed while resuming")

        if offset and not append:
            logger.warning(
                "Download server ignored the resume request; restarting from byte 0."
            )
            offset = 0
            _reset_partial(partial)

        total_size = _response_total_size(response, offset)
        downloaded = offset
        last_logged = -1
        mode = "ab" if append else "wb"
        if not append:
            _save_partial_validator(partial, url, response_validator)
        with open(partial, mode) as f:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int(100 * downloaded / total_size)
                        if percent % 25 == 0 and percent != last_logged:
                            last_logged = percent
                            logger.info(
                                "Downloading: %d%% (%.1f/%.1f MB)",
                                percent,
                                downloaded / 1024 / 1024,
                                total_size / 1024 / 1024,
                            )

        if total_size > 0 and downloaded != total_size:
            raise IOError(
                f"Incomplete download: got {downloaded} of {total_size} bytes"
            )
    finally:
        response.close()


def download_weights(model_path: str, size: str):
    """Download weights from Hugging Face if not found locally."""
    path = Path(model_path)
    if path.exists():
        # A concurrent downloader may have atomically placed the file but still
        # be running its family-specific verification hook.
        with _download_lock(path):
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

    notice = cls.get_download_notice(path.name, url)
    if notice:
        logger.warning(notice)

    download_url_to_path(url, path, verify=cls.verify_downloaded_file)


def download_url_to_path(url: str, path: Path, *, verify=None) -> None:
    """Download ``url`` to ``path`` with locking, resume, retries and verify.

    ``verify`` is an optional ``(local_path, source_url) -> None`` hook run on
    the temporary file before it is atomically published at ``path``.
    """
    logger.info(
        "Model weights not found at %s. Attempting download from %s...",
        path,
        url,
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    host = urlparse(url).netloc
    is_hf = host.endswith("huggingface.co")

    # The same CDN also serves PP-YOLOE, whose weights are covered by neither
    # the YOLO-NAS nor the DEKR notice, so key the notice on the object name
    # rather than on the host alone.
    object_name = url.lower()
    if ("cloudfront.net" in host or host.endswith("deci.ai")) and (
        "yolo_nas" in object_name or "dekr_" in object_name
    ):
        _notify_deci_cdn_license_once(url)

    headers = {}
    token = _get_hf_token()
    if token and is_hf:
        # Only attach the HF token to HF URLs — never leak it to third parties.
        headers["Authorization"] = f"Bearer {token}"
    elif is_hf and not token:
        logger.info(
            "Tip: Run `huggingface-cli login` or set HF_TOKEN for faster downloads."
        )

    # Stream to a temp file and rename at the end so a killed process can
    # never leave a truncated weight at the final path (loading one fails
    # with an opaque zip error and requires a manual delete).
    partial = path.with_name(path.name + ".part")
    with _download_lock(path):
        # Another process may have completed the same download while this
        # process waited for the lock.
        if path.exists():
            return

        for attempt in range(_DOWNLOAD_RETRIES + 1):
            try:
                _download_once(url, partial, headers)
                break
            except WeightsNotPublishedError:
                raise
            except Exception as e:
                if attempt == _DOWNLOAD_RETRIES:
                    partial_size = partial.stat().st_size if partial.exists() else 0
                    raise RuntimeError(
                        f"Failed to download weights from {url} after "
                        f"{_DOWNLOAD_RETRIES + 1} attempts: {e}. "
                        f"Partial download kept at {partial} ({partial_size} bytes)."
                    ) from e

                partial_size = partial.stat().st_size if partial.exists() else 0
                logger.warning(
                    "Download interrupted (%s). Retrying %d/%d from byte %d.",
                    e,
                    attempt + 1,
                    _DOWNLOAD_RETRIES,
                    partial_size,
                )
                time.sleep(_DOWNLOAD_BACKOFF_SECONDS * (2**attempt))

        # Verify the temporary file before publishing it at the final path.
        if verify is not None:
            try:
                verify(str(partial), url)
            except Exception:
                _reset_partial(partial)
                raise
        os.replace(partial, path)
        _partial_metadata_path(partial).unlink(missing_ok=True)
        logger.info("Download complete.")
