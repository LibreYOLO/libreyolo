"""Tests for retrying and resumable weight downloads."""

import multiprocessing
import queue
import time
from pathlib import Path

import pytest
import requests
import torch

from libreyolo.models.base.model import BaseModel
from libreyolo.models.deim.model import LibreDEIM
from libreyolo.models.deimv2.model import LibreDEIMv2
from libreyolo.models.dfine.model import LibreDFINE
from libreyolo.models.ec.model import LibreEC
from libreyolo.utils import download

pytestmark = pytest.mark.unit

_PROCESS_SYNC_TIMEOUT = 30


class _DownloadFamily:
    @classmethod
    def get_download_url(cls, _filename):
        return "https://huggingface.co/LibreYOLO/test/resolve/model.pt"

    @classmethod
    def get_download_notice(cls, _filename, _url):
        return None

    @classmethod
    def verify_downloaded_file(cls, _path, _url):
        return None


class _Response:
    def __init__(self, chunks, *, status_code=200, headers=None):
        self._chunks = chunks
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        assert chunk_size == download._DOWNLOAD_CHUNK_SIZE
        for chunk in self._chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


def _prepare(monkeypatch):
    monkeypatch.setattr(BaseModel, "_registry", [_DownloadFamily])
    monkeypatch.setattr(download, "_get_hf_token", lambda: None)
    monkeypatch.setattr(download.time, "sleep", lambda _seconds: None)


def _hold_download_lock(target, entered, release):
    with download._download_lock(Path(target)):
        entered.put(time.monotonic())
        release.wait(timeout=_PROCESS_SYNC_TIMEOUT)


def test_interrupted_download_retries_from_partial(monkeypatch, tmp_path):
    _prepare(monkeypatch)
    responses = [
        _Response(
            [b"abcd", requests.ConnectionError("connection dropped")],
            headers={"content-length": "10", "etag": '"version-1"'},
        ),
        _Response(
            [b"efghij"],
            status_code=206,
            headers={
                "content-length": "6",
                "content-range": "bytes 4-9/10",
                "etag": '"version-1"',
            },
        ),
    ]
    calls = []

    def fake_get(url, *, stream, headers, timeout):
        calls.append(
            {"url": url, "stream": stream, "headers": headers, "timeout": timeout}
        )
        return responses.pop(0)

    monkeypatch.setattr(download.requests, "get", fake_get)
    target = tmp_path / "model.pt"

    download.download_weights(str(target), "s")

    assert target.read_bytes() == b"abcdefghij"
    assert calls[0]["headers"].get("Range") is None
    assert calls[1]["headers"]["Range"] == "bytes=4-"
    assert calls[1]["headers"]["If-Range"] == '"version-1"'
    assert calls[1]["timeout"] == download._DOWNLOAD_TIMEOUT
    assert not target.with_name("model.pt.part").exists()
    assert not target.with_name("model.pt.part.json").exists()


def test_partial_without_validator_restarts(monkeypatch, tmp_path):
    _prepare(monkeypatch)
    target = tmp_path / "model.pt"
    partial = target.with_name("model.pt.part")
    partial.write_bytes(b"stale")
    response = _Response([b"complete"], headers={"content-length": "8"})
    calls = []

    def fake_get(_url, **kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(download.requests, "get", fake_get)

    download.download_weights(str(target), "s")

    assert target.read_bytes() == b"complete"
    assert "Range" not in calls[0]["headers"]


def test_changed_object_restarts_instead_of_mixing_bytes(monkeypatch, tmp_path):
    _prepare(monkeypatch)
    responses = [
        _Response(
            [b"old-", requests.ConnectionError("connection dropped")],
            headers={"content-length": "8", "etag": '"version-1"'},
        ),
        # A conforming server ignores Range and returns the complete new object
        # when If-Range does not match.
        _Response(
            [b"new-data"],
            headers={"content-length": "8", "etag": '"version-2"'},
        ),
    ]
    calls = []

    def fake_get(_url, **kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr(download.requests, "get", fake_get)
    target = tmp_path / "model.pt"

    download.download_weights(str(target), "s")

    assert calls[1]["headers"]["Range"] == "bytes=4-"
    assert calls[1]["headers"]["If-Range"] == '"version-1"'
    assert target.read_bytes() == b"new-data"


def test_complete_partial_is_finalized_after_range_416(monkeypatch, tmp_path):
    _prepare(monkeypatch)
    target = tmp_path / "model.pt"
    partial = target.with_name("model.pt.part")
    partial.write_bytes(b"complete")
    download._save_partial_validator(
        partial,
        _DownloadFamily.get_download_url(target.name),
        '"version-1"',
    )
    response = _Response(
        [],
        status_code=416,
        headers={"content-range": "bytes */8", "etag": '"version-1"'},
    )
    calls = []

    def fake_get(_url, **kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(download.requests, "get", fake_get)

    download.download_weights(str(target), "s")

    assert target.read_bytes() == b"complete"
    assert calls[0]["headers"]["If-Range"] == '"version-1"'


def test_range_416_with_changed_validator_restarts(monkeypatch, tmp_path):
    _prepare(monkeypatch)
    target = tmp_path / "model.pt"
    partial = target.with_name("model.pt.part")
    partial.write_bytes(b"old-data")
    download._save_partial_validator(
        partial,
        _DownloadFamily.get_download_url(target.name),
        '"version-1"',
    )
    responses = [
        _Response(
            [],
            status_code=416,
            headers={"content-range": "bytes */8", "etag": '"version-2"'},
        ),
        _Response(
            [b"new-data"],
            headers={"content-length": "8", "etag": '"version-2"'},
        ),
    ]
    monkeypatch.setattr(
        download.requests,
        "get",
        lambda _url, **_kwargs: responses.pop(0),
    )

    download.download_weights(str(target), "s")

    assert target.read_bytes() == b"new-data"


def test_exhausted_download_keeps_partial(monkeypatch, tmp_path):
    _prepare(monkeypatch)
    monkeypatch.setattr(download, "_DOWNLOAD_RETRIES", 0)
    response = _Response(
        [b"partial", requests.ConnectionError("still offline")],
        headers={"content-length": "20", "etag": '"version-1"'},
    )
    monkeypatch.setattr(
        download.requests,
        "get",
        lambda _url, **_kwargs: response,
    )
    target = tmp_path / "model.pt"

    with pytest.raises(RuntimeError, match="Partial download kept"):
        download.download_weights(str(target), "s")

    assert target.with_name("model.pt.part").read_bytes() == b"partial"
    assert target.with_name("model.pt.part.json").exists()


def test_failed_verification_removes_final_file(monkeypatch, tmp_path):
    _prepare(monkeypatch)

    def fail_verification(_cls, _path, _url):
        raise ValueError("checksum mismatch")

    monkeypatch.setattr(
        _DownloadFamily,
        "verify_downloaded_file",
        classmethod(fail_verification),
    )
    monkeypatch.setattr(
        download.requests,
        "get",
        lambda _url, **_kwargs: _Response(
            [b"complete"],
            headers={"content-length": "8", "etag": '"version-1"'},
        ),
    )
    target = tmp_path / "model.pt"

    with pytest.raises(ValueError, match="checksum mismatch"):
        download.download_weights(str(target), "s")

    assert not target.exists()


def test_download_lock_serializes_processes(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    entered = ctx.Queue()
    release_first = ctx.Event()
    release_second = ctx.Event()
    target = tmp_path / "model.pt"
    first = ctx.Process(
        target=_hold_download_lock,
        args=(str(target), entered, release_first),
    )
    second = ctx.Process(
        target=_hold_download_lock,
        args=(str(target), entered, release_second),
    )

    try:
        first.start()
        entered.get(timeout=_PROCESS_SYNC_TIMEOUT)
        second.start()
        with pytest.raises(queue.Empty):
            entered.get(timeout=0.3)

        release_first.set()
        entered.get(timeout=_PROCESS_SYNC_TIMEOUT)
        release_second.set()
        first.join(timeout=_PROCESS_SYNC_TIMEOUT)
        second.join(timeout=_PROCESS_SYNC_TIMEOUT)
        assert first.exitcode == 0
        assert second.exitcode == 0
    finally:
        release_first.set()
        release_second.set()
        for process in (first, second):
            if process.pid is None:
                continue
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def test_factory_reports_and_chains_download_failure(monkeypatch, tmp_path):
    import libreyolo.models as models

    failure = RuntimeError("connection reset while fetching checkpoint")

    def fail_download(_model_path, _size):
        raise failure

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(models, "download_weights", fail_download)

    with pytest.raises(
        FileNotFoundError,
        match="Auto-download failed: connection reset while fetching checkpoint",
    ) as exc_info:
        models.LibreYOLO("LibreYOLO9t.pt")

    assert exc_info.value.__cause__ is failure


@pytest.mark.parametrize(
    ("model_cls", "filename", "size"),
    [
        (LibreDFINE, "LibreDFINEn.pt", "n"),
        (LibreDEIM, "LibreDEIMn.pt", "n"),
        (LibreDEIMv2, "LibreDEIMv2atto.pt", "atto"),
        (LibreEC, "LibreECs.pt", "s"),
    ],
    ids=("dfine", "deim", "deimv2", "ec"),
)
def test_direct_constructor_attempts_autodownload(
    model_cls, filename, size, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    target = Path("weights") / filename
    calls = []

    failure = RuntimeError("connection reset while fetching checkpoint")

    def fake_download(model_path, requested_size):
        calls.append((Path(model_path), requested_size))
        raise failure

    monkeypatch.setattr(model_cls, "_init_model", lambda _self: torch.nn.Identity())
    monkeypatch.setattr(download, "download_weights", fake_download)

    with pytest.raises(
        FileNotFoundError,
        match="Auto-download failed: connection reset while fetching checkpoint",
    ) as exc_info:
        model_cls(filename, size=size, device="cpu")

    assert calls == [(target, size)]
    assert exc_info.value.__cause__ is failure


@pytest.mark.parametrize("status_code", [401, 404])
def test_unpublished_weights_fail_fast_with_clear_error(
    monkeypatch, tmp_path, status_code
):
    _prepare(monkeypatch)
    calls = []

    def fake_get(url, **_kwargs):
        calls.append(url)
        return _Response([], status_code=status_code)

    monkeypatch.setattr(download.requests, "get", fake_get)
    target = tmp_path / "model.pt"

    with pytest.raises(
        download.WeightsNotPublishedError,
        match=rf"No weights are published at .*\(HTTP {status_code}\)",
    ) as exc_info:
        download.download_weights(str(target), "s")

    assert isinstance(exc_info.value, FileNotFoundError)
    assert "HF_TOKEN" not in str(exc_info.value)
    assert len(calls) == 1
    assert not target.exists()
    assert not target.with_name("model.pt.part").exists()


def test_unpublished_weights_error_mentions_token_when_one_was_sent(
    monkeypatch, tmp_path
):
    _prepare(monkeypatch)
    monkeypatch.setattr(download, "_get_hf_token", lambda: "hf_test")
    monkeypatch.setattr(
        download.requests,
        "get",
        lambda _url, **_kwargs: _Response([], status_code=401),
    )

    with pytest.raises(download.WeightsNotPublishedError, match="HF_TOKEN"):
        download.download_weights(str(tmp_path / "model.pt"), "s")
