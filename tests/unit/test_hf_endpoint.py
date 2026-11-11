"""HF_ENDPOINT support: Hub URLs follow the variable, failures point at it."""

import socket
import urllib.error

import pytest
import requests

from libreyolo.models.base.model import BaseModel
from libreyolo.utils import download, hf_hub
from libreyolo.utils.hf_hub import (
    apply_hf_endpoint,
    canonical_hf_url,
    hf_base_url,
    hf_unreachable_hint,
    hf_url,
)

pytestmark = pytest.mark.unit

MIRROR = "https://hf-mirror.com"


@pytest.fixture
def no_endpoint(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)


@pytest.fixture
def mirror(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", MIRROR + "/")


# --- URL building -----------------------------------------------------------


def test_base_url_defaults_to_huggingface(no_endpoint):
    assert hf_base_url() == "https://huggingface.co"
    assert hf_url("LibreYOLO/x/resolve/main/x.pt") == (
        "https://huggingface.co/LibreYOLO/x/resolve/main/x.pt"
    )


def test_base_url_follows_hf_endpoint_and_strips_slash(mirror):
    assert hf_base_url() == MIRROR
    assert hf_url("/LibreYOLO/x/resolve/main/x.pt") == (
        f"{MIRROR}/LibreYOLO/x/resolve/main/x.pt"
    )


def test_blank_hf_endpoint_is_ignored(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "  ")
    assert hf_base_url() == "https://huggingface.co"


def test_apply_and_canonical_round_trip(mirror):
    canonical = "https://huggingface.co/datasets/LibreYOLO/coco8/resolve/main/coco8.zip"
    mirrored = apply_hf_endpoint(canonical)
    assert mirrored == f"{MIRROR}/datasets/LibreYOLO/coco8/resolve/main/coco8.zip"
    assert canonical_hf_url(mirrored) == canonical
    # Other hosts are never touched.
    other = "https://github.com/x/y/releases/download/v1/a.pt"
    assert apply_hf_endpoint(other) == other
    assert canonical_hf_url(other) == other


def test_apply_is_identity_without_endpoint(no_endpoint):
    url = "https://huggingface.co/LibreYOLO/a/resolve/main/a.pt"
    assert apply_hf_endpoint(url) == url
    assert canonical_hf_url(url) == url


@pytest.mark.parametrize("endpoint_set", [False, True])
def test_family_download_urls_follow_endpoint(monkeypatch, endpoint_set):
    from libreyolo.models.dinov2.model import LibreDINOv2
    from libreyolo.models.yolo9.model import LibreYOLO9

    if endpoint_set:
        monkeypatch.setenv("HF_ENDPOINT", MIRROR)
        base = MIRROR
    else:
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        base = "https://huggingface.co"

    assert LibreYOLO9.get_download_url("LibreYOLO9t.pt") == (
        f"{base}/LibreYOLO/LibreYOLO9t/resolve/main/LibreYOLO9t.pt"
    )
    legacy = next(iter(LibreDINOv2.LEGACY_SEMANTIC_WEIGHTS))
    assert LibreDINOv2.get_download_url(legacy).startswith(f"{base}/LibreYOLO/")


def test_marigold_urls_follow_endpoint_and_still_verify(mirror):
    from libreyolo.models.marigold_v2 import config

    variant = next(iter(config.MIRRORS))
    assert config.mirror_url(variant).startswith(f"{MIRROR}/LibreYOLO/")
    assert config.upstream_url(variant).startswith(f"{MIRROR}/")


def test_moge2_mirror_url_passes_the_verify_route(mirror, tmp_path):
    from libreyolo.models.moge2.model import LibreMoGe2

    size = next(iter(LibreMoGe2._MIRROR_SHA256))
    url = LibreMoGe2.get_download_url(f"LibreMoGe2{size}-normal.pt")
    assert url.startswith(f"{MIRROR}/LibreYOLO/")
    # The file's digest is wrong on purpose: reaching the checksum error
    # proves the mirror URL was recognised as LibreYOLO's own mirror.
    bogus = tmp_path / "w.pt"
    bogus.write_bytes(b"x")
    with pytest.raises(ValueError, match="mirrored checkpoint SHA-256 mismatch"):
        LibreMoGe2.verify_downloaded_file(str(bogus), url)


def test_huggingface_hub_honours_hf_endpoint():
    """hf_hub_download/snapshot_download stay untouched: the library itself
    reads HF_ENDPOINT (at import) and strips the trailing slash."""
    import os
    import subprocess
    import sys

    pytest.importorskip("huggingface_hub")
    env = dict(os.environ, HF_ENDPOINT=MIRROR + "/")
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from huggingface_hub import hf_hub_url; "
            "print(hf_hub_url('LibreYOLO/x', 'x.pt'))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == f"{MIRROR}/LibreYOLO/x/resolve/main/x.pt"


# --- download path ------------------------------------------------------------


class _Family:
    @classmethod
    def get_download_url(cls, _filename):
        return "https://huggingface.co/LibreYOLO/test/resolve/main/model.pt"

    @classmethod
    def get_download_notice(cls, _filename, _url):
        return None

    seen_verify_urls: list = []

    @classmethod
    def verify_downloaded_file(cls, _path, url):
        cls.seen_verify_urls.append(url)


class _Response:
    status_code = 200
    headers = {"content-length": "3", "etag": '"v1"'}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield b"abc"

    def close(self):
        pass


def _prepare(monkeypatch):
    monkeypatch.setattr(BaseModel, "_registry", [_Family])
    monkeypatch.setattr(download, "_get_hf_token", lambda: "hf_secret")
    monkeypatch.setattr(download.time, "sleep", lambda _s: None)
    _Family.seen_verify_urls = []


def test_canonical_url_is_fetched_from_mirror_without_token(
    monkeypatch, tmp_path, mirror
):
    _prepare(monkeypatch)
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs["headers"]))
        return _Response()

    monkeypatch.setattr(download.requests, "get", fake_get)
    target = tmp_path / "model.pt"
    download.download_weights(str(target), "s")

    assert target.read_bytes() == b"abc"
    assert calls[0][0] == f"{MIRROR}/LibreYOLO/test/resolve/main/model.pt"
    # The HF token is never sent to a third-party mirror.
    assert "Authorization" not in calls[0][1]
    # verify hooks still see the URL they built.
    assert _Family.seen_verify_urls == [
        "https://huggingface.co/LibreYOLO/test/resolve/main/model.pt"
    ]


def _fail_with(monkeypatch, exc):
    _prepare(monkeypatch)
    monkeypatch.setattr(download, "_DOWNLOAD_RETRIES", 0)

    def fake_get(_url, **_kwargs):
        raise exc

    monkeypatch.setattr(download.requests, "get", fake_get)


@pytest.mark.parametrize(
    "exc",
    [
        requests.ConnectionError("Max retries exceeded"),
        requests.ConnectTimeout("connect timed out"),
        requests.ReadTimeout("read timed out"),
    ],
    ids=("connection", "connect-timeout", "read-timeout"),
)
def test_connection_failure_suggests_hf_endpoint(
    monkeypatch, tmp_path, no_endpoint, exc
):
    _fail_with(monkeypatch, exc)
    with pytest.raises(RuntimeError) as info:
        download.download_weights(str(tmp_path / "model.pt"), "s")
    message = str(info.value)
    assert "huggingface.co may be unreachable from your network" in message
    assert "export HF_ENDPOINT=https://hf-mirror.com" in message
    assert "community mirror" in message


def test_no_hint_when_hf_endpoint_already_set(monkeypatch, tmp_path, mirror):
    _fail_with(monkeypatch, requests.ConnectionError("down"))
    with pytest.raises(RuntimeError) as info:
        download.download_weights(str(tmp_path / "model.pt"), "s")
    assert "HF_ENDPOINT" not in str(info.value)


def test_no_hint_for_http_errors(monkeypatch, tmp_path, no_endpoint):
    _fail_with(monkeypatch, requests.HTTPError("HTTP 500"))
    with pytest.raises(RuntimeError) as info:
        download.download_weights(str(tmp_path / "model.pt"), "s")
    assert "HF_ENDPOINT" not in str(info.value)


def test_factory_error_carries_the_hint(monkeypatch, tmp_path, no_endpoint):
    import libreyolo.models as models

    _fail_with(monkeypatch, requests.ConnectionError("down"))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="export HF_ENDPOINT="):
        models.LibreYOLO("model.pt", size="s")


def test_hint_rules(no_endpoint):
    wrapped = RuntimeError("outer")
    wrapped.__cause__ = socket.timeout("timed out")
    assert hf_unreachable_hint(wrapped)
    assert hf_unreachable_hint(urllib.error.URLError("no route"))
    assert hf_unreachable_hint(ValueError("bad")) == ""
    # Not a Hub URL: the mirror would not help.
    assert (
        hf_unreachable_hint(
            requests.ConnectionError("x"), "https://github.com/a/b/c.pt"
        )
        == ""
    )


def test_hub_cache_miss_error_carries_the_hint(monkeypatch, no_endpoint):
    pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import LocalEntryNotFoundError

    def fake_download(**_kwargs):
        try:
            raise requests.ConnectionError("blocked")
        except requests.ConnectionError as e:
            raise LocalEntryNotFoundError("not cached") from e

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    ref = hf_hub.HubRef(repo_id="someone/model", filename="model.pt")
    with pytest.raises(ConnectionError, match="export HF_ENDPOINT="):
        hf_hub.resolve_hub_checkpoint(ref)
