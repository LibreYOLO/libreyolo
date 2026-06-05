import os
import urllib.request
from urllib.parse import urlparse

import pytest


_LOCAL_HTTP_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _is_external_http_url(url) -> bool:
    raw_url = getattr(url, "full_url", url)
    parsed = urlparse(str(raw_url))
    return parsed.scheme in {"http", "https"} and parsed.hostname not in _LOCAL_HTTP_HOSTS


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end (requires GPU and datasets)"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (fast, no external deps)"
    )
    config.addinivalue_line(
        "markers", "external_data: test requires external datasets, weights, or staged large files"
    )
    config.addinivalue_line(
        "markers", "network: test intentionally uses non-local network access"
    )


@pytest.fixture(autouse=True)
def _block_external_http_in_pr_gate(request, monkeypatch):
    """Keep PR-gate unit tests hermetic while allowing localhost sockets."""
    if os.environ.get("LIBREYOLO_PR_GATE") != "1":
        return
    if request.node.get_closest_marker("network"):
        return

    def fail_if_external(url):
        if _is_external_http_url(url):
            pytest.fail(
                "External HTTP is blocked in the LibreYOLO PR gate. "
                "Use a local fixture, or mark the test as network and keep it "
                "out of the PR-gate marker expression."
            )

    try:
        import requests
    except ImportError:
        requests = None

    if requests is not None:
        original_request = requests.sessions.Session.request

        def guarded_request(self, method, url, *args, **kwargs):
            fail_if_external(url)
            return original_request(self, method, url, *args, **kwargs)

        monkeypatch.setattr(requests.sessions.Session, "request", guarded_request)

    original_urlopen = urllib.request.urlopen
    original_urlretrieve = urllib.request.urlretrieve

    def guarded_urlopen(url, *args, **kwargs):
        fail_if_external(url)
        return original_urlopen(url, *args, **kwargs)

    def guarded_urlretrieve(url, *args, **kwargs):
        fail_if_external(url)
        return original_urlretrieve(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded_urlopen)
    monkeypatch.setattr(urllib.request, "urlretrieve", guarded_urlretrieve)
