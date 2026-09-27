from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"


def _load_parity_module(name: str):
    weights_path = str(WEIGHTS_DIR)
    if weights_path not in sys.path:
        sys.path.insert(0, weights_path)
    module_path = WEIGHTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hvi_parity_enforces_source_and_checkpoint_pins(monkeypatch, tmp_path):
    parity = _load_parity_module("parity_hvi_cidnet")
    monkeypatch.setattr(
        parity.subprocess,
        "check_output",
        lambda *args, **kwargs: "not-the-pinned-commit\n",
    )
    with pytest.raises(RuntimeError, match=parity.UPSTREAM_COMMIT):
        parity._assert_upstream_pin(tmp_path)

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"audited test bytes")
    monkeypatch.setattr(parity, "OFFICIAL_SIZE", checkpoint.stat().st_size)
    monkeypatch.setattr(
        parity,
        "OFFICIAL_SHA256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    parity._assert_checkpoint_pin(checkpoint)
    monkeypatch.setattr(parity, "OFFICIAL_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        parity._assert_checkpoint_pin(checkpoint)


def test_ddcolor_parity_verifies_artifact_before_loading(monkeypatch):
    parity = _load_parity_module("parity_ddcolor")
    calls = []

    def reject_unpinned(path, size):
        calls.append((path, size))
        raise ValueError("not an audited artifact")

    monkeypatch.setattr(parity, "_resolve_official_artifact", reject_unpinned)
    with pytest.raises(ValueError, match="not an audited artifact"):
        parity.run_parity(
            "untrusted.bin",
            size="t",
            upstream_dir="unused",
        )
    assert calls == [("untrusted.bin", "t")]


def test_vitmatte_parity_enforces_checkout_and_import_origin(monkeypatch, tmp_path):
    parity = _load_parity_module("parity_vitmatte")
    monkeypatch.setattr(
        parity.subprocess,
        "check_output",
        lambda *args, **kwargs: "not-the-pinned-commit\n",
    )
    with pytest.raises(RuntimeError, match=parity.TRANSFORMERS_UPSTREAM_COMMIT):
        parity._assert_upstream_pin(tmp_path)

    package = tmp_path / "src" / "transformers"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(parity, "_assert_upstream_pin", lambda path: None)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    fake = SimpleNamespace(
        __file__=str(tmp_path / "somewhere-else" / "transformers" / "__init__.py")
    )
    monkeypatch.setattr(parity.importlib, "import_module", lambda name: fake)
    with pytest.raises(RuntimeError, match="not imported from the pinned"):
        parity._load_pinned_transformers(str(tmp_path))

    expected = (object(), object(), object())
    pinned = SimpleNamespace(
        __file__=str(package / "__init__.py"),
        VitMatteConfig=expected[0],
        VitMatteForImageMatting=expected[1],
        VitMatteImageProcessor=expected[2],
    )
    monkeypatch.setattr(parity.importlib, "import_module", lambda name: pinned)
    assert parity._load_pinned_transformers(str(tmp_path)) == expected
