"""Tests for declared dependency floors."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import pytest

pytestmark = pytest.mark.unit


def test_rfdetr_extra_uses_native_dependencies():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["optional-dependencies"]["rfdetr"]
    assert "transformers>=4.40.0" in deps
    assert "scipy>=1.7.0" in deps
    assert all(not dep.startswith("rfdetr") for dep in deps)


def test_torch_floor_supports_weights_only_load():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    assert "torch>=1.13.0" in deps
