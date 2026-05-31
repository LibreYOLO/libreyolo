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
    assert "transformers>=5.1.0" in deps
    assert "scipy>=1.7.0" not in deps
    assert all(not dep.startswith("rfdetr") for dep in deps)


def test_core_dependencies_include_import_chain_requirements():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    assert "Pillow>=9.1.0" in deps
    assert "scipy>=1.7.0" in deps
    assert "torchvision>=0.19.0" in deps


def test_torch_floor_supports_amp_grad_scaler():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    assert "torch>=2.4.0" in deps
