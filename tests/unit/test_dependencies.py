"""Tests for declared dependency floors."""

from pathlib import Path
import tomllib

import pytest

pytestmark = pytest.mark.unit


def test_rfdetr_floor_matches_known_required_symbol_version():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["optional-dependencies"]["rfdetr"]
    assert "rfdetr>=1.6.2,<2.0.0" in deps
