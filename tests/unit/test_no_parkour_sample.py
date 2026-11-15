"""Guard: the parkour photo must never return to user-facing material.

The only showcase/sample image is ``libreyolo/assets/guggenheim-bilbao.jpg``.
A frozen copy of the old parkour photo lives in ``tests/data/`` for parity and
golden tests only; ``docs/provenance/`` may still cite it as a test fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import libreyolo

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BANNED = "park" + "our"
_SCANNED_DIRS = ("libreyolo", "docs", "notebooks", "scripts", "examples", "skills", "tools")
_ALLOWED_DIRS = (_REPO_ROOT / "docs" / "provenance",)
_SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}


def _scanned_files() -> list[Path]:
    files = sorted(_REPO_ROOT.glob("README*"))
    for name in _SCANNED_DIRS:
        root = _REPO_ROOT / name
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or _SKIP_PARTS.intersection(path.parts):
                continue
            if any(path.is_relative_to(allowed) for allowed in _ALLOWED_DIRS):
                continue
            files.append(path)
    return files


def test_no_parkour_reference_in_user_facing_files():
    offenders = []
    for path in _scanned_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if _BANNED in rel.lower():
            offenders.append(rel)
            continue
        if path.stat().st_size > 5_000_000:
            continue  # weights or other large binaries, never prose
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary asset: covered by the filename check above
        if _BANNED in text.lower():
            offenders.append(rel)
    assert not offenders, (
        "The only sample image is libreyolo/assets/guggenheim-bilbao.jpg; "
        f"remove these references: {offenders}"
    )


def test_no_parkour_file_in_package_assets():
    assets = _REPO_ROOT / "libreyolo" / "assets"
    found = [p.name for p in assets.iterdir() if _BANNED in p.name.lower()]
    assert not found, f"parkour files must not ship in libreyolo/assets: {found}"


def test_sample_image_is_guggenheim():
    path = Path(libreyolo.SAMPLE_IMAGE)
    assert path.name == "guggenheim-bilbao.jpg"
    assert path.is_file()
