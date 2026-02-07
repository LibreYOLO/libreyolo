"""Unit tests for factory module path resolution."""

import os
import tempfile
from pathlib import Path

import pytest

from libreyolo.factory import _resolve_weights_path

pytestmark = pytest.mark.unit


class TestResolveWeightsPath:
    """Test _resolve_weights_path bare-filename → weights/ resolution."""

    def test_bare_filename_no_file_anywhere(self):
        """Bare filename with no existing file resolves to weights/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = _resolve_weights_path("model.pt")
                assert result == str(Path("weights") / "model.pt")
            finally:
                os.chdir(orig)

    def test_bare_filename_exists_in_weights(self):
        """Bare filename found in weights/ returns weights/ path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            try:
                os.chdir(tmpdir)
                (Path("weights")).mkdir()
                (Path("weights") / "model.pt").touch()
                result = _resolve_weights_path("model.pt")
                assert result == str(Path("weights") / "model.pt")
            finally:
                os.chdir(orig)

    def test_bare_filename_exists_in_cwd_only(self):
        """Bare filename found in CWD (but not weights/) falls back to CWD."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            try:
                os.chdir(tmpdir)
                Path("model.pt").touch()
                result = _resolve_weights_path("model.pt")
                assert result == "model.pt"
            finally:
                os.chdir(orig)

    def test_bare_filename_exists_in_both_prefers_weights(self):
        """When file exists in both weights/ and CWD, weights/ wins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            try:
                os.chdir(tmpdir)
                Path("model.pt").touch()
                (Path("weights")).mkdir()
                (Path("weights") / "model.pt").touch()
                result = _resolve_weights_path("model.pt")
                assert result == str(Path("weights") / "model.pt")
            finally:
                os.chdir(orig)

    def test_explicit_dir_unchanged(self):
        """Path with directory component is returned as-is."""
        result = _resolve_weights_path("my_dir/model.pt")
        assert result == "my_dir/model.pt"

    def test_absolute_path_unchanged(self):
        """Absolute path is returned as-is."""
        result = _resolve_weights_path("/tmp/model.pt")
        assert result == "/tmp/model.pt"

    def test_dot_slash_prefix_unchanged(self):
        """Explicit ./filename is returned as-is (has directory component)."""
        result = _resolve_weights_path("./model.pt")
        assert result == "./model.pt"

    def test_onnx_extension(self):
        """Works for .onnx files too."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = _resolve_weights_path("model.onnx")
                assert result == str(Path("weights") / "model.onnx")
            finally:
                os.chdir(orig)

    def test_engine_extension(self):
        """Works for .engine files too."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = _resolve_weights_path("model.engine")
                assert result == str(Path("weights") / "model.engine")
            finally:
                os.chdir(orig)
