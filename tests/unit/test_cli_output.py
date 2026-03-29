"""Tests for CLI output routing."""

import json
import pytest

from libreyolo.cli.errors import CLIError
from libreyolo.cli.output import OutputHandler

pytestmark = pytest.mark.unit


class TestResultOutput:
    """Test result() routing to stdout."""

    def test_json_mode_adds_schema_version(self, capsys):
        out = OutputHandler(json_mode=True)
        out.result({"key": "value"})
        stdout = capsys.readouterr().out
        data = json.loads(stdout)
        assert data["schema_version"] == 1
        assert data["key"] == "value"

    def test_human_mode_uses_human_text(self, capsys):
        out = OutputHandler(json_mode=False)
        out.result({"_human_text": "hello world", "key": "value"})
        stdout = capsys.readouterr().out
        assert stdout.strip() == "hello world"

    def test_human_mode_fallback_key_value(self, capsys):
        out = OutputHandler(json_mode=False)
        out.result({"name": "test", "count": 42})
        stdout = capsys.readouterr().out
        assert "name: test" in stdout
        assert "count: 42" in stdout

    def test_human_mode_skips_underscore_keys(self, capsys):
        out = OutputHandler(json_mode=False)
        out.result({"_internal": "hidden", "visible": "yes"})
        stdout = capsys.readouterr().out
        assert "_internal" not in stdout
        assert "visible: yes" in stdout


class TestErrorOutput:
    """Test error() routing."""

    def test_json_error_to_stdout(self, capsys):
        out = OutputHandler(json_mode=True)
        err = CLIError("model_not_found", "not found", suggestion="check path")
        out.error(err)
        stdout = capsys.readouterr().out
        data = json.loads(stdout)
        assert data["error"] == "model_not_found"
        assert data["message"] == "not found"
        assert data["suggestion"] == "check path"
        assert data["schema_version"] == 1

    def test_human_error_to_stderr(self, capsys):
        """In human mode, errors go to stderr via logger.

        Since we don't set up the logger in unit tests, we just verify
        no crash and nothing goes to stdout.
        """
        out = OutputHandler(json_mode=False)
        err = CLIError("io_error", "disk full")
        out.error(err)
        captured = capsys.readouterr()
        # Nothing should go to stdout in human error mode
        assert captured.out == ""
