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

    def test_json_mode_strips_private_keys(self, capsys):
        out = OutputHandler(json_mode=True)
        out.result({"key": "value", "_human_text": "hidden", "_debug": "hidden"})
        stdout = capsys.readouterr().out
        data = json.loads(stdout)
        assert data["key"] == "value"
        assert "_human_text" not in data
        assert "_debug" not in data

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


_NOISY_JSON_COMMAND = '''
import os
import sys

import typer

from libreyolo.cli.output import OutputHandler
from libreyolo.cli.parsing import KeyValueCommand

app = typer.Typer()


@app.command("noisy", cls=KeyValueCommand)
def noisy(json_output: bool = typer.Option(False, "--json")) -> None:
    out = OutputHandler(json_mode=json_output)
    print("python-level chatter")
    sys.stdout.flush()
    os.write(1, b"fd-level chatter\\n")
    out.result({"ok": True})


@app.command("other")
def other() -> None:
    pass


app()
'''


def test_json_command_keeps_third_party_stdout_off_the_json_document(tmp_path):
    """Only the JSON document reaches stdout, even for C-level writes to fd 1."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    import libreyolo

    script = tmp_path / "noisy.py"
    script.write_text(_NOISY_JSON_COMMAND)
    repo_root = str(Path(libreyolo.__file__).resolve().parents[1])
    proc = subprocess.run(
        [sys.executable, str(script), "noisy", "json=true"],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": repo_root},
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"ok": True, "schema_version": 1}
    assert "python-level chatter" in proc.stderr
    assert "fd-level chatter" in proc.stderr


def test_json_command_diverts_prints_under_the_cli_runner():
    import typer
    from typer.testing import CliRunner

    from libreyolo.cli.parsing import KeyValueCommand

    app = typer.Typer()

    @app.command("noisy", cls=KeyValueCommand)
    def noisy(json_output: bool = typer.Option(False, "--json")) -> None:
        out = OutputHandler(json_mode=json_output)
        print(" Average Precision  (AP) @[ IoU=0.50:0.95 ] = 0.220")
        out.result({"ok": True})

    @app.command("other")
    def other() -> None:
        pass

    result = CliRunner().invoke(app, ["noisy", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["ok"] is True
    assert "Average Precision" in result.stderr

    human = CliRunner().invoke(app, ["noisy"])
    assert "Average Precision" in human.stdout
