"""Tests for KeyValueCommand — the key=value → --key value rewriting."""

import sys

import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli import _normalize_logging_flags, _setup_logging_from_argv
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit

runner = CliRunner()


def _make_app(**params):
    """Build a tiny Typer app with a single command for testing parse_args."""
    app = typer.Typer()
    captured = {}

    @app.command(cls=KeyValueCommand)
    def cmd(
        name: str = typer.Option("default"),
        count: int = typer.Option(0),
        lr: float = typer.Option(0.01),
        half: bool = typer.Option(False),
        save: bool = typer.Option(False),
    ):
        captured.update(
            {"name": name, "count": count, "lr": lr, "half": half, "save": save}
        )

    return app, captured


class TestKeyValueSyntax:
    """Test that key=value tokens are rewritten to --key value."""

    def test_basic_key_value(self):
        app, captured = _make_app()
        result = runner.invoke(app, ["name=hello", "count=42"])
        assert result.exit_code == 0
        assert captured["name"] == "hello"
        assert captured["count"] == 42

    def test_float_value(self):
        app, captured = _make_app()
        result = runner.invoke(app, ["lr=0.001"])
        assert result.exit_code == 0
        assert captured["lr"] == 0.001

    def test_underscore_converted_to_dash(self):
        """key_name=value should work (converted to --key-name)."""
        app = typer.Typer()
        captured = {}

        @app.command(cls=KeyValueCommand)
        def cmd(data_dir: str = typer.Option("default")):
            captured["data_dir"] = data_dir

        result = runner.invoke(app, ["data_dir=/tmp/data"])
        assert result.exit_code == 0
        assert captured["data_dir"] == "/tmp/data"


class TestStandardSyntax:
    """Test that --key value syntax still works alongside key=value."""

    def test_double_dash_key_value(self):
        app, captured = _make_app()
        result = runner.invoke(app, ["--name", "hello", "--count", "42"])
        assert result.exit_code == 0
        assert captured["name"] == "hello"
        assert captured["count"] == 42

    def test_double_dash_equals(self):
        app, captured = _make_app()
        result = runner.invoke(app, ["--name=hello", "--count=42"])
        assert result.exit_code == 0
        assert captured["name"] == "hello"
        assert captured["count"] == 42

    def test_mixed_syntax(self):
        app, captured = _make_app()
        result = runner.invoke(app, ["name=hello", "--count", "42", "lr=0.5"])
        assert result.exit_code == 0
        assert captured["name"] == "hello"
        assert captured["count"] == 42
        assert captured["lr"] == 0.5


class TestBoolFlags:
    """Test boolean flag handling: bare words, key=true, key=false."""

    def test_bare_word_sets_true(self):
        app, captured = _make_app()
        result = runner.invoke(app, ["half"])
        assert result.exit_code == 0
        assert captured["half"] is True

    def test_key_equals_true(self):
        app, captured = _make_app()
        result = runner.invoke(app, ["half=true"])
        assert result.exit_code == 0
        assert captured["half"] is True

    def test_key_equals_false(self):
        app, captured = _make_app()
        result = runner.invoke(app, ["half=false"])
        assert result.exit_code == 0
        assert captured["half"] is False

    def test_key_equals_True_capitalized(self):
        app, captured = _make_app()
        result = runner.invoke(app, ["save=True"])
        assert result.exit_code == 0
        assert captured["save"] is True

    def test_multiple_bare_bools(self):
        app, captured = _make_app()
        result = runner.invoke(app, ["half", "save"])
        assert result.exit_code == 0
        assert captured["half"] is True
        assert captured["save"] is True

    def test_mixed_bools_and_values(self):
        app, captured = _make_app()
        result = runner.invoke(app, ["name=test", "half", "count=5"])
        assert result.exit_code == 0
        assert captured["name"] == "test"
        assert captured["half"] is True
        assert captured["count"] == 5


class TestLoggingFlagNormalization:
    def test_normalize_logging_flags_rewrites_key_value_syntax(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["libreyolo", "predict", "quiet=true", "verbose=false", "model=yolox-s"],
        )

        _normalize_logging_flags()

        assert sys.argv == [
            "libreyolo",
            "predict",
            "--quiet",
            "--no-verbose",
            "model=yolox-s",
        ]

    def test_setup_logging_reads_normalized_quiet_flag(self, monkeypatch):
        calls = {}

        monkeypatch.setattr(sys, "argv", ["libreyolo", "predict", "--quiet"])
        monkeypatch.setattr(
            "libreyolo.utils.logging.setup_logging",
            lambda quiet, verbose: calls.update(
                {"quiet": quiet, "verbose": verbose}
            ),
        )

        _setup_logging_from_argv()

        assert calls == {"quiet": True, "verbose": False}


class TestEdgeCases:
    """Test edge cases in the key=value parser."""

    def test_value_with_equals_sign(self):
        """Values containing = should work (e.g. path=a=b)."""
        app = typer.Typer()
        captured = {}

        @app.command(cls=KeyValueCommand)
        def cmd(path: str = typer.Option("default")):
            captured["path"] = path

        result = runner.invoke(app, ["path=a=b"])
        assert result.exit_code == 0
        assert captured["path"] == "a=b"

    def test_non_bool_bare_word_not_converted(self):
        """A bare word that isn't a bool flag should not be silently consumed."""
        app, captured = _make_app()
        result = runner.invoke(app, ["name=ok", "unknown_word"])
        # Should fail because 'unknown_word' is not a known option or bool flag
        assert result.exit_code != 0

    def test_defaults_unchanged_when_no_args(self):
        app, captured = _make_app()
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert captured["name"] == "default"
        assert captured["count"] == 0
        assert captured["half"] is False
