"""The CLI's user-facing text must survive legacy Windows console codepages.

A U+2014 em dash in the root help string rendered ``libreyolo --help`` as
``LibreYOLO ? open source YOLO detection toolkit`` on cp1252 consoles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

CLI_ROOT = Path(__file__).resolve().parents[3] / "libreyolo" / "cli"


def test_root_help_strings_are_ascii():
    from libreyolo.cli import _root, app

    assert app.info.help.isascii()
    assert _root.__doc__.isascii()
    assert "LibreYOLO" in app.info.help


def test_no_em_or_en_dashes_in_cli_string_literals():
    """Scan every CLI source file: no U+2013/U+2014 outside comments.

    Comments never reach a console; help strings, docstrings (typer renders
    command docstrings as help), and print/echo literals all do.
    """
    import io
    import tokenize

    offenders = []
    for path in sorted(CLI_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                continue
            if any(ord(ch) > 127 for ch in token.string):
                offenders.append(f"{path.name}:{token.start[0]}")
    assert not offenders, f"non-ASCII in CLI strings: {offenders}"
