"""Repository-level checks for capability language and status markers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_WORDS = ("experi" + "mental", "\u5b9e\u9a8c" + "\u6027")
_SHORT_STATUS = "ex" + "p"
_FORBIDDEN_STATUS_MARKERS = (
    f"`{_SHORT_STATUS}` means",
    f"<td>{_SHORT_STATUS}</td>",
    f"| {_SHORT_STATUS} |",
)


def _tracked_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        Path(raw.decode("utf-8"))
        for raw in result.stdout.split(b"\0")
        if raw
    ]


# The ban is on *labelling* a capability with the retired vocabulary. Two files
# have the opposite job: they record that the vocabulary was retired. A release
# that deletes the acknowledgement kwarg has to spell it out, or the user whose
# script now raises TypeError searches the docs for the argument in their own
# traceback and finds nothing. Keep those two exempt and the guard absolute
# everywhere else, including every other doc page, model card and source file.
# Reviewed again after the 1.5.0 cut (#744 item 17): the exemption stays.
# Naming the retired argument in the historical record is the point of those
# two files; widening the ban to them would hide the migration path.
_HISTORICAL_RECORD = ("CHANGELOG.md", "docs/upgrading.md")


def _is_historical_record(line: str) -> bool:
    path = line.split(":", 1)[0]
    return path in _HISTORICAL_RECORD


def _tracked_text_matches(patterns: tuple[str, ...]) -> list[str]:
    command = ["git", "grep", "-n", "-I", "-i", "-F"]
    for pattern in patterns:
        command.extend(("-e", pattern))
    command.extend(("--", "."))
    result = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode not in (0, 1):
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout.splitlines()


def test_tracked_tree_has_no_deprecated_capability_vocabulary() -> None:
    """Keep capability access separate from validation depth."""
    forbidden_words = tuple(word.casefold() for word in _FORBIDDEN_WORDS)
    violations = [
        str(path)
        for path in _tracked_paths()
        if any(word in path.as_posix().casefold() for word in forbidden_words)
    ]
    violations.extend(
        line
        for line in _tracked_text_matches(
            _FORBIDDEN_WORDS + _FORBIDDEN_STATUS_MARKERS
        )
        if not _is_historical_record(line)
    )

    assert not violations, "Forbidden capability labels found:\n" + "\n".join(
        violations
    )
