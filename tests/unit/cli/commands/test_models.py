"""`libreyolo models` advertises only names the CLI can load."""

import json

import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands.special import models_cmd
from libreyolo.cli.config import get_all_cli_names, weight_unavailable_reason
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def rows():
    application = typer.Typer()
    application.command("models", cls=KeyValueCommand)(models_cmd)
    result = CliRunner().invoke(application, ["--json"])
    assert result.exit_code == 0, result.output
    return {row["name"]: row for row in json.loads(result.stdout)["families"]}


def test_every_advertised_name_resolves_to_a_download(rows):
    known = set(get_all_cli_names())
    for row in rows.values():
        if row["cli_command"]:
            continue
        for name in row["cli_names"]:
            assert name in known, name
            assert weight_unavailable_reason(name) is None, name


def test_unpublished_names_are_listed_with_their_reason(rows):
    assert "yolo1-t" not in rows["yolo1"]["cli_names"]
    assert "lost upstream" in rows["yolo1"]["unpublished_names"]["yolo1-t"]
    assert "eomt-s-seg" in rows["eomt"]["unpublished_names"]
    assert "eomt-s-panoptic" in rows["eomt"]["cli_names"]


def test_python_only_families_advertise_no_cli_names(rows):
    if "sam2" not in rows:
        pytest.skip("SAM 2 adapter not importable here")
    sam2 = rows["sam2"]
    assert sam2["cli_names"] == []
    assert sam2["python_class"].endswith(".LibreSAM2")


@pytest.mark.parametrize(
    ("family", "class_name", "extra", "tasks"),
    [
        (
            "sensenovavision",
            "LibreSenseNovaVision",
            "sensenova",
            ["detect", "segment", "panoptic", "pose", "point", "depth", "ocr"],
        ),
        ("libremodus", "LibreMODUS", "modus", ["detect", "depth", "normal", "edge"]),
    ],
)
def test_lazy_multimodal_families_are_listed(rows, family, class_name, extra, tasks):
    row = rows[family]
    assert row["python_class"].endswith(f".{class_name}")
    assert row["tasks"] == tasks
    assert row["cli_names"] == []
    assert row["optional_extra"] == extra
    assert row["install_hint"] == f"pip install libreyolo[{extra}]"


def test_command_families_keep_their_command(rows):
    assert rows["3dmood"]["cli_names"] == ["3dmood"]
    assert rows["3dmood"]["cli_command"] == "3dmood"
