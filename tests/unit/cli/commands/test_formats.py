"""`libreyolo formats --family` reports only precisions export() accepts."""

import json

import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands.special import formats_cmd
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit


def _formats(*args):
    application = typer.Typer()
    application.command("formats", cls=KeyValueCommand)(formats_cmd)
    result = CliRunner().invoke(application, [*args, "--json"])
    assert result.exit_code == 0, result.output
    return {row["name"]: row for row in json.loads(result.stdout)["formats"]}


def test_gtr_reports_fp32_only():
    rows = _formats("--family", "gtr", "--task", "semantic")
    assert rows["onnx"]["tier"] == "available"
    assert (rows["onnx"]["fp16"], rows["onnx"]["int8"]) == (False, False)


def test_blocked_rows_report_no_precision():
    rows = _formats("--family", "gtr", "--task", "semantic")
    assert rows["openvino"]["tier"] == "blocked"
    assert (rows["openvino"]["fp16"], rows["openvino"]["int8"]) == (False, False)


def test_onnx_int8_follows_the_exporter_gate():
    assert _formats("--family", "yolo9", "--task", "detect")["onnx"]["int8"] is True
    assert _formats("--family", "yolox", "--task", "detect")["onnx"]["int8"] is False


def test_without_family_rows_describe_the_format():
    rows = _formats()
    assert (rows["onnx"]["fp16"], rows["onnx"]["int8"]) == (True, True)
