from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
import yaml
from PIL import Image
from typer.testing import CliRunner

from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit
runner = CliRunner()


def _build_app() -> typer.Typer:
    from libreyolo.cli.commands import doctor

    app = typer.Typer(add_completion=False)
    app.command("doctor", cls=KeyValueCommand)(doctor.doctor_cmd)

    # A second command keeps Typer in subcommand mode, matching the real CLI
    # (a single-command app would swallow "doctor" as the DATA argument).
    @app.command("noop")
    def _noop() -> None:
        pass

    return app


def _parse_json_output(output: str) -> dict:
    for line in output.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"No JSON found in output:\n{output}")


def _make_dataset(root: Path, label_line: str = "0 0.5 0.5 0.2 0.2\n") -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (64, 64), (10, 200, 30)).save(
            root / "images" / split / "a.jpg"
        )
        (root / "labels" / split / "a.txt").write_text(label_line)
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": ["cat"],
            }
        )
    )
    return yaml_path


def test_doctor_healthy_dataset_exits_zero(tmp_path):
    yaml_path = _make_dataset(tmp_path / "ds")
    result = runner.invoke(_build_app(), ["doctor", str(yaml_path), "--fast"])
    assert result.exit_code == 0
    assert "LibreDoctor report" in result.output


def test_doctor_positional_and_keyvalue_forms(tmp_path):
    yaml_path = _make_dataset(tmp_path / "ds")
    positional = runner.invoke(_build_app(), ["doctor", str(yaml_path), "--fast"])
    keyvalue = runner.invoke(_build_app(), ["doctor", f"data={yaml_path}", "--fast"])
    assert positional.exit_code == 0
    assert keyvalue.exit_code == 0


def test_doctor_errors_exit_one_with_json(tmp_path):
    yaml_path = _make_dataset(tmp_path / "ds", label_line="7 0.5 0.5 0.2 0.2\n")
    result = runner.invoke(_build_app(), ["doctor", str(yaml_path), "--fast", "--json"])
    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    assert payload["summary"]["errors"] >= 1
    assert payload["schema_version"] == 1
    ids = {f["check_id"] for f in payload["findings"]}
    assert "labels.class_out_of_range" in ids


def test_doctor_strict_promotes_warnings(tmp_path):
    yaml_path = _make_dataset(tmp_path / "ds", label_line="0 0.5 0.5 0.99 0.99\n")
    relaxed = runner.invoke(_build_app(), ["doctor", str(yaml_path), "--fast"])
    strict = runner.invoke(
        _build_app(), ["doctor", str(yaml_path), "--fast", "--strict"]
    )
    assert relaxed.exit_code == 0
    assert strict.exit_code == 1


def test_doctor_missing_data_argument():
    result = runner.invoke(_build_app(), ["doctor"])
    assert result.exit_code != 0


def test_doctor_unknown_check_selector(tmp_path):
    yaml_path = _make_dataset(tmp_path / "ds")
    result = runner.invoke(
        _build_app(), ["doctor", str(yaml_path), "--fast", "skip=bogus"]
    )
    assert result.exit_code == 2


def test_doctor_pose_dataset_rejected(tmp_path):
    root = tmp_path / "ds"
    yaml_path = _make_dataset(root)
    config = yaml.safe_load(yaml_path.read_text())
    config["kpt_shape"] = [17, 3]
    yaml_path.write_text(yaml.safe_dump(config))
    result = runner.invoke(_build_app(), ["doctor", str(yaml_path), "--fast", "--json"])
    assert result.exit_code == 3
    payload = _parse_json_output(result.output)
    assert payload["error"] == "data_invalid"
