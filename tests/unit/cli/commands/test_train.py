"""Behavior tests for the train command.

These verify observable CLI behavior (dry-run config resolution).
Real training is covered in e2e/test_rf1_training.py.
"""

import json

import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands.train import train_cmd
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit

runner = CliRunner()


def _make_app() -> typer.Typer:
    app = typer.Typer()
    app.command("train", cls=KeyValueCommand)(train_cmd)
    return app


def test_train_dry_run_uses_rtdetr_defaults():
    """Dry-run shows correct family-specific defaults for RT-DETR."""
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=rtdetr-r18",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "rtdetr"
    assert data["resolved_config"]["epochs"] == 72
    assert data["resolved_config"]["batch"] == 4
    assert data["resolved_config"]["optimizer"] == "adamw"
    assert data["resolved_config"]["lr0"] == 0.0001
    assert data["resolved_config"]["scheduler"] == "constant"


def test_train_dry_run_uses_rtdetr_defaults_for_weight_filename():
    """Dry-run detects family defaults from supported weight filenames."""
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreRTDETRr18.pt",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "rtdetr"
    assert data["resolved_config"]["epochs"] == 72
    assert data["resolved_config"]["batch"] == 4
    assert data["resolved_config"]["optimizer"] == "adamw"
    assert data["resolved_config"]["lr0"] == 0.0001
    assert data["resolved_config"]["scheduler"] == "constant"


def test_train_dry_run_uses_rfdetr_defaults():
    """Dry-run shows native RF-DETR defaults instead of generic YOLO defaults."""
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=rfdetr-m",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    cfg = data["resolved_config"]
    assert cfg["epochs"] == 100
    assert cfg["batch"] == 4
    assert cfg["lr0"] == 0.0001
    assert cfg["workers"] == 0
    assert cfg["weight_decay"] == 0.0001
    assert cfg["eval_interval"] == 1
    assert cfg["warmup_epochs"] == 0
    assert cfg["lr_drop"] == 100
    assert cfg["ema_decay"] == 0.993
    from libreyolo.models.rfdetr.config import RFDETRConfig

    assert RFDETRConfig().ema_tau == 100
    assert "optimizer" not in cfg
    assert "scheduler" not in cfg


def test_train_dry_run_rfdetr_user_override_wins():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreRFDETRm.pt",
            "epochs=3",
            "batch=2",
            "lr0=0.001",
            "lr_drop=7",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    cfg = data["resolved_config"]
    assert cfg["epochs"] == 3
    assert cfg["batch"] == 2
    assert cfg["lr0"] == 0.001
    assert cfg["lr_drop"] == 7


def test_train_dry_run_rfdetr_lora_flag_is_visible():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreRFDETRm.pt",
            "--lora",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    assert data["resolved_config"]["lora"] is True


def test_train_dry_run_rfdetr_freeze_flag_is_visible():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreRFDETRm.pt",
            "--freeze",
            "backbone",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    assert data["resolved_config"]["freeze"] == "backbone"


def test_train_dry_run_rejects_ambiguous_freeze_true():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreYOLO9t.pt",
            "--freeze",
            "true",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 2
    data = json.loads(result.stdout)
    assert data["error"] == "config_type_error"
    assert "freeze=True is ambiguous" in data["message"]


def test_train_dry_run_rejects_lora_for_unsupported_family():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreYOLO9t.pt",
            "--lora",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 2
    data = json.loads(result.stdout)
    assert data["error"] == "config_unsupported"
    assert "not supported for yolo9" in data["message"]


def test_train_rfdetr_actual_call_uses_reported_defaults(monkeypatch, tmp_path):
    """RF-DETR train should receive the same defaults shown by dry-run."""
    app = _make_app()
    captured = {}

    class _RFDETRLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def train(self, data, **kwargs):
            captured["data"] = data
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_exp")}

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda out, model, model_path, device: _RFDETRLike(),
    )

    result = runner.invoke(
        app,
        [
            "data=dummy.yaml",
            "model=LibreRFDETRm.pt",
            f"project={tmp_path}",
            "exist_ok=true",
            "save_plots=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["data"] == "dummy.yaml"
    kwargs = captured["kwargs"]
    assert kwargs["epochs"] == 100
    assert kwargs["batch"] == 4
    assert kwargs["lr0"] == 0.0001
    assert kwargs["num_workers"] == 0
    assert kwargs["weight_decay"] == 0.0001
    assert kwargs["eval_interval"] == 1
    assert kwargs["warmup_epochs"] == 0
    assert kwargs["scheduler"] == "step"
    assert kwargs["lr_drop"] == 100
    assert kwargs["use_ema"] is True
    assert kwargs["ema_decay"] == 0.993
    assert kwargs["save_plots"] is True
    assert kwargs["early_stopping"] is False

    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    assert data["epochs_completed"] == 100


def test_train_rfdetr_scheduler_override_reaches_trainer(monkeypatch, tmp_path):
    app = _make_app()
    captured = {}

    class _RFDETRLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def train(self, data, **kwargs):
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_exp")}

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda out, model, model_path, device: _RFDETRLike(),
    )

    result = runner.invoke(
        app,
        [
            "data=dummy.yaml",
            "model=LibreRFDETRm.pt",
            "scheduler=cosine",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["scheduler"] == "cosine"
    assert "ignores these parameters" not in result.output


def test_train_rfdetr_lora_flag_reaches_trainer(monkeypatch, tmp_path):
    app = _make_app()
    captured = {}

    class _RFDETRLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def train(self, data, **kwargs):
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_exp")}

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda out, model, model_path, device: _RFDETRLike(),
    )

    result = runner.invoke(
        app,
        [
            "data=dummy.yaml",
            "model=LibreRFDETRm.pt",
            "--lora",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["lora"] is True
    assert "ignores these parameters" not in result.output


def test_train_rfdetr_lr_drop_override_reaches_trainer(monkeypatch, tmp_path):
    app = _make_app()
    captured = {}

    class _RFDETRLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def train(self, data, **kwargs):
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_exp")}

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda out, model, model_path, device: _RFDETRLike(),
    )

    result = runner.invoke(
        app,
        [
            "data=dummy.yaml",
            "model=LibreRFDETRm.pt",
            "lr_drop=12",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["lr_drop"] == 12
    assert "ignores these parameters" not in result.output


def test_train_dry_run_reports_explicit_obb_task():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=uav-obb.yaml",
            "model=yolo9-t",
            "task=obb",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "yolo9"
    assert data["resolved_config"]["task"] == "obb"
    assert data["resolved_config"]["scheduler"] == "linear"


@pytest.mark.parametrize(
    "model_args",
    [
        ["model=yolo9-t", "task=obb"],
        ["model=yolo9-t-obb"],
    ],
)
def test_train_obb_uses_task_architecture_without_loading_missing_obb_weights(
    monkeypatch, tmp_path, model_args
):
    app = _make_app()
    captured = {}

    def fail_load(*_args, **_kwargs):
        raise AssertionError("OBB training should instantiate the task architecture")

    def fake_train(self, data, **kwargs):
        captured["task"] = self.task
        captured["size"] = self.size
        captured["data"] = data
        captured["kwargs"] = kwargs
        return {"output_dir": str(tmp_path / "yolo9_obb_exp")}

    monkeypatch.setattr("libreyolo.cli.commands.train.load_model_or_exit", fail_load)
    monkeypatch.setattr("libreyolo.models.yolo9.model.LibreYOLO9.train", fake_train)

    result = runner.invoke(
        app,
        [
            "data=uav-obb.yaml",
            *model_args,
            "epochs=1",
            "pretrained=false",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["task"] == "obb"
    assert captured["size"] == "t"
    assert captured["data"] == "uav-obb.yaml"
    assert captured["kwargs"]["pretrained"] is False
    data = json.loads(result.stdout)
    assert data["model_family"] == "yolo9"
    assert data["epochs_completed"] == 1


def test_train_yolo9_segment_task_uses_segment_architecture(monkeypatch, tmp_path):
    app = _make_app()
    captured = {}

    def fail_load(*_args, **_kwargs):
        raise AssertionError(
            "Segment training should instantiate the task architecture"
        )

    def fake_train(self, data, **kwargs):
        captured["task"] = self.task
        captured["size"] = self.size
        captured["data"] = data
        captured["kwargs"] = kwargs
        return {"output_dir": str(tmp_path / "yolo9_seg_exp")}

    monkeypatch.setattr("libreyolo.cli.commands.train.load_model_or_exit", fail_load)
    monkeypatch.setattr("libreyolo.models.yolo9.model.LibreYOLO9.train", fake_train)

    result = runner.invoke(
        app,
        [
            "data=coco8-seg.yaml",
            "model=yolo9-t",
            "task=segment",
            "epochs=1",
            "pretrained=false",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["task"] == "segment"
    assert captured["size"] == "t"
    assert captured["data"] == "coco8-seg.yaml"
    assert captured["kwargs"]["pretrained"] is False
    data = json.loads(result.stdout)
    assert data["model_family"] == "yolo9"
    assert data["epochs_completed"] == 1


@pytest.mark.parametrize(
    "model_args",
    [
        ["model=yolo9-t", "task=classify"],
        ["model=yolo9-t-cls"],
    ],
)
def test_train_yolo9_classify_uses_task_architecture_without_loading_missing_weights(
    monkeypatch, tmp_path, model_args
):
    app = _make_app()
    captured = {}

    def fail_load(*_args, **_kwargs):
        raise AssertionError(
            "Classification training should instantiate the task architecture"
        )

    def fake_train(self, data, **kwargs):
        captured["task"] = self.task
        captured["size"] = self.size
        captured["data"] = data
        captured["kwargs"] = kwargs
        return {"output_dir": str(tmp_path / "yolo9_cls_exp")}

    monkeypatch.setattr("libreyolo.cli.commands.train.load_model_or_exit", fail_load)
    monkeypatch.setattr("libreyolo.models.yolo9.model.LibreYOLO9.train", fake_train)

    result = runner.invoke(
        app,
        [
            "data=imagenet10",
            *model_args,
            "epochs=1",
            "pretrained=false",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["task"] == "classify"
    assert captured["size"] == "t"
    assert captured["data"] == "imagenet10"
    assert captured["kwargs"]["pretrained"] is False
    data = json.loads(result.stdout)
    assert data["model_family"] == "yolo9"
    assert data["epochs_completed"] == 1


def test_train_rfdetr_obb_uses_task_architecture_without_generic_load(
    monkeypatch, tmp_path
):
    app = _make_app()
    captured = {}

    class _RFDETROBBLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def __init__(
            self,
            model_path=None,
            size=None,
            task=None,
            device="auto",
            allow_detect_to_obb_transfer=False,
        ):
            captured["init"] = {
                "model_path": model_path,
                "size": size,
                "task": task,
                "device": device,
                "allow_detect_to_obb_transfer": allow_detect_to_obb_transfer,
            }
            self.size = size
            self.task = task
            self.device = device

        @classmethod
        def detect_task_from_filename(cls, filename):
            return "obb" if filename.lower().endswith("-obb.pt") else None

        @classmethod
        def detect_size_from_filename(cls, filename):
            return "n" if "rfdetrn" in filename.lower() else None

        def train(self, data, **kwargs):
            captured["data"] = data
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_obb_exp")}

    def fail_load(*_args, **_kwargs):
        raise AssertionError(
            "RF-DETR OBB training should instantiate the task architecture"
        )

    import libreyolo.models.rfdetr.model as rfdetr_model

    monkeypatch.setattr("libreyolo.cli.commands.train.load_model_or_exit", fail_load)
    monkeypatch.setattr(
        "libreyolo.cli.commands.train._model_ref_exists", lambda _: False
    )
    monkeypatch.setattr(rfdetr_model, "LibreRFDETR", _RFDETROBBLike)

    result = runner.invoke(
        app,
        [
            "data=uav-obb.yaml",
            "model=LibreRFDETRn.pt",
            "task=obb",
            "epochs=1",
            "pretrained=true",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["init"] == {
        "model_path": None,
        "size": "n",
        "task": "obb",
        "device": "auto",
        "allow_detect_to_obb_transfer": True,
    }
    assert captured["data"] == "uav-obb.yaml"
    assert "pretrained" not in captured["kwargs"]
    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    assert data["epochs_completed"] == 1


@pytest.mark.parametrize(
    "model_args",
    [
        ["model=LibreRFDETRn.pt", "task=classify"],
        ["model=rfdetr-n-cls"],
    ],
)
def test_train_rfdetr_classify_uses_task_architecture_without_generic_load(
    monkeypatch, tmp_path, model_args
):
    app = _make_app()
    captured = {}

    class _RFDETRClassifyLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def __init__(
            self,
            model_path=None,
            size=None,
            task=None,
            device="auto",
            allow_detect_to_obb_transfer=False,
        ):
            captured["init"] = {
                "model_path": model_path,
                "size": size,
                "task": task,
                "device": device,
                "allow_detect_to_obb_transfer": allow_detect_to_obb_transfer,
            }
            self.size = size
            self.task = task
            self.device = device

        @classmethod
        def detect_task_from_filename(cls, filename):
            return "classify" if filename.lower().endswith("-cls.pt") else None

        @classmethod
        def detect_size_from_filename(cls, filename):
            return "n" if "rfdetrn" in filename.lower() else None

        def train(self, data, **kwargs):
            captured["data"] = data
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_cls_exp")}

    def fail_load(*_args, **_kwargs):
        raise AssertionError(
            "RF-DETR classification training should instantiate the task architecture"
        )

    import libreyolo.models.rfdetr.model as rfdetr_model

    monkeypatch.setattr("libreyolo.cli.commands.train.load_model_or_exit", fail_load)
    monkeypatch.setattr(
        "libreyolo.cli.commands.train._model_ref_exists", lambda _: False
    )
    monkeypatch.setattr(rfdetr_model, "LibreRFDETR", _RFDETRClassifyLike)

    result = runner.invoke(
        app,
        [
            "data=imagenet10",
            *model_args,
            "epochs=1",
            "pretrained=true",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["init"] == {
        "model_path": None,
        "size": "n",
        "task": "classify",
        "device": "auto",
        "allow_detect_to_obb_transfer": False,
    }
    assert captured["data"] == "imagenet10"
    assert "pretrained" not in captured["kwargs"]
    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    assert data["epochs_completed"] == 1


def test_train_rfdetr_detect_checkpoint_switches_to_obb_architecture(
    monkeypatch, tmp_path
):
    app = _make_app()
    detect_path = tmp_path / "custom-rfdetr.pt"
    detect_path.write_bytes(b"placeholder")
    captured = {}

    class _LoadedRFDETRDetect:
        FAMILY = "rfdetr"
        task = "detect"
        size = "n"
        device = "cpu"

    class _RFDETROBBLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def __init__(
            self,
            model_path=None,
            size=None,
            task=None,
            device="auto",
            allow_detect_to_obb_transfer=False,
        ):
            captured["init"] = {
                "model_path": model_path,
                "size": size,
                "task": task,
                "device": device,
                "allow_detect_to_obb_transfer": allow_detect_to_obb_transfer,
            }
            self.size = size
            self.task = task
            self.device = device

        def train(self, data, **kwargs):
            captured["data"] = data
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_obb_custom_transfer")}

    import libreyolo.models.rfdetr.model as rfdetr_model

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda out, model, model_path, device: _LoadedRFDETRDetect(),
    )
    monkeypatch.setattr(rfdetr_model, "LibreRFDETR", _RFDETROBBLike)

    result = runner.invoke(
        app,
        [
            "data=uav-obb.yaml",
            f"model={detect_path}",
            "task=obb",
            "epochs=1",
            "pretrained=true",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["init"] == {
        "model_path": str(detect_path),
        "size": "n",
        "task": "obb",
        "device": "auto",
        "allow_detect_to_obb_transfer": True,
    }
    assert captured["data"] == "uav-obb.yaml"
    assert "pretrained" not in captured["kwargs"]
    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    assert data["epochs_completed"] == 1


def test_train_obb_custom_detect_checkpoint_uses_checkpoint_as_transfer(
    monkeypatch, tmp_path
):
    app = _make_app()
    detect_path = tmp_path / "best.pt"
    detect_path.write_bytes(b"placeholder")
    captured = {}

    class _DetectYOLO9:
        FAMILY = "yolo9"
        task = "detect"
        size = "t"

    def fake_train(self, data, **kwargs):
        captured["task"] = self.task
        captured["size"] = self.size
        captured["data"] = data
        captured["kwargs"] = kwargs
        return {"output_dir": str(tmp_path / "yolo9_obb_custom_transfer")}

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda out, model, model_path, device: _DetectYOLO9(),
    )
    monkeypatch.setattr("libreyolo.models.yolo9.model.LibreYOLO9.train", fake_train)

    result = runner.invoke(
        app,
        [
            "data=uav-obb.yaml",
            f"model={detect_path}",
            "task=obb",
            "epochs=1",
            "pretrained=true",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["task"] == "obb"
    assert captured["size"] == "t"
    assert captured["data"] == "uav-obb.yaml"
    assert captured["kwargs"]["pretrained"] == str(detect_path)
    data = json.loads(result.stdout)
    assert data["model_family"] == "yolo9"
    assert data["epochs_completed"] == 1


def test_train_obb_known_detect_checkpoint_path_is_transfer_without_direct_load(
    monkeypatch, tmp_path
):
    app = _make_app()
    detect_path = tmp_path / "LibreYOLO9t.pt"
    detect_path.write_bytes(b"placeholder")
    captured = {}

    def fail_load(*_args, **_kwargs):
        raise AssertionError(
            "Detect checkpoint should be transfer weights, not direct OBB load"
        )

    def fake_train(self, data, **kwargs):
        captured["task"] = self.task
        captured["size"] = self.size
        captured["data"] = data
        captured["kwargs"] = kwargs
        return {"output_dir": str(tmp_path / "yolo9_obb_known_transfer")}

    monkeypatch.setattr("libreyolo.cli.commands.train.load_model_or_exit", fail_load)
    monkeypatch.setattr("libreyolo.models.yolo9.model.LibreYOLO9.train", fake_train)

    result = runner.invoke(
        app,
        [
            "data=uav-obb.yaml",
            f"model={detect_path}",
            "task=obb",
            "epochs=1",
            "pretrained=true",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["task"] == "obb"
    assert captured["size"] == "t"
    assert captured["data"] == "uav-obb.yaml"
    assert captured["kwargs"]["pretrained"] == str(detect_path)
    data = json.loads(result.stdout)
    assert data["model_family"] == "yolo9"
    assert data["epochs_completed"] == 1
