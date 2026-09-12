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
    assert cfg["amp_dtype"] == "float16"
    assert cfg["max_det"] == 300
    assert "eval_max_det" not in cfg
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
            "amp_dtype=bf16",
            "max_det=500",
            "eval_max_det=500",
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
    assert cfg["amp_dtype"] == "bfloat16"
    assert cfg["max_det"] == 500
    assert cfg["eval_max_det"] == 500


def test_train_dry_run_rejects_invalid_max_det():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreYOLO9t.pt",
            "max_det=0",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 2
    data = json.loads(result.stdout)
    assert data["error"] == "config_type_error"
    assert "max_det must be >= 1" in data["message"]


def test_train_dry_run_rejects_invalid_eval_max_det():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreYOLO9t.pt",
            "eval_max_det=0",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 2
    data = json.loads(result.stdout)
    assert data["error"] == "config_type_error"
    assert "eval_max_det must be >= 1" in data["message"]


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


@pytest.mark.parametrize("model", ["LibreYOLO9t.pt", "LibreRFDETRn.pt"])
def test_train_dry_run_single_cls_is_visible_for_g0_detectors(model):
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            f"model={model}",
            "single_cls=true",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["resolved_config"]["single_cls"] is True


@pytest.mark.parametrize(
    "args",
    [
        # Unsupported family (G2), detect task.
        ["model=LibreYOLOXn.pt", "single_cls=true"],
        # Supported family, unsupported task. Uses a published checkpoint plus
        # an explicit task so the PR gate never needs to fetch weights.
        ["model=LibreRFDETRm.pt", "task=segment", "single_cls=true"],
    ],
)
def test_train_dry_run_rejects_single_cls_outside_g0_g1_detection(args):
    app = _make_app()
    result = runner.invoke(
        app,
        ["data=coco8.yaml", *args, "--dry-run", "--json"],
    )

    assert result.exit_code == 2
    data = json.loads(result.stdout)
    assert data["error"] == "config_unsupported"
    assert "G0/G1 detection" in data["message"]


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


@pytest.mark.parametrize("model_name", ["LibreDFINEs.pt"])
def test_train_dry_run_rejects_class_balanced_on_custom_loaders(model_name):
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            f"model={model_name}",
            "class_balanced=true",
            "--dry-run",
            "--json",
        ],
    )
    assert result.exit_code == 2, result.stdout
    data = json.loads(result.stdout)
    assert data["error"] == "config_unsupported"
    assert "class_balanced" in data["message"]


def test_train_dry_run_accepts_class_balanced_for_rfdetr_detection(monkeypatch):
    monkeypatch.setattr(
        "libreyolo.cli.commands.train._create_explicit_task_train_model",
        lambda **_kwargs: None,
    )
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=rfdetr-n",
            "task=detect",
            "class_balanced=true",
            "--dry-run",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    config = json.loads(result.stdout)["resolved_config"]
    assert config["class_balanced"] is True


def test_train_dry_run_optin_helpers_both_grammars():
    """New #768 knobs parse in both CLI grammars and stay off by default."""
    app = _make_app()

    default = runner.invoke(
        app,
        ["data=coco8.yaml", "model=LibreYOLO9t.pt", "--dry-run", "--json"],
    )
    assert default.exit_code == 0, default.stdout
    default_cfg = json.loads(default.stdout)["resolved_config"]
    assert default_cfg["class_balanced"] is False
    assert default_cfg["average_best"] == 0
    assert default_cfg["export_check"] is False
    assert default_cfg["precise_bn"] == 0

    kv = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreYOLO9t.pt",
            "class_balanced=true",
            "average_best=5",
            "export_check=true",
            "precise_bn=128",
            "--dry-run",
            "--json",
        ],
    )
    assert kv.exit_code == 0, kv.stdout
    kv_cfg = json.loads(kv.stdout)["resolved_config"]
    assert kv_cfg["class_balanced"] is True
    assert kv_cfg["average_best"] == 5
    assert kv_cfg["export_check"] is True
    assert kv_cfg["precise_bn"] == 128

    dashed = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreYOLO9t.pt",
            "--class-balanced",
            "--average-best",
            "5",
            "--export-check",
            "--precise-bn",
            "128",
            "--dry-run",
            "--json",
        ],
    )
    assert dashed.exit_code == 0, dashed.stdout
    dashed_cfg = json.loads(dashed.stdout)["resolved_config"]
    assert dashed_cfg["class_balanced"] is True
    assert dashed_cfg["average_best"] == 5
    assert dashed_cfg["export_check"] is True
    assert dashed_cfg["precise_bn"] == 128


def test_train_dry_run_distill_model_is_visible():
    """A distillation teacher resolves into the config without error."""
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreYOLO9t.pt",
            "distill_model=LibreYOLO9m.pt",
            "dis=2.0",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["resolved_config"]["distill_model"] == "LibreYOLO9m.pt"
    assert data["resolved_config"]["dis"] == 2.0


def test_train_dry_run_accepts_lora_for_dfine_and_deim():
    app = _make_app()
    for model_name, family in (
        ("LibreDFINEs.pt", "dfine"),
        ("LibreDEIMs.pt", "deim"),
        ("LibreDEIMv2s.pt", "deimv2"),
        ("LibreRTDETRr18.pt", "rtdetr"),
        ("LibreRTDETRv2r18.pt", "rtdetrv2"),
        ("LibreRTDETRv4s.pt", "rtdetrv4"),
        ("LibreECs.pt", "ec"),
    ):
        result = runner.invoke(
            app,
            [
                "data=coco8.yaml",
                f"model={model_name}",
                "--lora",
                "--dry-run",
                "--json",
            ],
        )

        assert result.exit_code == 0, result.stdout
        data = json.loads(result.stdout)
        assert data["model_family"] == family
        assert data["resolved_config"]["lora"] is True


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
    assert kwargs["amp_dtype"] == "float16"
    assert kwargs["max_det"] == 300
    assert kwargs["eval_max_det"] is None
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


def test_train_rfdetr_single_cls_reaches_trainer(monkeypatch, tmp_path):
    app = _make_app()
    captured = {}

    class _RFDETRLike:
        FAMILY = "rfdetr"
        DEFAULT_TASK = "detect"
        device = "cpu"

        @classmethod
        def detect_task_from_filename(cls, filename):
            return None

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
            "single_cls=true",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["single_cls"] is True


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


def test_train_rfdetr_pose_uses_explicit_detect_transfer_flag(monkeypatch, tmp_path):
    app = _make_app()
    captured = {}

    class _RFDETRPoseLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def __init__(
            self,
            model_path=None,
            size=None,
            task=None,
            device="auto",
            allow_detect_to_obb_transfer=False,
            allow_detect_to_pose_transfer=False,
        ):
            captured["init"] = {
                "model_path": model_path,
                "size": size,
                "task": task,
                "device": device,
                "allow_detect_to_obb_transfer": allow_detect_to_obb_transfer,
                "allow_detect_to_pose_transfer": allow_detect_to_pose_transfer,
            }
            self.size = size
            self.task = task
            self.device = device

        @classmethod
        def detect_task_from_filename(cls, filename):
            return "pose" if filename.lower().endswith("-pose.pt") else None

        @classmethod
        def detect_size_from_filename(cls, filename):
            return "n" if "rfdetrn" in filename.lower() else None

        def train(self, data, **kwargs):
            captured["data"] = data
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_pose_exp")}

    def fail_load(*_args, **_kwargs):
        raise AssertionError(
            "RF-DETR pose training should instantiate the task architecture"
        )

    import libreyolo.models.rfdetr.model as rfdetr_model

    monkeypatch.setattr("libreyolo.cli.commands.train.load_model_or_exit", fail_load)
    monkeypatch.setattr(
        "libreyolo.cli.commands.train._model_ref_exists", lambda _: True
    )
    monkeypatch.setattr(rfdetr_model, "LibreRFDETR", _RFDETRPoseLike)

    result = runner.invoke(
        app,
        [
            "data=coco-pose.yaml",
            "model=LibreRFDETRn.pt",
            "task=pose",
            "epochs=1",
            "pretrained=true",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["init"] == {
        "model_path": "LibreRFDETRn.pt",
        "size": "n",
        "task": "pose",
        "device": "auto",
        "allow_detect_to_obb_transfer": False,
        "allow_detect_to_pose_transfer": True,
    }
    assert captured["data"] == "coco-pose.yaml"
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


@pytest.mark.parametrize(
    "option",
    [
        ["class_weights=true"],
        ["--class-weights"],
        ["class_weights=false"],
        ["--no-class-weights"],
    ],
)
@pytest.mark.parametrize("model", ["LibreResNet18-cls.pt", "LibreDINOv2s-cls.pt"])
def test_class_weights_cli_grammars(option, model):
    result = runner.invoke(
        _make_app(),
        [f"model={model}", "data=unused", *option, "--dry-run", "--json", "--quiet"],
    )
    assert result.exit_code == 0, result.output
    cfg = json.loads(result.stdout)["resolved_config"]
    assert cfg["class_weights"] is (
        option[0] in ("class_weights=true", "--class-weights")
    )


def test_class_weights_help_json():
    result = runner.invoke(_make_app(), ["--help-json"])
    assert result.exit_code == 0, result.output
    assert "class_weights" in result.stdout or "class-weights" in result.stdout


@pytest.mark.parametrize(
    "option,value",
    [([], 0.0), (["cls_pw=0.5"], 0.5), (["--cls-pw", "0.5"], 0.5), (["cls_pw=1"], 1.0)],
)
@pytest.mark.parametrize("model", ["LibreResNet18-cls.pt", "LibreDINOv2s-cls.pt"])
def test_cls_pw_cli_grammars_and_default(option, value, model):
    result = runner.invoke(
        _make_app(),
        [f"model={model}", "data=unused", *option, "--dry-run", "--json", "--quiet"],
    )
    assert result.exit_code == 0, result.output
    cfg = json.loads(result.stdout)["resolved_config"]
    assert cfg["cls_pw"] == value
    assert cfg["class_weights"] is False


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf", "-inf", "true"])
def test_cls_pw_cli_rejects_invalid_values(value):
    result = runner.invoke(
        _make_app(),
        [
            "model=LibreResNet18-cls.pt",
            "data=unused",
            f"cls_pw={value}",
            "--dry-run",
            "--json",
        ],
    )
    assert result.exit_code != 0
    assert "cls_pw" in result.output or "cls-pw" in result.output
    assert "Traceback" not in result.output


def test_cls_pw_cli_rejects_enabled_legacy_conflict():
    result = runner.invoke(
        _make_app(),
        [
            "model=LibreResNet18-cls.pt",
            "data=unused",
            "cls_pw=0.5",
            "class_weights=true",
            "--dry-run",
            "--json",
        ],
    )
    assert result.exit_code != 0
    assert "not both" in result.output


@pytest.mark.parametrize(
    "family,model",
    [("resnet", "LibreResNet18-cls.pt"), ("dinov2", "LibreDINOv2s-cls.pt")],
)
@pytest.mark.parametrize("option", [["cls_pw=0.5"], ["--cls-pw", "0.5"]])
def test_cls_pw_cli_forwards_numeric_strength(
    monkeypatch, tmp_path, family, model, option
):
    captured = {}

    class Classifier:
        FAMILY = family
        device = "cpu"
        task = "classify"

        def train(self, data, **kwargs):
            captured.update(kwargs)
            return {"output_dir": str(tmp_path)}

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda *args, **kwargs: Classifier(),
    )
    result = runner.invoke(
        _make_app(), [f"model={model}", "data=unused", *option, "--json", "--quiet"]
    )
    assert result.exit_code == 0, result.output
    assert captured["cls_pw"] == 0.5
    assert captured["class_weights"] is False


def test_cls_pw_help_json_exposes_numeric_default():
    result = runner.invoke(_make_app(), ["--help-json"])
    assert result.exit_code == 0, result.output
    schema = json.loads(result.stdout)
    assert "cls_pw" in result.stdout

    parameter = next(p for p in schema["parameters"] if p["name"] == "cls_pw")
    assert parameter["default"] == 0.0


@pytest.mark.parametrize(
    "family,model",
    [
        ("yolo9", "LibreYOLO9s.pt"),
        ("rfdetr", "LibreRFDETRs.pt"),
        ("resnet", "LibreResNet18-cls.pt"),
    ],
)
@pytest.mark.parametrize("option", [["best_metric=f1"], ["--best-metric", "f1"]])
def test_best_metric_cli_forwards_alias(monkeypatch, tmp_path, family, model, option):
    captured = {}

    class Stub:
        FAMILY = family
        device = "cpu"
        task = "classify" if family == "resnet" else "detect"

        def train(self, data, **kwargs):
            captured.update(kwargs)
            return {"output_dir": str(tmp_path)}

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda *args, **kwargs: Stub(),
    )
    result = runner.invoke(
        _make_app(),
        [
            f"model={model}",
            "data=unused",
            *option,
            f"project={tmp_path}",
            "--json",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["best_metric"] == "f1"


def test_best_metric_cli_default_is_none(monkeypatch, tmp_path):
    captured = {}

    class Stub:
        FAMILY = "yolo9"
        device = "cpu"
        task = "detect"

        def train(self, data, **kwargs):
            captured.update(kwargs)
            return {"output_dir": str(tmp_path)}

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda *args, **kwargs: Stub(),
    )
    result = runner.invoke(
        _make_app(),
        [
            "model=LibreYOLO9s.pt",
            "data=unused",
            f"project={tmp_path}",
            "--json",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("best_metric") is None


def test_best_metric_help_json_exposes_option():
    result = runner.invoke(_make_app(), ["--help-json"])
    assert result.exit_code == 0, result.output
    schema = json.loads(result.stdout)
    parameter = next(p for p in schema["parameters"] if p["name"] == "best_metric")
    assert parameter.get("default") is None


def test_train_json_output_reports_best_metric_key(monkeypatch, tmp_path):
    """The JSON payload must say which metric key drove best.pt selection."""

    class Stub:
        FAMILY = "resnet"
        device = "cpu"
        task = "classify"

        def train(self, data, **kwargs):
            return {
                "output_dir": str(tmp_path),
                "best_mAP50": 0.1,
                "best_mAP50_95": 0.2,
                "best_metric_key": "metrics/best_conf_f1",
            }

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda *args, **kwargs: Stub(),
    )
    result = runner.invoke(
        _make_app(),
        [
            "model=LibreResNet18-cls.pt",
            "data=unused",
            f"project={tmp_path}",
            "--json",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["best_metrics"]["best_metric_key"] == "metrics/best_conf_f1"
