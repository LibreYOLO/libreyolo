"""Unit tests for CoreML export. Mocks coremltools so it runs on every platform."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Install a fake `coremltools` module so the import inside libreyolo.export.coreml
# succeeds even on machines without coremltools installed.
_fake_ct = MagicMock()
_fake_ct.ComputeUnit.ALL = "ALL"
_fake_ct.ComputeUnit.CPU_AND_GPU = "CPU_AND_GPU"
_fake_ct.ComputeUnit.CPU_AND_NE = "CPU_AND_NE"
_fake_ct.ComputeUnit.CPU_ONLY = "CPU_ONLY"
_fake_ct.precision.FLOAT32 = "FLOAT32"
_fake_ct.precision.FLOAT16 = "FLOAT16"
_fake_ct.target.iOS15 = "iOS15"
sys.modules.setdefault("coremltools", _fake_ct)

from libreyolo.export.coreml import _to_compute_unit  # noqa: E402


pytestmark = pytest.mark.unit


class TestToComputeUnit:
    def test_all(self):
        assert _to_compute_unit("all") == "ALL"

    def test_cpu_and_gpu(self):
        assert _to_compute_unit("cpu_and_gpu") == "CPU_AND_GPU"

    def test_cpu_and_ne(self):
        assert _to_compute_unit("cpu_and_ne") == "CPU_AND_NE"

    def test_cpu_only(self):
        assert _to_compute_unit("cpu_only") == "CPU_ONLY"

    def test_case_insensitive(self):
        assert _to_compute_unit("ALL") == "ALL"
        assert _to_compute_unit("Cpu_And_Ne") == "CPU_AND_NE"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="compute_units"):
            _to_compute_unit("tpu")


class _DummyModel(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3))


def _patch_ct(monkeypatch):
    """Reset the fake coremltools module and return the mock for assertions."""
    # Create the main coremltools mock
    fake = MagicMock()
    fake.ComputeUnit.ALL = "ALL"
    fake.ComputeUnit.CPU_AND_GPU = "CPU_AND_GPU"
    fake.ComputeUnit.CPU_AND_NE = "CPU_AND_NE"
    fake.ComputeUnit.CPU_ONLY = "CPU_ONLY"
    fake.precision.FLOAT32 = "FLOAT32"
    fake.precision.FLOAT16 = "FLOAT16"
    fake.target.iOS15 = "iOS15"
    fake.ImageType = MagicMock(side_effect=lambda **kw: ("ImageType", kw))
    
    # Create models submodule mock
    fake_models = MagicMock()
    fake_models.pipeline = MagicMock()
    fake.models = fake_models
    # MLModel is in the models submodule
    fake.models.MLModel = MagicMock()
    
    # Create the MLModel mock that gets returned by convert
    mlmodel = MagicMock()
    mlmodel.user_defined_metadata = {}
    fake.convert = MagicMock(return_value=mlmodel)
    
    # Patch the module and submodules
    monkeypatch.setitem(sys.modules, "coremltools", fake)
    monkeypatch.setitem(sys.modules, "coremltools.models", fake_models)
    monkeypatch.setitem(sys.modules, "coremltools.models.pipeline", fake_models.pipeline)
    
    return fake, mlmodel


class TestExportCoreML:
    def test_fp32_basic_call(self, tmp_path, monkeypatch):
        fake, mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        nn_model = _DummyModel().eval()
        dummy = torch.randn(1, 3, 640, 640)
        out = tmp_path / "model.mlpackage"

        result = export_coreml(
            nn_model, dummy,
            output_path=str(out),
            precision="fp32",
            compute_units="all",
            nms=False,
            metadata={"libreyolo_version": "0.0.1", "model_family": "yolox",
                      "names": {"0": "person"}, "imgsz": 640},
            model_family="yolox",
        )

        assert result == str(out)
        # ct.convert called with mlprogram + FLOAT32 + iOS15 + ImageType input
        kwargs = fake.convert.call_args.kwargs
        assert kwargs["convert_to"] == "mlprogram"
        assert kwargs["compute_precision"] == "FLOAT32"
        assert kwargs["minimum_deployment_target"] == "iOS15"
        # ImageType called with scale=1/255 and image input name 'image'
        img_kwargs = fake.ImageType.call_args.kwargs
        assert img_kwargs["name"] == "image"
        assert img_kwargs["scale"] == pytest.approx(1.0 / 255.0)
        assert img_kwargs["bias"] == [0.0, 0.0, 0.0]
        # Compute unit set
        assert mlmodel.compute_unit == "ALL"
        # Metadata was stringified and stored
        assert all(isinstance(v, str) for v in mlmodel.user_defined_metadata.values())
        assert mlmodel.user_defined_metadata["model_family"] == "yolox"
        # Save called
        mlmodel.save.assert_called_once_with(str(out))

    def test_fp16_uses_float16_precision(self, tmp_path, monkeypatch):
        fake, mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        export_coreml(
            _DummyModel().eval(),
            torch.randn(1, 3, 640, 640),
            output_path=str(tmp_path / "m.mlpackage"),
            precision="fp16",
            compute_units="cpu_and_ne",
            nms=False,
            metadata={"model_family": "yolox"},
            model_family="yolox",
        )
        assert fake.convert.call_args.kwargs["compute_precision"] == "FLOAT16"
        assert mlmodel.compute_unit == "CPU_AND_NE"

    def test_metadata_names_json_encoded(self, tmp_path, monkeypatch):
        import json
        fake, mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        export_coreml(
            _DummyModel().eval(),
            torch.randn(1, 3, 640, 640),
            output_path=str(tmp_path / "m.mlpackage"),
            precision="fp32",
            compute_units="all",
            nms=False,
            metadata={"names": {"0": "person", "1": "cat"}, "imgsz": 640},
            model_family="yolox",
        )
        decoded = json.loads(mlmodel.user_defined_metadata["names"])
        assert decoded == {"0": "person", "1": "cat"}


class TestNMSWrap:
    def test_rfdetr_raises(self, tmp_path, monkeypatch):
        fake, mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        with pytest.raises(NotImplementedError, match="RF-DETR"):
            export_coreml(
                _DummyModel().eval(),
                torch.randn(1, 3, 640, 640),
                output_path=str(tmp_path / "m.mlpackage"),
                precision="fp32",
                compute_units="all",
                nms=True,
                metadata={"model_family": "rfdetr"},
                model_family="rfdetr",
            )

    def test_yolox_calls_pipeline(self, tmp_path, monkeypatch):
        fake, mlmodel = _patch_ct(monkeypatch)
        # ct.models.pipeline.Pipeline returns a mock pipeline whose .spec
        # is what gets saved.
        pipeline_mock = MagicMock()
        pipeline_mock.spec.user_defined_metadata = {}
        fake.models.pipeline.Pipeline = MagicMock(return_value=pipeline_mock)
        fake.models.MLModel = MagicMock(return_value=MagicMock(
            user_defined_metadata={}, compute_unit=None,
        ))

        from libreyolo.export.coreml import export_coreml

        export_coreml(
            _DummyModel().eval(),
            torch.randn(1, 3, 640, 640),
            output_path=str(tmp_path / "m.mlpackage"),
            precision="fp32",
            compute_units="all",
            nms=True,
            metadata={"model_family": "yolox", "nb_classes": 80},
            model_family="yolox",
        )
        # Pipeline was constructed
        assert fake.models.pipeline.Pipeline.called