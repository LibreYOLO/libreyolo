# CoreML Export + Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Apple Core ML as a supported export target — `.mlpackage` writer plus a macOS Python inference backend — so users can deploy LibreYOLO models (YOLOv9, YOLOX, RF-DETR) to Apple platforms.

**Architecture:** Direct PyTorch (`torch.jit.trace`) → `coremltools.convert` to ML Program (`.mlpackage`). New `CoreMLExporter` slots into the existing `BaseExporter` ABC + auto-registry pattern in `libreyolo/export/exporter.py`. New `CoreMLBackend` mirrors `OnnxBackend` and is dispatched by `.mlpackage` suffix in `libreyolo/models/__init__.py`. fp32 / fp16 only; opt-in embedded NMS via `nms=True` (raises on RF-DETR).

**Tech Stack:** `coremltools>=7.2`, `torch`, `numpy`, `Pillow`. Tests use `pytest` + `unittest.mock`. macOS-only e2e via `@pytest.mark.coreml`.

**Spec:** `docs/superpowers/specs/2026-04-26-coreml-export-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `libreyolo/export/coreml.py` | CREATE | `export_coreml()` core converter; `_to_compute_unit()` mapping; NMS pipeline wrapper |
| `libreyolo/export/exporter.py` | MODIFY | Append `CoreMLExporter` subclass (auto-registers) |
| `libreyolo/backends/coreml.py` | CREATE | `CoreMLBackend` — macOS-only inference via `coremltools.models.MLModel` |
| `libreyolo/models/__init__.py` | MODIFY | `.mlpackage` directory → `CoreMLBackend` dispatch |
| `libreyolo/__init__.py` | MODIFY | Lazy import of `CoreMLBackend` + `__all__` entry |
| `pyproject.toml` | MODIFY | `coreml` extra, `all` rollup entry, `coreml` pytest marker |
| `tests/unit/test_export_coreml.py` | CREATE | Mocked-coremltools logic tests (cross-platform) |
| `tests/e2e/test_coreml_roundtrip.py` | CREATE | `@pytest.mark.coreml` e2e tests (macOS only) |

---

## Task 1: Add `coreml` extra and pytest marker to pyproject.toml

**Files:**
- Modify: `pyproject.toml:46-58, 109-117`

- [ ] **Step 1: Add the `coreml` extra**

In `pyproject.toml`, after the `ncnn = [...]` block (currently lines 46-49) insert:

```toml
coreml = [
    "coremltools>=7.2",
]
```

- [ ] **Step 2: Add coreml to the `all` rollup**

Modify the `all = [...]` block (currently lines 51-58) to include `"libreyolo[coreml]"`:

```toml
all = [
    "libreyolo[onnx]",
    "libreyolo[rfdetr]",
    "libreyolo[tensorrt]",
    "libreyolo[openvino]",
    "libreyolo[ncnn]",
    "libreyolo[rtdetr]",
    "libreyolo[coreml]",
]
```

- [ ] **Step 3: Add the `coreml` pytest marker**

In `[tool.pytest.ini_options]` markers list (currently lines 109-117), add after the `ncnn` marker:

```toml
    "coreml: tests requiring CoreML (macOS only)",
```

- [ ] **Step 4: Validate the toml parses**

Run: `python -c "import tomllib; tomllib.loads(open('pyproject.toml').read())"`
Expected: no output (parse succeeded).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build(coreml): add coremltools extra and pytest marker (#90)"
```

---

## Task 2: Implement `_to_compute_unit` helper with TDD

**Files:**
- Create: `libreyolo/export/coreml.py`
- Create: `tests/unit/test_export_coreml.py`

- [ ] **Step 1: Write the failing test for `_to_compute_unit`**

Create `tests/unit/test_export_coreml.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_export_coreml.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'libreyolo.export.coreml'`.

- [ ] **Step 3: Create `libreyolo/export/coreml.py` with the helper**

Create `libreyolo/export/coreml.py`:

```python
"""CoreML (Apple .mlpackage) export implementation.

Direct PyTorch -> coremltools.convert path. Produces ML Program format
(.mlpackage), targeting iOS 15 / macOS 12 minimum.
"""

from __future__ import annotations

from typing import Any


def _to_compute_unit(compute_units: str):
    """Map a string compute_units value to a coremltools.ComputeUnit enum.

    Accepted: 'all', 'cpu_and_gpu', 'cpu_and_ne', 'cpu_only' (case-insensitive).
    """
    import coremltools as ct

    key = compute_units.lower()
    mapping = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    if key not in mapping:
        raise ValueError(
            f"Invalid compute_units {compute_units!r}. "
            f"Must be one of: {sorted(mapping)}"
        )
    return mapping[key]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_export_coreml.py -v`
Expected: PASS — all six `TestToComputeUnit` tests green.

- [ ] **Step 5: Commit**

```bash
git add libreyolo/export/coreml.py tests/unit/test_export_coreml.py
git commit -m "feat(coreml): add _to_compute_unit helper with tests (#90)"
```

---

## Task 3: Implement `export_coreml` core conversion (no NMS)

**Files:**
- Modify: `libreyolo/export/coreml.py`
- Modify: `tests/unit/test_export_coreml.py`

- [ ] **Step 1: Write failing tests for `export_coreml`**

Append to `tests/unit/test_export_coreml.py`:

```python
class _DummyModel(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3))


def _patch_ct(monkeypatch):
    """Reset the fake coremltools module and return the mock for assertions."""
    fake = MagicMock()
    fake.ComputeUnit.ALL = "ALL"
    fake.ComputeUnit.CPU_AND_GPU = "CPU_AND_GPU"
    fake.ComputeUnit.CPU_AND_NE = "CPU_AND_NE"
    fake.ComputeUnit.CPU_ONLY = "CPU_ONLY"
    fake.precision.FLOAT32 = "FLOAT32"
    fake.precision.FLOAT16 = "FLOAT16"
    fake.target.iOS15 = "iOS15"
    fake.ImageType = MagicMock(side_effect=lambda **kw: ("ImageType", kw))
    mlmodel = MagicMock()
    mlmodel.user_defined_metadata = {}
    fake.convert = MagicMock(return_value=mlmodel)
    monkeypatch.setitem(sys.modules, "coremltools", fake)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_export_coreml.py::TestExportCoreML -v`
Expected: FAIL — `ImportError: cannot import name 'export_coreml'`.

- [ ] **Step 3: Implement `export_coreml`**

Append to `libreyolo/export/coreml.py`:

```python
import json


def _stringify_metadata(metadata: dict) -> dict:
    """Convert metadata values to strings (CoreML user_defined_metadata requires str).

    Dict-typed values (e.g. ``names``) are JSON-encoded so they round-trip cleanly.
    """
    out: dict[str, str] = {}
    for k, v in metadata.items():
        if isinstance(v, dict):
            out[str(k)] = json.dumps(v)
        else:
            out[str(k)] = str(v)
    return out


def export_coreml(
    nn_model,
    dummy,
    *,
    output_path: str,
    precision: str = "fp32",
    compute_units: str = "all",
    nms: bool = False,
    metadata: dict | None = None,
    model_family: str | None = None,
) -> str:
    """Export a PyTorch model to CoreML (.mlpackage / ML Program format).

    Args:
        nn_model: The PyTorch nn.Module to export. Must already be in eval/export mode.
        dummy: Dummy input tensor of shape (B, 3, H, W) for tracing.
        output_path: Destination .mlpackage path (a directory bundle).
        precision: 'fp32' or 'fp16'. Maps to ct.precision.FLOAT32/FLOAT16.
        compute_units: 'all' | 'cpu_and_gpu' | 'cpu_and_ne' | 'cpu_only'.
        nms: If True, embed Apple's NonMaximumSuppression as a CoreML pipeline.
            Not supported for RF-DETR (raises NotImplementedError).
        metadata: Dict of metadata to embed under user_defined_metadata.
        model_family: Model family string (e.g. 'yolox', 'yolo9', 'rfdetr') used
            to gate features (NMS pipeline shape).

    Returns:
        ``output_path`` on success.
    """
    import coremltools as ct
    import torch

    traced = torch.jit.trace(nn_model, dummy)

    image_input = ct.ImageType(
        name="image",
        shape=tuple(dummy.shape),
        scale=1.0 / 255.0,
        bias=[0.0, 0.0, 0.0],
    )
    compute_precision = (
        ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    )

    mlmodel = ct.convert(
        traced,
        inputs=[image_input],
        convert_to="mlprogram",
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.iOS15,
    )

    mlmodel.compute_unit = _to_compute_unit(compute_units)

    if nms:
        mlmodel = _wrap_with_nms(mlmodel, model_family=model_family)

    if metadata:
        mlmodel.user_defined_metadata.update(_stringify_metadata(metadata))

    mlmodel.save(output_path)
    return output_path


def _wrap_with_nms(mlmodel: Any, *, model_family: str | None) -> Any:
    """Stub — implemented in Task 4. Defined here so export_coreml type-checks."""
    raise NotImplementedError("NMS pipeline not yet implemented")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_export_coreml.py -v`
Expected: PASS — all `TestToComputeUnit` and `TestExportCoreML` tests green (10 tests).

- [ ] **Step 5: Commit**

```bash
git add libreyolo/export/coreml.py tests/unit/test_export_coreml.py
git commit -m "feat(coreml): implement core export_coreml conversion (#90)"
```

---

## Task 4: Implement embedded-NMS pipeline (`nms=True`)

**Files:**
- Modify: `libreyolo/export/coreml.py:_wrap_with_nms`
- Modify: `tests/unit/test_export_coreml.py`

CoreML's `NonMaximumSuppression` model takes two inputs (`confidence` shape `(N, C)`, `coordinates` shape `(N, 4)` normalized xywh) and produces post-NMS detections. This task wraps the converted detector + a NonMaximumSuppression spec into a `Pipeline`. RF-DETR raises because its head emits per-query proposals, not the (N, C) / (N, 4) layout.

- [ ] **Step 1: Write failing tests for NMS wrapping**

Append to `tests/unit/test_export_coreml.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_export_coreml.py::TestNMSWrap -v`
Expected: FAIL — first test raises generic NotImplementedError without 'RF-DETR'; second test fails because `_wrap_with_nms` raises NotImplementedError.

- [ ] **Step 3: Implement `_wrap_with_nms`**

Replace the `_wrap_with_nms` stub in `libreyolo/export/coreml.py`:

```python
def _wrap_with_nms(mlmodel: Any, *, model_family: str | None) -> Any:
    """Wrap a detector mlmodel in a Pipeline that embeds Apple's NMS layer.

    Output names: 'confidence' (N x nb_classes), 'coordinates' (N x 4 normalized xywh).
    """
    if model_family == "rfdetr":
        raise NotImplementedError(
            "nms=True is not supported for RF-DETR; export with nms=False and run "
            "NMS in your application."
        )

    import coremltools as ct
    from coremltools.models import pipeline as ct_pipeline

    # Build an NMS model spec. Defaults match LibreYOLO's runtime defaults.
    nms_spec = ct.proto.Model_pb2.Model()
    nms_spec.specificationVersion = 5
    nms = nms_spec.nonMaximumSuppression
    nms.iouThreshold = 0.45
    nms.confidenceThreshold = 0.25
    nms.confidenceInputFeatureName = "confidence"
    nms.coordinatesInputFeatureName = "coordinates"
    nms.confidenceOutputFeatureName = "confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"

    nms_model = ct.models.MLModel(nms_spec)

    pipeline = ct_pipeline.Pipeline(
        input_features=[("image", None)],
        output_features=[("confidence", None), ("coordinates", None)],
    )
    pipeline.add_model(mlmodel)
    pipeline.add_model(nms_model)
    # Carry compute_unit forward
    pipeline.spec.compute_unit = mlmodel.compute_unit
    return ct.models.MLModel(pipeline.spec)
```

Note: this is a *minimal* pipeline scaffold. Real-world NMS wrapping for YOLO-style outputs sometimes also requires renaming the detector's output features so they match `confidence`/`coordinates`; that polish is left as a follow-up if the e2e test surfaces a mismatch (Task 7).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_export_coreml.py -v`
Expected: PASS — all 12 tests green.

- [ ] **Step 5: Commit**

```bash
git add libreyolo/export/coreml.py tests/unit/test_export_coreml.py
git commit -m "feat(coreml): embed NMS pipeline for nms=True; raise on RF-DETR (#90)"
```

---

## Task 5: Register `CoreMLExporter` subclass

**Files:**
- Modify: `libreyolo/export/exporter.py` (append subclass after `NcnnExporter`)
- Modify: `tests/unit/test_export_coreml.py`

- [ ] **Step 1: Write failing test for registry**

Append to `tests/unit/test_export_coreml.py`:

```python
class TestCoreMLExporterRegistry:
    def test_format_registered(self):
        from libreyolo.export.exporter import BaseExporter, CoreMLExporter
        assert "coreml" in BaseExporter._registry
        assert BaseExporter._registry["coreml"] is CoreMLExporter

    def test_class_attrs(self):
        from libreyolo.export.exporter import CoreMLExporter
        assert CoreMLExporter.format_name == "coreml"
        assert CoreMLExporter.suffix == ".mlpackage"
        assert CoreMLExporter.requires_onnx is False
        assert CoreMLExporter.supports_int8 is False
        assert CoreMLExporter.apply_model_half is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_export_coreml.py::TestCoreMLExporterRegistry -v`
Expected: FAIL — `ImportError: cannot import name 'CoreMLExporter'`.

- [ ] **Step 3: Append `CoreMLExporter` to `libreyolo/export/exporter.py`**

Append at the end of `libreyolo/export/exporter.py` (after `NcnnExporter`, around line 553):

```python
class CoreMLExporter(BaseExporter):
    format_name = "coreml"
    suffix = ".mlpackage"
    requires_onnx = False
    supports_int8 = False
    apply_model_half = False  # ct.convert handles precision via compute_precision

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        precision,
        metadata,
        compute_units="all",
        nms=False,
        **kwargs,
    ):
        from .coreml import export_coreml

        return export_coreml(
            nn_model,
            dummy,
            output_path=output_path,
            precision=precision,
            compute_units=compute_units,
            nms=nms,
            metadata=metadata,
            model_family=self.model._get_model_name(),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_export_coreml.py -v`
Expected: PASS — all 14 tests green.

- [ ] **Step 5: Commit**

```bash
git add libreyolo/export/exporter.py tests/unit/test_export_coreml.py
git commit -m "feat(coreml): register CoreMLExporter in BaseExporter registry (#90)"
```

---

## Task 6: Implement `CoreMLBackend` (macOS inference)

**Files:**
- Create: `libreyolo/backends/coreml.py`
- Modify: `libreyolo/__init__.py`
- Modify: `libreyolo/models/__init__.py`
- Modify: `tests/unit/test_export_coreml.py`

- [ ] **Step 1: Write failing test for backend lazy import + dispatch**

Append to `tests/unit/test_export_coreml.py`:

```python
class TestCoreMLBackendModule:
    def test_backend_class_importable(self):
        # On non-macOS, importing the class itself must succeed (only
        # instantiation should refuse). Use the lazy import path.
        import libreyolo
        assert hasattr(libreyolo, "CoreMLBackend")
        cls = libreyolo.CoreMLBackend
        assert cls.__name__ == "CoreMLBackend"

    def test_dispatch_mlpackage(self, tmp_path, monkeypatch):
        # Create a fake .mlpackage directory and ensure the model factory
        # routes it to CoreMLBackend (we patch the class to a sentinel).
        pkg = tmp_path / "fake.mlpackage"
        pkg.mkdir()

        sentinel = MagicMock(name="CoreMLBackendSentinel")
        import libreyolo.backends.coreml as coreml_mod
        monkeypatch.setattr(coreml_mod, "CoreMLBackend", sentinel)

        from libreyolo.models import LibreYOLO
        LibreYOLO(str(pkg), nb_classes=80, device="cpu")
        sentinel.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_export_coreml.py::TestCoreMLBackendModule -v`
Expected: FAIL — `AttributeError: module 'libreyolo' has no attribute 'CoreMLBackend'`.

- [ ] **Step 3: Create `libreyolo/backends/coreml.py`**

Create `libreyolo/backends/coreml.py`:

```python
"""CoreML inference backend for LibreYOLO. macOS only.

Loads .mlpackage models produced by libreyolo.export.coreml and runs inference
via coremltools.models.MLModel. Mirrors OnnxBackend's public surface so the
rest of LibreYOLO (Results, drawing, etc.) sees the same interface.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..utils.general import COCO_CLASSES
from .base import BaseBackend

logger = logging.getLogger(__name__)


def _to_compute_unit(compute_units: str):
    """Same mapping as the exporter — duplicated to avoid pulling export deps in."""
    import coremltools as ct

    key = compute_units.lower()
    mapping = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    if key not in mapping:
        raise ValueError(
            f"Invalid compute_units {compute_units!r}. "
            f"Must be one of: {sorted(mapping)}"
        )
    return mapping[key]


class CoreMLBackend(BaseBackend):
    """CoreML inference backend (macOS only).

    Args:
        model_path: Path to a .mlpackage directory.
        nb_classes: Number of classes (default: 80, overridden by metadata if present).
        device: Ignored — CoreML routes via compute_units instead.
        compute_units: 'all' | 'cpu_and_gpu' | 'cpu_and_ne' | 'cpu_only'. Default 'all'.
    """

    def __init__(
        self,
        model_path: str,
        nb_classes: int = 80,
        device: str = "auto",
        compute_units: str = "all",
    ):
        if sys.platform != "darwin":
            raise RuntimeError(
                "CoreML inference requires macOS. "
                f"Current platform: {sys.platform}."
            )
        try:
            import coremltools as ct
        except ImportError as e:
            raise ImportError(
                "CoreML inference requires coremltools. "
                "Install with: pip install libreyolo[coreml]"
            ) from e

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"CoreML model not found: {model_path}")

        self.model = ct.models.MLModel(
            str(path), compute_units=_to_compute_unit(compute_units)
        )

        meta = dict(self.model.user_defined_metadata) if self.model.user_defined_metadata else {}
        model_family, names, imgsz, has_embedded_nms = self._parse_metadata(meta, nb_classes)

        self._has_embedded_nms = has_embedded_nms

        super().__init__(
            model_path=str(path),
            nb_classes=len(names) if names else nb_classes,
            device="coreml",
            imgsz=imgsz,
            model_family=model_family,
            names=names if names else self.build_names(nb_classes),
        )

    @staticmethod
    def _parse_metadata(meta: dict, default_nb_classes: int):
        model_family: Optional[str] = meta.get("model_family") or None
        names: Optional[dict] = None
        imgsz = 640
        has_embedded_nms = False

        if "names" in meta:
            try:
                raw = json.loads(meta["names"])
                names = {int(k): v for k, v in raw.items()}
            except (ValueError, TypeError) as e:
                logger.warning("Failed to parse names metadata: %s", e)

        if names is None and meta.get("nb_classes"):
            try:
                nc = int(meta["nb_classes"])
                names = (
                    {i: n for i, n in enumerate(COCO_CLASSES)}
                    if nc == 80
                    else {i: f"class_{i}" for i in range(nc)}
                )
            except ValueError:
                pass

        if "imgsz" in meta:
            try:
                imgsz = int(meta["imgsz"])
            except ValueError:
                pass

        # If the model has 'confidence'/'coordinates' outputs (post-NMS pipeline),
        # we should not run python-side NMS again.
        try:
            output_names = {o.name for o in meta.get("__output_descriptions__", [])}
        except Exception:
            output_names = set()
        has_embedded_nms = output_names == {"confidence", "coordinates"}

        return model_family, names, imgsz, has_embedded_nms

    def _run_inference(self, blob: np.ndarray) -> list:
        """Run CoreML inference on a (1, C, H, W) preprocessed float blob.

        The exported model expects a CoreML ImageType input (uint8 PIL image,
        normalization baked in). We undo the libreyolo preprocess normalization
        to reconstruct a uint8 PIL image, then feed it.
        """
        if blob.ndim != 4 or blob.shape[0] != 1:
            raise ValueError(
                f"CoreMLBackend expects (1, C, H, W) blob; got {blob.shape}"
            )
        # Reverse the (x/255) normalization: blob is float in [0, 1].
        # CoreML model has scale=1/255 baked in, so it wants uint8 [0, 255].
        chw = blob[0]
        hwc = np.transpose(chw, (1, 2, 0))
        uint8 = np.clip(hwc * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(uint8)

        out = self.model.predict({"image": pil})
        # Return in stable order — caller (BaseBackend postprocess) maps by index.
        return [np.asarray(v) for _, v in sorted(out.items())]
```

- [ ] **Step 4: Add lazy import in `libreyolo/__init__.py`**

In `libreyolo/__init__.py`, add a `CoreMLBackend` entry to the `_lazy` dict (currently lines 20-33) and to `__all__` (lines 60-65).

In the `_lazy` dict insert (after the `NcnnBackend` line):

```python
        "CoreMLBackend": (".backends.coreml", "CoreMLBackend"),
```

In `__all__` (after `"NcnnBackend",`) insert:

```python
    "CoreMLBackend",
```

- [ ] **Step 5: Wire `.mlpackage` dispatch in `libreyolo/models/__init__.py`**

In `libreyolo/models/__init__.py`, after the OpenVINO directory dispatch (currently at lines 112-115) and before the ncnn directory check, add:

```python
    if Path(model_path).is_dir() and Path(model_path).suffix == ".mlpackage":
        from ..backends.coreml import CoreMLBackend

        return CoreMLBackend(model_path, nb_classes=nb_classes or 80, device=device)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_export_coreml.py -v`
Expected: PASS — `test_backend_class_importable` and `test_dispatch_mlpackage` green. (The dispatch test runs on every platform because the macOS guard is inside `__init__`, after class definition; the test patches the class itself before instantiation.)

- [ ] **Step 7: Commit**

```bash
git add libreyolo/backends/coreml.py libreyolo/__init__.py libreyolo/models/__init__.py tests/unit/test_export_coreml.py
git commit -m "feat(coreml): add CoreMLBackend + .mlpackage dispatch (#90)"
```

---

## Task 7: macOS-gated end-to-end roundtrip test

**Files:**
- Create: `tests/e2e/test_coreml_roundtrip.py`

This test only runs on macOS with `coremltools` installed and the `coreml` marker selected. It exports a small YOLOX model, loads it back through `CoreMLBackend`, and checks detection counts agree with the PyTorch backend on the bundled sample image.

- [ ] **Step 1: Create the e2e test file**

Create `tests/e2e/test_coreml_roundtrip.py`:

```python
"""End-to-end CoreML export + load roundtrip. macOS only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.coreml, pytest.mark.e2e]


@pytest.fixture(autouse=True)
def _macos_only():
    if sys.platform != "darwin":
        pytest.skip("CoreML tests require macOS")
    pytest.importorskip("coremltools")


def _load_yolox_nano():
    from libreyolo import LibreYOLO

    # Use the smallest available registered weights for fast tests.
    return LibreYOLO("LibreYOLOXn.pt")


def test_export_fp32_and_load(tmp_path):
    from libreyolo import LibreYOLO, SAMPLE_IMAGE

    model = _load_yolox_nano()
    out_path = tmp_path / "model.mlpackage"
    model.export(format="coreml", output_path=str(out_path))
    assert out_path.is_dir()

    coreml_model = LibreYOLO(str(out_path))
    result = coreml_model(image=SAMPLE_IMAGE)
    assert result["num_detections"] >= 1


def test_export_fp16_and_load(tmp_path):
    from libreyolo import LibreYOLO, SAMPLE_IMAGE

    model = _load_yolox_nano()
    out_path = tmp_path / "model_fp16.mlpackage"
    model.export(format="coreml", output_path=str(out_path), half=True)
    assert out_path.is_dir()

    coreml_model = LibreYOLO(str(out_path))
    result = coreml_model(image=SAMPLE_IMAGE)
    assert result["num_detections"] >= 1


def test_compute_units_kwarg_accepted(tmp_path):
    from libreyolo import LibreYOLO

    model = _load_yolox_nano()
    out_path = tmp_path / "model_cpu.mlpackage"
    model.export(
        format="coreml",
        output_path=str(out_path),
        compute_units="cpu_only",
    )
    assert out_path.is_dir()


def test_rfdetr_nms_true_raises(tmp_path):
    from libreyolo import LibreYOLO

    rfdetr = LibreYOLO("rf-detr-nano.pth")  # adjust to whatever the test fixture uses
    with pytest.raises(NotImplementedError, match="RF-DETR"):
        rfdetr.export(
            format="coreml",
            output_path=str(tmp_path / "rfdetr.mlpackage"),
            nms=True,
        )
```

- [ ] **Step 2: Run the e2e test (will skip on non-macOS)**

Run: `pytest tests/e2e/test_coreml_roundtrip.py -m coreml -v`
Expected on macOS with coremltools installed: tests run. Expected on Linux/CI: all 4 tests skipped with "CoreML tests require macOS".

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_coreml_roundtrip.py
git commit -m "test(coreml): add macOS e2e roundtrip tests (#90)"
```

---

## Task 8: Verify the full unit suite still passes

**Files:** none (verification only)

- [ ] **Step 1: Run the unit test suite**

Run: `pytest -m unit -q`
Expected: all existing unit tests still pass; the 16 new CoreML unit tests are included and green.

- [ ] **Step 2: Sanity-check the export factory error message**

Run: `python -c "from libreyolo.export import BaseExporter; print(sorted(BaseExporter._registry))"`
Expected output (sorted): `['coreml', 'ncnn', 'onnx', 'openvino', 'tensorrt', 'torchscript']`

- [ ] **Step 3: Sanity-check the lazy import**

Run: `python -c "import libreyolo; print(libreyolo.CoreMLBackend.__name__)"`
Expected: `CoreMLBackend`

- [ ] **Step 4: Final commit (only if anything was tweaked above)**

If steps 1-3 surfaced any minor fix, commit it:

```bash
git add -A
git commit -m "fix(coreml): address verification findings (#90)"
```

Otherwise no commit needed.

---

## Self-Review Notes

**Spec coverage**

| Spec section | Task |
|---|---|
| `.mlpackage` ML Program output | Task 3 (`convert_to="mlprogram"`, suffix in Task 5) |
| fp32 + fp16 only | Task 3 (`compute_precision` mapping) + Task 5 (`apply_model_half=False`, `supports_int8=False`) |
| Direct PyTorch trace path | Task 3 (`torch.jit.trace`) |
| `compute_units` kwarg, default `"all"` | Task 2 (helper) + Task 3 (export plumbing) + Task 5 (subclass kwarg) + Task 6 (backend) |
| Opt-in `nms=True`, RF-DETR raises | Task 4 |
| `CoreMLExporter` subclass | Task 5 |
| `CoreMLBackend` macOS-only | Task 6 |
| `.mlpackage` dispatch in factory | Task 6, Step 5 |
| `coreml` extra + `all` rollup + pytest marker | Task 1 |
| Unit tests cross-platform via mocks | Task 2-6 |
| E2E tests gated `@pytest.mark.coreml` macOS-only | Task 7 |
| Lazy import + `__all__` entry | Task 6, Step 4 |

**Out of scope (per spec):** INT8/palettization, `.mlmodel`, ONNX→CoreML, iOS sample app, configurable NMS thresholds, native batched Python inference. None of those have tasks — correct.

**Notes on what's intentionally minimal:**

- `_wrap_with_nms` (Task 4) builds a basic Pipeline with hard-coded thresholds (`0.45`/`0.25`). If the e2e test in Task 7 reveals that YOLO heads emit output names other than `confidence`/`coordinates`, an additional rename step on the detector spec will be needed. Leaving that as a follow-up keeps this plan testable end-to-end.
- Docs page (`docs/export/coreml.md`) deferred — this repo currently has no `docs/export/` directory; user-facing docs live on libreyolo.com (separate repo). A docstring on `CoreMLExporter` and `export_coreml` covers the in-repo surface; the site doc is best handled as a follow-up issue.
