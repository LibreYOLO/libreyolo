# CoreML Export + Backend Support

**Issue:** #90 — Add CoreML support
**Date:** 2026-04-26
**Status:** Draft

## Goal

Add Apple Core ML as a supported export target so users can deploy LibreYOLO models (YOLOv9, YOLOX, RF-DETR) to iPhone, iPad, Mac, Apple Watch, and Vision Pro. Ship both an exporter (writes `.mlpackage`) and a Python inference backend (loads `.mlpackage` on macOS), matching the convention every other format follows in this repo.

## Scope decisions (locked in during brainstorm)

| Decision | Choice | Rationale |
|---|---|---|
| Export only vs export + backend | **Export + Python backend** | Symmetry with onnx/tensorrt/openvino/ncnn — every existing format has a paired backend. |
| Model format | **`.mlpackage` (ML Program) only** | Default in coremltools ≥7. Required for fp16 on Neural Engine. Apple's recommended path. Min target: iOS 15 / macOS 12. |
| Precision | **fp32 + fp16 only** | INT8/palettization story across coremltools 6/7/8 is messy; ship the common case first. |
| Conversion path | **Direct PyTorch trace → `coremltools.convert`** | The ONNX→CoreML path is dead (deprecated in coremltools 5, removed in 7). Direct PyTorch is what coremltools maintains. |
| Compute units | **`compute_units` kwarg, default `"all"`** | Plumbed through both export (baked into model) and backend (load-time hint). Useful for benchmarking and CPU-only repro. |
| NMS | **`nms` flag, default `False`** | Default-off matches every other exporter. Opt-in embeds Apple's `NonMaximumSuppression` for iOS app developers. RF-DETR raises `NotImplementedError` when `nms=True` (output layout doesn't fit standard NMS). |

## Architecture

Two new modules + one new exporter subclass + one extras entry:

```
libreyolo/export/coreml.py            # export_coreml() — direct trace → ct.convert
libreyolo/export/exporter.py          # + CoreMLExporter subclass (auto-registers)
libreyolo/backends/coreml.py          # CoreMLBackend — coremltools.models.MLModel runtime
libreyolo/backends/__init__.py        # extend suffix → backend dispatch (.mlpackage)
pyproject.toml                        # add coreml extra + 'all' rollup + pytest marker
docs/export/coreml.md                 # new format doc page
tests/unit/test_export_coreml.py      # mocked, runs everywhere
tests/e2e/test_coreml_roundtrip.py    # @pytest.mark.coreml, macOS-only
```

## Component design

### `CoreMLExporter` (in `libreyolo/export/exporter.py`)

```python
class CoreMLExporter(BaseExporter):
    format_name = "coreml"
    suffix = ".mlpackage"
    requires_onnx = False
    supports_int8 = False
    apply_model_half = False  # ct.convert handles precision via compute_precision

    def _export(self, nn_model, dummy, *, output_path, precision, metadata,
                compute_units="all", nms=False, **kwargs):
        from .coreml import export_coreml
        return export_coreml(
            nn_model, dummy,
            output_path=output_path,
            precision=precision,
            compute_units=compute_units,
            nms=nms,
            metadata=metadata,
            model_family=self.model._get_model_name(),
        )
```

### `export_coreml()` (in `libreyolo/export/coreml.py`)

Flow:
1. `traced = torch.jit.trace(nn_model, dummy)` — head export-mode is already set by `_model_context`.
2. Build `ct.ImageType` input with `scale=1/255.0`, `bias=[0,0,0]`, name `"image"`, shape `dummy.shape`. (Standard YOLO normalization; no mean subtraction.)
3. `mlmodel = ct.convert(traced, inputs=[image_input], convert_to="mlprogram", compute_precision=ct.precision.FLOAT16 if precision=="fp16" else ct.precision.FLOAT32, minimum_deployment_target=ct.target.iOS15)`.
4. `mlmodel.compute_unit = _to_compute_unit(compute_units)` (helper maps the four string values to `ct.ComputeUnit.*`).
5. If `nms=True`:
   - If `model_family` is RF-DETR → `raise NotImplementedError("nms=True is not supported for RF-DETR; export with nms=False and run NMS in your application.")`.
   - Otherwise wrap in `coremltools.models.pipeline.Pipeline` with a `NonMaximumSuppression` model spec. Defaults: `iou_threshold=0.45`, `confidence_threshold=0.25` (LibreYOLO defaults). Output names: `"confidence"` (N×nb_classes) and `"coordinates"` (N×4, normalized xywh).
6. Stuff `metadata` dict into `mlmodel.user_defined_metadata` after stringifying values (CoreML requires string values; `names` JSON-encoded same as ONNX path).
7. `mlmodel.save(output_path)`.

### `CoreMLBackend` (in `libreyolo/backends/coreml.py`)

Mirrors `OnnxBackend`'s public surface. Responsibilities:
- **Load:** `import coremltools as ct; self.model = ct.models.MLModel(path, compute_units=_to_compute_unit(compute_units))`. Read `user_defined_metadata` to recover class names, imgsz, model family, precision.
- **Predict:** CoreML's Python API does not natively batch via `predict()`. Loop over the input batch, convert each `np.ndarray` → `PIL.Image`, call `model.predict({"image": img})`, stack outputs to a `(B, ...)` tensor, then hand off to the same NMS/decode helper used by `OnnxBackend`.
- **Platform guard:** at import time, `if sys.platform != "darwin": raise RuntimeError("CoreML inference requires macOS")`.
- **Metadata round-trip:** if the model was exported with `nms=True`, `predict` returns post-NMS detections directly — skip the Python NMS step.

### Backend dispatch

Extend whatever maps file suffix → backend class so `.mlpackage` (a directory, not a file) routes to `CoreMLBackend`. Implementation note: `.mlpackage` is a bundle directory; existence check should use `Path.is_dir() and path.suffix == ".mlpackage"`.

### Validation

`_validate` inherited as-is from `BaseExporter`:
- `int8=True` → raises (since `supports_int8=False`, the user gets the standard error).
- `half=True` → routes to `compute_precision=FLOAT16`. Model is *not* `.half()`-ed before tracing (`apply_model_half=False`).

### `pyproject.toml` changes

```toml
[project.optional-dependencies]
coreml = [
    "coremltools>=7.2",
]

# rollup
all = [
    ...,
    "libreyolo[coreml]",
]

[tool.pytest.ini_options]
markers = [
    ...,
    "coreml: tests requiring CoreML (macOS only)",
]
```

## Tests

### Unit (`tests/unit/test_export_coreml.py`)
Runs on every platform. Mocks `coremltools` so it doesn't need to be installed for the test to pass logic checks.
- `_to_compute_unit` mapping covers all four strings + invalid value raises.
- `export_coreml` calls `ct.convert` with `convert_to="mlprogram"`, correct `compute_precision` for fp32/fp16, correct `ImageType` scale/bias.
- `compute_units` kwarg flows through to `mlmodel.compute_unit`.
- `nms=True` on YOLOX/YOLOv9 wraps in pipeline; on RF-DETR raises `NotImplementedError`.
- `metadata` dict gets stringified into `user_defined_metadata`.
- `int8=True` raises via base validator.

### E2E (`tests/e2e/test_coreml_roundtrip.py`)
`@pytest.mark.coreml` + `pytest.skip` if `sys.platform != "darwin"` or coremltools not installed.
- Export `LibreYOLOXn` at fp32 and fp16 → load via `CoreMLBackend` → run on `assets/parkour.jpg` → assert detection count and top-1 class match the ONNX backend on the same image (within tolerance for fp16).
- Export with `nms=True` → assert returned tensor shapes are post-NMS (variable count) and a Python-side NMS is *not* run.
- RF-DETR + `nms=True` → assert `NotImplementedError`.

CI: macOS runner gets `pip install libreyolo[coreml]` and runs `pytest -m coreml`. Linux runners skip.

## Documentation

- New page `docs/export/coreml.md` following the structure of existing format docs:
  - Install: `pip install libreyolo[coreml]`
  - Quick example: `model.export(format="coreml")` → `.mlpackage`
  - Precision: fp32 vs fp16 (when each makes sense for Neural Engine)
  - `compute_units`: when to override the default
  - `nms=True`: what it does, which model families support it
  - Deployment target: iOS 15 / macOS 12
  - Loading from Python (macOS): `LibreYOLO("model.mlpackage")` example
- Add `coreml` row to the export-format table in main README and docs.

## Out of scope (explicit YAGNI)

- INT8 / palettization / weight compression
- `.mlmodel` (legacy NeuralNetwork) format
- ONNX → CoreML conversion path
- iOS / Swift deployment example app
- Configurable NMS thresholds at export time (user can re-set on the saved spec; default to LibreYOLO's standard 0.45 / 0.25)
- Native batched inference in the Python backend (CoreML's Python `predict` is per-sample; we loop)

## Risks & mitigations

- **`coremltools.convert` failing on a specific op** in YOLOv9/YOLOX/RF-DETR — surface the converter's traceback verbatim; don't catch. Document known-good model variants in `docs/export/coreml.md` as we discover them.
- **`.mlpackage` being a directory** confuses code that assumes file outputs — explicitly handle `is_dir()` in any cleanup or backend-dispatch logic.
- **macOS-only backend** means CI coverage is partial — unit tests carry most of the logic verification; e2e on macOS catches integration regressions.
- **fp16 accuracy drift on Neural Engine** — e2e test uses tolerance bands rather than exact match.
