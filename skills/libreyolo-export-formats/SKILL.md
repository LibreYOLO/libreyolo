---
name: libreyolo-export-formats
description: >-
  Work on LibreYOLO's export subsystem: debug a failing export, fix an
  exported model that predicts differently from the .pt, extend a format's
  coverage to a new family/task, or add a new export format. Use when someone
  reports "export to X fails", "the ONNX/TensorRT model gives different
  results", asks which formats a family supports, or wants a new format
  wired. Covers the exporter/backend twin architecture, the metadata
  round-trip contract, per-format quirks and extras, the parity-testing
  method, and the support-claim bar. For a user just running export, see
  use-libreyolo; for Hailo specifically, libreyolo-export-hailo.
---

# LibreYOLO export formats (developer guide)

Export is two mirrored subsystems, and almost every export bug is a mismatch
between them:

- **Exporters** (`libreyolo/export/`): `BaseExporter` ABC + one subclass per
  format (`OnnxExporter`, `TorchScriptExporter`, `TensorRTExporter`,
  `OpenVINOExporter`, `NcnnExporter`, `TFLiteExporter`, `CoreMLExporter` in
  `exporter.py`, with format-specific helpers in sibling modules). They
  serialize the graph **and embed LibreYOLO metadata** (family, size, task,
  imgsz, class names, pose keypoint shape, ...).
- **Backends** (`libreyolo/backends/`): `BaseBackend` + one subclass per
  runtime. `LibreYOLO("best.onnx")` routes here; the backend reads the
  embedded metadata, reproduces the family's preprocessing and
  postprocessing (letterbox, NMS or DETR top-k, task-specific decode) in
  numpy, and returns the same `Results` a `.pt` would.

The two REVIEW.md axioms that govern all of it: **backends must behave like
models** (same API, same Results, original-canvas coordinates) and
**exported runtimes round-trip metadata** (what the exporter writes, the
backend must read; a model loaded from an export must not need the user to
re-specify imgsz/task/classes).

## The support tiers (do not overclaim)

Formats are not equally supported, and support is per (format, family,
task), not per format. ONNX is the release-gated supported backend; the
others are covered by e2e but have extended runtime coverage
(`supported_backend` vs `extended_backend` markers in `tests/e2e/`).
`libreyolo formats` prints the format list with capabilities; the e2e files
(`test_onnx.py`, `test_tensorrt.py`, ...) are the truth for which families
each format actually covers. Before telling anyone "family X exports to Y",
find the (X, Y) case in those tests or run it yourself.

## Per-format quirks (the ones that cost hours)

- **ONNX**: the Python API defaults `dynamic=True`; downstream consumers
  with static-shape compilers (TensorRT profiles aside, Hailo, some
  runtimes) need `dynamic=False`. `simplify=True` runs onnxsim. `nms=True`
  embeds NMS into the graph (changes the output contract; the backend knows,
  external consumers must be told). Raw detect output for external
  consumers is the `[1, 84, 8400]`-style head tensor; at least one external
  C++ project depends on that exact contract, and `tests/e2e/`'s
  sam3dbody contract test guards it: do not reshape it casually.
- **TensorRT**: version-pinned (`tensorrt-cu12==...` in pyproject, matching
  `[[tool.uv.dependency-metadata]]`); engines are GPU-arch-specific
  artifacts, never redistributable; fp16 is where accuracy drift first
  appears, compare against fp32 before blaming the model.
- **TFLite**: routes through onnx2tf (needs Python 3.12+ per the extra's
  constraint) and includes graph surgery for RF-DETR (GridSample rewrite);
  it is the most fragile chain, keep its extra versions synced with
  pyproject comments.
- **OpenVINO / ncnn**: install their extras to test; ncnn goes through pnnx.
- **Axelera**: INT8 only, YOLO-NAS detect and pose only. The exporter cuts
  the graph after the head convolutions (`_YOLONASRawHeadsWrapper`) and the
  backend decodes the raw heads on the host
  (`libreyolo/postprocess/yolonas_raw.py`); `metadata.yaml` carries a
  versioned `output_layout` and the compiled model's output order
  (`output_names`), which can differ from the graph's. The SDK is not a pip
  extra (Axelera's own index, EULA); unit tests fake it. Details in
  `skills/libreyolo-export-axelera/` and `docs/axelera.md`.
- **CoreML**: macOS-only end to end; e2e coverage lives in
  `test_coreml_roundtrip.py` and skips elsewhere.
- **Rectangular imgsz** is only supported by the formats in
  `_RECTANGULAR_EXPORT_FORMATS` (see `exporter.py`); everywhere else,
  square-check errors are intentional.
- Some families need export-time wrappers or state snapshots (RT-DETR
  wrapper, RF-DETR export-state snapshot/restore in `exporter.py`); when an
  export mutates model behavior, that pattern is the sanctioned fix.

## Debugging "the exported model is wrong"

Work the pipeline in order; the bug class differs by stage:

1. **Export succeeded but predictions differ from `.pt`**: run both on the
   same image and compare stage by stage. Preprocessing first (the backend
   reimplements letterbox in numpy; a 1px offset here looks like "slightly
   wrong boxes"), then raw tensor closeness (fp32 should be ~1e-5; fp16
   looser), then postprocess (NMS thresholds, top-k count, class-map).
   The parity assertions in `tests/e2e/test_deterministic_inference.py` and
   the backend unit tests (`tests/unit/test_backend_postprocess.py`,
   `test_backend_validation_adapter.py`) show the expected tolerances.
2. **Export itself fails**: check the extra is installed
   (`libreyolo checks` reports export backends), then whether the op is
   task-specific (pose/seg heads export differently; the exporter embeds
   task metadata like keypoint shapes). Only then debug the converter
   stack, and pin versions before patching code.
3. **Backend load fails on a file that exported fine**: metadata round-trip
   bug. Inspect what was embedded (for ONNX, the model's metadata_props)
   and what `BaseBackend._read_metadata_*` expects; fix the *pair*, and add
   the missing key to both sides plus a unit test.

## Extending coverage (family or task to an existing format)

1. Make the model's forward export-clean for the format (no data-dependent
   Python control flow on tensor values in the traced path).
2. Teach the exporter any task metadata the backend will need (the pose
   `kpt_shape` flow in `exporter.py` is the template).
3. Teach the backend the family's decode: NMS-using YOLO-grid vs NMS-free
   vs DETR top-k selection is centralized (`_is_nms_free_family`,
   `_rfdetr_num_select` in `backends/base.py`); route, don't fork.
4. Add the e2e case to that format's test file with the right markers
   (`libreyolo-write-e2e-tests`), and a val-parity check if the family has
   one (exported-backend val must agree with native val; there is adapter
   machinery for exactly this).

## Adding a whole new format

Copy the twin structure: exporter subclass (declare `format_name`, deps via
`requires_*`, write metadata), backend subclass (read metadata, reuse the
shared pre/postprocess helpers in `BaseBackend`, never fork them), a lazy
export in `libreyolo/__init__.py`, an optional-dependency extra in
`pyproject.toml` (never a core dep), `libreyolo formats` row, an
`extended_backend`-marked e2e file, and docs. A new format uses the extended
marker until family coverage and parity tests say
otherwise. Note the precedent that not everything becomes a format: when the
toolchain cannot be a pip dependency (Hailo's proprietary compiler), ship a
skill/doc for the two-stage flow instead of a `format=` target.

## Related

- `skills/use-libreyolo/`: the user-facing export + exported-inference API.
- `skills/libreyolo-export-hailo/`: the no-format precedent, done right.
- `skills/libreyolo-write-e2e-tests/`: markers for export coverage.
- `docs/testing.md`: which backends gate releases vs run on the extended schedule.
