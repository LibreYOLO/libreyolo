# ADR 0021: Promptable 3D detection

Status: implemented and runtime-validated
Date: 2026-09-09

## Decision

Introduce `detect3d` and the `Results.boxes3d` payload. The first integration,
`LibreWildDet3D`, is an independent MIT adapter calling an optional upstream
runtime, following the SAM 3D Body separation in ADR 0013. Upstream source is
not ported, adapted, or vendored. WildDet3D and its weights retain their SAM
License terms. The user installs the runtime separately and supplies a local
unchanged full checkpoint. This does not assert that upstream components or
weights are MIT-compatible.

The public interface reference is WildDet3D revision
`1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a`, specifically its
[interface documentation](https://github.com/allenai/WildDet3D/blob/1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a/docs/INFERENCE.md).
No upstream implementation source was used to write the adapter.

## Contract

- `Boxes3D.data` is `(N, 14)`: center xyz, dimensions wlh, quaternion wxyz,
  combined confidence, class id, 2D confidence, 3D confidence.
- Units are metres, in camera coordinates (x right, y down, z forward).
  The quaternion rotates local box axes into the camera frame. Dimensions are
  width, length, height; local x, y, z carry length, height, width. Quaternion
  sign is not canonicalized. This mapping matches vis4d v1.0.0's Apache-2.0
  `AxisMode.OPENCV` contract and is pinned by a golden corner fixture.
- `Boxes3D.intrinsics` is the shared original-image `(3, 3)` calibration.
  It is preserved when detections are sliced and moved with tensor payloads.
- `Results.boxes` contains aligned original-image xyxy boxes and the same
  combined confidence/class ids. No NMS, deduplication, or score reinterpretation
  is added by the adapter. Per-prompt output groups are concatenated in order.
- Summary/JSON records include metric geometry under `box3d`. Normalizing a
  result normalizes 2D coordinates only; metres and quaternions stay unchanged.
- `result.plot(image)` projects wireframe cuboids using the calibration,
  clipping edges at the near plane. File-backed results may omit `image`.
- Text prompts provide class names. Box/point prompts use `prompt_N` names
  corresponding to the upstream prompt-order ids.

Known camera calibration is required initially. Upstream documentation is
inconsistent about fallback versus predicted intrinsics, and does not specify
unambiguously the returned predicted calibration's image space. Guessing
would silently misproject 3D results. Predicted-calibration support can be
added once that contract is verified. Input depth is optional and must match
the original image. Predicted depth output is not exposed because its output
canvas is not specified precisely enough to meet the original-canvas rule.

## Use

Install the separate runtime following its [installation guide](https://github.com/allenai/WildDet3D#installation).
Its upstream CUDA stack uses Python 3.11, PyTorch 2.5.1/CUDA 12.1 and compiled
vis4d operators. It is not installed by a LibreYOLO extra. Obtain the upstream
full checkpoint from the publisher under its applicable terms, or omit
`model_path` to download LibreYOLO's byte-identical mirror. The mirror is pinned
to Hugging Face revision `e2baa17eb2279225094a2d7e7d7620ff06dc4676` and
verified against SHA-256
`f8b6a9e548f733ba62625a0d2adc4b0f4fdb6007ee11d9927f9c1027010fee57`.

On CUDA, expose the checkout on `PYTHONPATH` and the adapter calls it directly.
On macOS, pass `runtime_path=` (or `WILDDET3D_PATH`) and optionally a dedicated
`runtime_python=` (or `WILDDET3D_PYTHON`). `device="auto"` selects CPU when
CUDA is absent. The CPU path runs the upstream package in a private long-lived
worker because upstream creates CUDA scheduling streams even for CPU inference.
LibreYOLO supplies serial stream semantics only inside that worker. Image and
tensor data cross private pipes as JSON and typed bytes; the main process's
PyTorch module is never patched. The worker is reused for all predictions and
is released by `close()`, a context manager, timeout, or owner collection.

MPS is rejected explicitly. Real MPS forward testing reached multiple upstream
mixed-device failures in SAM's prompt and geometry encoders. Automatic fallback
would be slower and less predictable than the validated CPU path.

```python
import numpy as np
from libreyolo import LibreWildDet3D

model = LibreWildDet3D()
result = model.predict("image.jpg", intrinsics=np.load("intrinsics.npy"),
                       text=["car", "person"])
print(result.boxes3d.xyz)
result.plot().save("cuboids.png")
```

macOS with an isolated upstream environment:

```sh
git clone --recurse-submodules https://github.com/allenai/WildDet3D.git
git -C WildDet3D checkout 1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a
git -C WildDet3D submodule update --init --recursive
python3.11 -m venv WildDet3D/.venv
WildDet3D/.venv/bin/pip install vis4d==1.0.0 --no-deps
WildDet3D/.venv/bin/pip install -r WildDet3D/requirements.txt \
  absl-py termcolor psutil
WildDet3D/.venv/bin/pip install --upgrade torch torchvision
WildDet3D/.venv/bin/pip install 'setuptools<80'
```

Do not install `vis4d_cuda_ops` on macOS. It is needed by the upstream CUDA
environment, while the validated CPU image path does not call it. The final
upgrade is intentional: upstream's CUDA requirements pin PyTorch 2.5.1, which
fails in SAM's CPU geometry-prompt path on Apple Silicon. Re-pinning setuptools
is also required because SAM still imports `pkg_resources`. The clean macOS
check uses PyTorch 2.14.0, torchvision 0.29.0, and setuptools 79.0.1.

```python
with LibreWildDet3D(
    "wilddet3d_alldata_all_prompt_v1.0.pt",
    runtime_path="/path/to/WildDet3D",
    runtime_python="/path/to/wilddet3d-venv/bin/python",
) as model:
    result = model("image.jpg", intrinsics=np.load("intrinsics.npy"), text=["dog"])
```

```sh
libreyolo wilddet3d model=wilddet3d_alldata_all_prompt_v1.0.pt \
  source=image.jpg intrinsics=intrinsics.npy 'text=["car","person"]' --json
```

The command accepts both `key=value` and `--key value`. `--help-json` lists
all options. `bboxes` accepts xyxy boxes; `points` and `labels` accept grouped
pixel xy and binary labels; `prompt_mode` is geometric or visual. Supply
exactly one prompt kind. `set_classes()` supplies a reusable text vocabulary.
Shared calibration and prompts apply to every image in list/directory calls.
A single image returns `Results`; lists/directories return lists;
`stream=True` returns a generator.

`conf`, `conf3d`, and `iou` are constructor settings, matching the upstream
builder. Geometric prompts retain upstream one-result-per-prompt behavior,
which bypasses the confidence floors. `use_depth=True` requires a depth map on
every call; its default false requires no map. CLI `depth=...npy` sets both.

## Scope and evidence

This integration is inference-only. Training needs a 3D annotation loader,
label contract and practical convergence evidence, none of which the existing
2D training pipeline supplies. `train`, `val`, `export`, and `track` raise
explicit unsupported errors. Upstream benchmark evaluation remains the route
to accuracy measurement; no generic 2D mAP result is reported as 3D accuracy.
No datasets are downloaded or redistributed. Generic factory/GUI integration
is deferred because those surfaces do not carry camera and prompt inputs.
YOLO9 and RF-DETR remain the 2D flagships; neither has a 3D head to wire here.

The real Stage 3 checkpoint was checked on ten COCO128 images with ten text
prompts (`dog`, `cat`, `horse`, `bicycle`, `bus`, `car`, `chair`, `couch`,
`bottle`, `person`) and identical assumed pinhole calibration. Direct upstream
CUDA and LibreYOLO CUDA matched exactly for 2D boxes, 3D boxes, combined score,
2D confidence, 3D confidence, class id, and intrinsics: maximum absolute
difference `0.0` for every field and image. Detection counts also matched the
Mac CPU run for all ten images. Lossless Mac CPU versus direct upstream CUDA
comparison found maximum drift of 0.173 pixels in 2D boxes, 0.0374 metres in
3D centers, 0.0195 metres in dimensions, 0.00250 in quaternion components,
and 0.00843 across confidence values. Class ids and intrinsics were exact.
Warm RTX 3090 calls took 0.876 to 0.925 seconds; Mac CPU calls took 20.777 to
21.533 seconds in the final run after the 38.163-second cold call. These
timings are evidence from one machine each, not performance guarantees.

This proves adapter parity and Mac CPU execution. It does not prove model
accuracy: the test uses approximate intrinsics because COCO128 carries no
camera calibration, and no 3D benchmark labels. Export compatibility remains
unverified and blocked.

The manual runtime check is collected under the `wilddet3d` marker:

```sh
LIBREYOLO_WILDDET3D_CHECKPOINT=/path/to/upstream.pt \
  pytest tests/e2e/test_wilddet3d.py -m e2e -q
```

Run it in the provisioned CUDA environment with the runtime on `PYTHONPATH`.
It compares the adapter's geometry and scores to a direct public-API call on
the same model, and checks that projection produces an overlay. This verifies
adapter mapping, not an independently implemented network or benchmark mAP.
