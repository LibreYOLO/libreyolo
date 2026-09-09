# ADR 0021: Promptable 3D detection

Status: implemented; real-runtime validation pending
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

- `Boxes3D.data` is `(N, 14)`: center xyz, dimensions whl, quaternion wxyz,
  combined confidence, class id, 2D confidence, 3D confidence.
- Units are metres, in camera coordinates (x right, y down, z forward).
  The quaternion rotates local box axes into the camera frame. Dimensions
  correspond to the local x, y, z axes; quaternion sign is not canonicalized.
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

Install the separate runtime following its [installation guide](https://github.com/allenai/WildDet3D#installation),
and expose the checkout on `PYTHONPATH`. Its tested stack uses Python 3.11,
PyTorch 2.5.1/CUDA 12.1 and compiled vis4d operators. It is not installed by a
LibreYOLO extra. Obtain the upstream full checkpoint from the publisher under
its applicable terms; the adapter performs no automatic download.

```python
import numpy as np
from libreyolo import LibreWildDet3D

model = LibreWildDet3D("wilddet3d_alldata_all_prompt_v1.0.pt")
result = model.predict("image.jpg", intrinsics=np.load("intrinsics.npy"),
                       text=["car", "person"])
print(result.boxes3d.xyz)
result.plot().save("cuboids.png")
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

CPU tests exercise calibration, quaternion projection, clipping, result
alignment, serialization, prompt forwarding, lazy dependency loading and CLI
behavior using synthetic outputs. These do not prove upstream loading,
inference quality, or numerical parity. Real CUDA inference and downstream
export compatibility must be recorded separately before claiming them.

The manual runtime check is collected under the `wilddet3d` marker:

```sh
LIBREYOLO_WILDDET3D_CHECKPOINT=/path/to/upstream.pt \
  pytest tests/e2e/test_wilddet3d.py -m e2e -q
```

Run it in the provisioned CUDA environment with the runtime on `PYTHONPATH`.
It compares the adapter's geometry and scores to a direct public-API call on
the same model, and checks that projection produces an overlay. This verifies
adapter mapping, not an independently implemented network or benchmark mAP.
