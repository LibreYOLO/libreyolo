# ADR 0022: 3D-MOOD runtime adapter

Status: implemented and runtime-validated
Date: 2026-09-09

## Decision

Add `Libre3DMOOD` as the second `detect3d` sibling API. It uses the existing
`Results.boxes3d` camera-frame contract from ADR 0021 and also returns the
model's metric prediction as `Results.depth_map`. Swin-T (`t`) and Swin-B (`b`)
are supported. This is an inference-only adapter around a separately installed
3D-MOOD checkout; it is not registered in the state-dict `LibreYOLO(...)`
factory because its upstream checkpoint and runtime remain intact.

The code reference is `cvg/3D-MOOD` revision
`41bb2904932d91507338e75ae4c802d67616ca1b`, Apache-2.0. Official checkpoints
come from `RoyYang0714/3D-MOOD` revision
`3d1fab552189f1a62fdb60ebb606d20625a30b90`, whose model card declares
Apache-2.0. LibreYOLO mirrors the learned parameters unchanged under canonical
filenames and verifies the original SHA-256 digest.
The immutable mirror revisions are `f1e163b249b43a8bb7b8b4ef4f446828270e35ef`
for Swin-T and `a4f6115d285a6983439f0e54f20c44b0ce87c261` for Swin-B.

The upstream installation guide requires `SysCV/vis4d_cuda_ops`, whose
repository has no declared license. LibreYOLO does not install, import, or copy
that extension. The private worker exposes only the import interface expected
by Apache-2.0 Vis4D and executes multi-scale deformable attention through
LibreYOLO's existing Apache-2.0 portable PyTorch implementation. Training-time
backward and 3D-IoU kernels are deliberately unavailable.

## Public contract

- `Libre3DMOOD(model_path=None, size=None, device="auto", conf=0.1,
  iou=0.5, max_det=100, runtime_path=None, runtime_python=None)` constructs a
  long-lived isolated worker. Size defaults to `t` and is inferred from either
  official checkpoint filename; an explicit mismatch raises.
- `predict(source, intrinsics=K, text=[...])` takes original-image calibration
  and text categories. `set_classes()` supplies a reusable vocabulary.
- A single image returns one `Results`; lists and directories return a list;
  `stream=True` returns a generator.
- `Results.boxes` and `Results.boxes3d` are row-aligned on the original canvas.
  Cuboids use metres, OpenCV camera axes, dimensions width/length/height, and
  quaternion wxyz, matching ADR 0021.
- The upstream model emits one language-conditioned detection score and no
  separate 3D confidence. `Boxes3D.conf` and `conf2d` contain that score;
  `conf3d` is the neutral value 1.0. This avoids inventing a geometry score.
- `Results.depth_map` is the upstream metric-depth prediction resampled to the
  original `(H, W)` canvas.
- Class-agnostic 2D NMS makes the public `iou` argument effective. `conf` and
  `max_det` are applied inside the upstream result decoder.
- `train`, `val`, `export`, and `track` raise explicit unsupported errors.

## Installation and use

Use a dedicated Python 3.11 environment because the pinned upstream runtime
requires Transformers 4.x while other LibreYOLO optional families may use
Transformers 5.x.

```sh
git clone https://github.com/cvg/3D-MOOD.git
git -C 3D-MOOD checkout 41bb2904932d91507338e75ae4c802d67616ca1b
python3.11 -m venv 3D-MOOD/.venv
3D-MOOD/.venv/bin/pip install vis4d==1.0.0 transformers==4.46.3 \
  fairscale mmengine nltk fvcore ml-collections
3D-MOOD/.venv/bin/pip install -e 3D-MOOD
```

Do not install `vis4d_cuda_ops`. Point LibreYOLO at the checkout and dedicated
interpreter:

```python
import numpy as np
from libreyolo import Libre3DMOOD

with Libre3DMOOD(
    size="t",
    runtime_path="/path/to/3D-MOOD",
    runtime_python="/path/to/3D-MOOD/.venv/bin/python",
) as model:
    result = model(
        "room.jpg",
        intrinsics=np.load("intrinsics.npy"),
        text=["chair", "table"],
    )
    result.plot().save("cuboids.png")
    print(result.depth_map.mean)
```

The corresponding CLI supports `key=value` and `--key value` forms:

```sh
libreyolo 3dmood source=room.jpg intrinsics=intrinsics.npy \
  'text=["chair","table"]' size=t runtime_path=/path/to/3D-MOOD \
  runtime_python=/path/to/3D-MOOD/.venv/bin/python --json
```

`MOOD3D_PATH` and `MOOD3D_PYTHON` provide the same runtime settings through
environment variables.

## Evidence and limits

The official Swin-T demo image was executed through both the direct public
upstream components and `Libre3DMOOD` on Apple Silicon CPU. The 2D boxes,
camera-frame cuboids, scores, class ids, and full metric depth map matched
exactly (`max_abs_diff=0.0`). MPS produced the same five detections; relative
to CPU its maximum drift was 0.0643 pixels for 2D coordinates, 0.00243 across
the full `Boxes3D` payload, and 0.0000167 metres in the depth map. Warm calls
took about 2.4 seconds on CPU and 0.9 seconds on MPS on the validation Mac.
Swin-B also completed a real MPS inference on the same image.

The CUDA worker path is implemented but not yet runtime-validated. Two guarded
RTX 3090 attempts were terminated while the provider was still pulling its
container image; neither reached SSH or executed model code. The committed e2e
case selects CUDA automatically on a CUDA host so that check can be completed
without changing the implementation.

This proves runtime mapping and Mac execution. It does not prove benchmark
accuracy. Accuracy evaluation still requires the upstream Omni3D, Argoverse 2,
or ScanNet data and evaluator. The adapter cannot train without a LibreYOLO 3D
annotation and metric contract, and it cannot export until text, calibration,
depth, and cuboid outputs have a backend contract.
