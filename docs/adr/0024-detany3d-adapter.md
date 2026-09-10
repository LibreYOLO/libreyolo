# ADR 0024: DetAny3D runtime adapter

Status: implemented and CPU runtime-validated
Date: 2026-09-10

## Decision

Expose `LibreDetAny3D` and `libreyolo detany3d` for box, positive-point and
text-prompted `detect3d` inference. The full upstream model is listed as size
`h` at 896 pixels. It predicts camera calibration. It is a sibling API,
outside the generic `LibreYOLO(...)` checkpoint factory.

The adapter follows the Apache-2.0 public inference interfaces in
`OpenDriveLab/DetAny3D` revision
`10e484be0837d80aa33cff6ed95a9be834a3f794`. The separately installed runtime
includes UniDepth under CC BY-NC 4.0. LibreYOLO does not bundle or derive its
implementation, and this adapter grants no commercial-use rights to that
runtime. The original full checkpoint is loaded strictly without conversion.
Checkpoint redistribution terms remain unresolved, so there is no automatic
download or Hugging Face mirror.

## Runtime and public contract

Provide the upstream checkout, its dedicated Python interpreter and the full
official checkpoint. The interpreter must contain the upstream dependencies,
including SAM, DINOv2, UniDepth and xFormers interfaces. The complete checkpoint
already contains the learned parameters for these components; redundant
backbone checkpoint loading is disabled. Text prompts additionally require
GroundingDINO Swin-B and its checkpoint. The pinned upstream checkout contains
an absolute GroundingDINO symlink; install GroundingDINO into the runtime
environment separately instead of relying on that link.

```python
from libreyolo import LibreDetAny3D

with LibreDetAny3D(
    "/models/detany3d.pth",
    runtime_path="/runtimes/DetAny3D",
    runtime_python="/runtimes/venv/bin/python",
    grounding_checkpoint="/models/groundingdino_swinb_cogcoor.pth",
    device="cpu",
) as model:
    result = model("street.jpg", text=["car", "person"])
    result.plot().save("cuboids.png")
```

```sh
libreyolo detany3d model=/models/detany3d.pth source=street.jpg \
  runtime_path=/runtimes/DetAny3D runtime_python=/runtimes/venv/bin/python \
  'bboxes=[[200,150,600,500]]' device=cpu --json
```

`DETANY3D_PATH` and `DETANY3D_PYTHON` are environment alternatives. CLI options
also accept `--key value`. Box and point coordinates refer to original image
pixels. `(N,2)` points describe one object, while `(G,N,2)` describes G objects;
all points are positive. Points cannot be combined with boxes or text. Boxes
and text can be combined. `set_classes()` stores a reusable text vocabulary.

Single images return one `Results`; lists and directories return lists;
`stream=True` yields results. `Results.boxes` and `Results.boxes3d` are aligned.
The estimated intrinsics and refined 2D boxes are mapped back through the
upstream resize and central crop to the original canvas. Cuboids follow ADR
0021: metres, gravity centres, OpenCV camera axes, width/length/height, and
quaternion wxyz. The source local width/height/length basis is transformed into
the canonical length/height/width basis before quaternion construction.

Text scores are the upstream 2D detector scores, controlled by `conf=0.37` and
`text_threshold=0.25`. Geometric prompts have score 1. The upstream quality head
selects candidates but is not its exported detection confidence; `conf3d` is
therefore neutral 1. Class names are registered consistently across calls.
Training, validation, export and tracking raise explicit unsupported errors.

## Evidence and limits

Nine CPU cases matched a direct upstream reference using its original public
preprocessing, decoder and corner utilities: boxes, positive points, text,
combined boxes/text, grouped points, an empty text result, and three modes on
an asymmetric crop. Labels and counts matched exactly. Checks cover scores
(absolute tolerance 1e-5), 2D boxes (0.05 pixels), centres (0.002 metres),
dimensions (0.001 metres), calibration (0.01 pixels), and corner-set distance
(0.005 metres). Relative tolerances are recorded in the manual e2e test.
An extracted wheel also completed a real box-prompt prediction.

The CPU worker uses PyTorch attention through the public xFormers API and the
existing permissive LibreYOLO deformable-attention implementation. The
reference uses explicit attention and MMCV's portable deformable attention.
These substitutions leave upstream model code and learned parameters intact.
They are installed only in the child process. The worker exposes only its own
LibreYOLO package to avoid shadowing the separate interpreter's dependencies.

CUDA requires the upstream compiled runtime and has not been validated. MPS,
benchmark accuracy, training and export are not claimed. Manual parity requires
locally staged weights and reference outputs; it is not part of the nightly
download catalogue. No model weights or reference images are included here.
