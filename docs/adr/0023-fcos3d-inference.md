# ADR 0023: Native FCOS3D inference

Status: implemented; CPU inference validated
Date: 2026-09-10

## Interface

`LibreFCOS3D` is a native PyTorch implementation of FCOS3D R101-DCN for the
ten nuScenes classes. It uses the `detect3d` and `Results.boxes3d` contracts
in ADR 0021. It is a sibling API because prediction requires camera
calibration that the generic factory and GUI do not currently carry.

```python
import numpy as np
from libreyolo import LibreFCOS3D

model = LibreFCOS3D("fcos3d-official.pth", device="cpu")
result = model("image.jpg", intrinsics=np.load("intrinsics.npy"), conf=0.15)
result.plot().save("cuboids.png")
```

```sh
libreyolo fcos3d model=fcos3d-official.pth source=image.jpg \
  intrinsics=intrinsics.npy conf=0.15 save=true --json
```

The CLI also accepts `--key value`. A single image returns `Results`, a list
or directory returns a list, and `stream=True` returns a generator. Shared
calibration applies to all images in a multi-image call. The calibration
must describe each original image, with positive focal lengths and zero
skew. Pixels are not resized: BGR Caffe mean subtraction and right/bottom
padding to a multiple of 32 match the official test pipeline.

CPU and CUDA are callable; CUDA has not been validated. MPS is rejected
because torchvision's deformable convolution has no MPS kernel. The runtime
requires neither MMCV nor a separate model environment.

## Geometry and score contract

The head predicts gravity-center xyz, local xyz extents, and camera yaw.
The adapter stores dimensions as wlh = (z extent, x extent, y extent) and a
scalar-first quaternion rotating around camera y. Centres and dimensions
are in metres. The 2D boxes are clipped projected cuboid hulls, aligned with
the 3D rows. Cuboids crossing the near plane are clipped before projection.

Ranking confidence is sigmoid(class logit) times sigmoid(centerness).
`conf2d` stores that joint score and `conf3d` is 1, a neutral factor rather
than an independently predicted confidence. Velocity and attribute logits
are not exposed. Suppression is per-class rotated bird's-eye-view IoU,
with default threshold 0.8, followed by a global `max_det` cap. Returned rows
are sorted by descending score. CPU polygon intersection uses OpenCV.

## Checkpoints and provenance

The constructor requires the unchanged official R101 nuScenes checkpoint
with a `state_dict` wrapper. It validates the class order when supplied and
loads all tensors strictly. There is no automatic download or converted
LibreYOLO checkpoint for this family yet.

Implementation sources are Apache-2.0, pinned in the family NOTICE:
MMDetection3D `fe25f7a51d36e3702f961e198894580d83c4387b` and MMDetection
`cfd5d3a985b0249de009b67d04f37263e11cdf3d`. No C++ or CUDA source is vendored.

The official model is listed in the
[FCOS3D model zoo](https://github.com/open-mmlab/mmdetection3d/tree/fe25f7a51d36e3702f961e198894580d83c4387b/configs/fcos3d).
The code license does not establish a separate checkpoint redistribution
license. nuScenes also has
[non-commercial dataset terms](https://www.nuscenes.org/terms-of-use).
Checkpoint rehosting remains unresolved; no Hugging Face upload is made.
No dataset images, annotations, checkpoint bytes, or validation artifacts
are distributed with this implementation.

## Validation and limits

The official finetuned checkpoint (published July 2021) loads with every
state-dict key and shape matching. Native backbone, FPN and all five head
output groups match the pinned upstream Python implementation exactly on
seeded 128x256 and 160x224 inputs. That comparison uses MMCV-lite Python
layers and torchvision deformable convolution on both sides. It verifies
the inference graph, not numerical parity with MMCV's compiled CUDA kernel.

A real 1600x900 nuScenes front-camera sample with its provided calibration
runs on CPU and produces a projected cuboid overlay. This is a smoke and
rendering check, not an accuracy benchmark. Unit tests cover preprocessing,
deformable-convolution baseline behaviour, camera decoding, axis mapping,
NMS class isolation, empty outputs, slicing, multi-input calls and the CLI.

Training, generic validation, export and tracking raise explicit unsupported
errors. There is no nuScenes mAP claim, no CUDA validation, and no generic
GUI registration because the GUI does not supply calibration.
