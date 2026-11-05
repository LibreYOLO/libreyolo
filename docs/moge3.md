# MoGe-3

`LibreMoGe3` ports Microsoft's MoGe-3 monocular geometry model: one RGB image
to metric depth, a camera-space point map, camera intrinsics, a validity mask,
and surface normals. It is MoGe-2 plus a sparse 3D UNet that voxelizes the
predicted log-depth at 1/256 resolution and refines it for three steps.

| Weight file | Backbone | Task |
| --- | --- | --- |
| `LibreMoGe3l-depth.pt` | DINOv2 ViT-L/14 | depth, point map, intrinsics, mask |
| `LibreMoGe3l-normal.pt` | DINOv2 ViT-L/14 | normals |
| `LibreMoGe3g-depth.pt` | DINOv2 ViT-g/14 | depth, point map, intrinsics, mask |
| `LibreMoGe3g-normal.pt` | DINOv2 ViT-g/14 | normals |

The `-depth` and `-normal` files of one size hold the same tensors; either
loads with either `task=`.

## Usage

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreMoGe3l-depth.pt")
depth = model.predict("image.jpg").depth_map
depth.data          # (H, W) metres, inf outside the valid mask
depth.point_map     # (H, W, 3) camera-frame points, metres, OpenCV axes
depth.intrinsics    # (3, 3) OpenCV pixel intrinsics, pixel centers at integers
depth.valid_mask    # (H, W) bool

normals = LibreYOLO("LibreMoGe3l-normal.pt").predict("image.jpg").normals
```

`imgsz` sets the ViT token budget, `(imgsz // 14) ** 2` tokens laid out to the
image aspect ratio. The default 840 is upstream's highest resolution level
(3600 tokens); 490 (1225 tokens) is close to its lowest (1200). Inputs keep their native
resolution and every output is interpolated back to it. Inference runs in
FP32 and always applies three refinement steps, as upstream does by default.

## Devices

The sparse refiner is plain PyTorch (sort, searchsorted, unique, gather,
index_add, matmul), so CPU, CUDA, and MPS work without a compiled extension.
Camera recovery solves a small least-squares problem with SciPy on the CPU.
On MPS the 4x4 camera matrix inverse runs on the CPU.

## Parity

Measured on an RTX 3090 in FP32 against upstream at
`74fbce054ebed49800de42d0ad0e83495065719a` with
`weights/parity_moge3.py`, 1536x1194 input, 3600 tokens:

| Check | `l` | `g` |
| --- | --- | --- |
| forward without refinement (points, mask, scale), max abs diff | 0 | 0 |
| normal head, max abs diff | 0 | 0 |
| one refiner step, max abs log-depth diff | 8.9e-7 | 7.7e-7 |
| final depth, p99 relative diff | 4.7e-6 | 3.7e-5 |
| final depth, max relative diff | 2.3e-4 | 5.8e-3 |
| intrinsics (normalized), max abs diff | 7.9e-8 | 9.6e-8 |
| validity mask disagreement | 0 | 0 |

The refiner residual comes from accumulation order: upstream runs FlexGEMM
GPU kernels, LibreYOLO runs gather and matmul. A log-depth difference near a
1/256 bin edge can move a voxel to the neighbouring bin, which is why the
maximum is larger than the p99. Upstream comparisons use FlexGEMM with TF32
off; with its default TF32 sparse GEMMs on, upstream itself drifts further
(`l`: final depth p99 relative diff 3.0e-3).

## Scope

- Inference, zero-shot `val` for both tasks, and ONNX and TorchScript export
  of the normal graph (fixed square canvas). Depth export is blocked: the
  refiner builds data-dependent voxel sets and camera recovery is a CPU
  solve.
- Not trainable in LibreYOLO. The upstream recipe is multi-dataset
  pretraining with camera-calibrated point-map labels, which LibreYOLO's
  depth dataset schema does not carry.
- Code and weights are MIT (Microsoft). The DINOv2 backbone is Apache-2.0.
