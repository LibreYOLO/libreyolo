# ADR 0027: Marigold V2 integration

Status: implemented and CUDA parity validated
Date: 2026-09-10

## Decision

Add checkpoint-driven family `marigold_v2`, class `LibreMarigoldV2`, size `b`,
to the generic `LibreYOLO` factory. It implements depth, surface normals and
albedo with the upstream one-step Qwen diffusion graph. The transformer and
VAE come from the optional Apache-2.0 Diffusers dependency; the small graph,
loading and preprocessing code is adapted into LibreYOLO.

Code source: `huawei-bayerlab/marigold-v2` revision
`cc6a7031abcd59fd9e1ceff7fdd0d9687d389bc5`, Apache-2.0.
Adapter source: `huawei-bayerlab/marigold-v2-0` revision
`6fd6d1ca246c9d2d99a4d8ac375a4eccc87178ad`, Apache-2.0.
Frozen base: `Qwen/Qwen-Image-Edit-2509` revision
`d3968ef930e841f4c73640fb8afa3b306a78167e`, Apache-2.0 model-card declaration.

## Checkpoints and runtime

Canonical checkpoints store the unchanged inference adapter tensors, optional
VAE decoder tensors, fixed prompt embeddings and a stable variant marker.
They do not duplicate the 20B frozen base. Two named training-only iREPA
projection tensors are omitted and recorded in conversion metadata. The
loader requires every LoRA tensor and, when present, the complete VAE decoder;
unexpected or missing inference tensors fail loading.

Default files are `LibreMarigoldV2b-depth.pt` (Log-stage2),
`LibreMarigoldV2b-normal.pt` and `LibreMarigoldV2b-albedo.pt`. Other depth files
use `LibreMarigoldV2b-depth-<variant>.pt`, with variants `log-stage1`,
`log-layered`, `uniform-base`, `uniform-layered`, `disparity-base` and
`disparity-layered`. Variant, task and base revision are checked before loading.

Install `libreyolo[marigold]`. The tested upstream stack uses Diffusers 0.38.0,
PEFT 0.18.1, Accelerate 1.13.0, Transformers 5.4.0 and bitsandbytes 0.49.2.
Default inference uses NF4 on CUDA with BF16 computation. It preserves the
upstream skipped quantization module, dequantized output projection, LoRA rank
128 and BF16 autocast context. Unquantized CPU loading is an explicit option
and needs memory for the full base. MPS is unsupported.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreMarigoldV2b-depth.pt", device="cuda")
result = model("photo.jpg")
print(result.depth_map.encoding)  # log_depth
result.plot().save("depth.png")
```

The existing Hub form also works, for example
`LibreYOLO("LibreYOLO/LibreMarigoldV2b-depth", device="cuda")`. It uses the
Hugging Face client cache and transfer backend, which is useful for these
multi-gigabyte checkpoints. Bare canonical filenames retain the standard
LibreYOLO downloader and its resume/checksum behavior.

```sh
libreyolo predict model=LibreMarigoldV2b-normal.pt source=photo.jpg device=cuda save=true
```

Prediction defaults to native dimensions rounded up to multiples of 16.
`imgsz=0` requests the same behavior explicitly. Positive `imgsz` selects a
fixed square canvas; Python also accepts `(height,width)`. Resize uses Lanczos
on normalized float RGB, matching upstream rather than resizing an 8-bit image.
Outputs return to original dimensions with OpenCV bilinear interpolation.

Depth encodings follow ADR 0025. Normals convert the upstream Hypersim/OpenGL
camera frame to OpenCV by negating y and z, then renormalize after resizing.
Albedo follows ADR 0026. Prediction uses seed 2025 by default, with an isolated
random-number context per image. Set `seed=` on direct `LibreMarigoldV2`
construction to select another seed. Shared validation uses a fixed canvas
and the same family input preprocessing.

## Scope

This integration supports pretrained inference and validation. Training,
tracking, tiling, TTA and export are not integrated. Upstream training code
exists, but this PR does not reproduce its data preparation, iREPA teacher,
losses, optimizer or convergence. Those require a separate training change.
No acknowledgement flags are required to use inference.

## Validation evidence

All nine adapters passed an exact tensor-name/shape check against the full
Qwen architecture before GPU execution. On an NVIDIA L40S with PyTorch
2.10.0+cu128, each adapter then matched the original upstream graph on two
images: an asymmetric native canvas and a fixed square canvas. All 18
original-canvas output comparisons had `max_abs_diff=0.0`, including the
specified normal-axis conversion and linear-albedo clipping.

The comparison constructs upstream and native models independently in the
same worker. Saved BF16 reference outputs from a different worker differed;
this evidence does not establish bitwise reproducibility across machines.
The committed manual e2e test therefore computes fresh upstream references.
No published benchmark accuracy, CPU runtime performance, MPS, training or
export claim is made.
