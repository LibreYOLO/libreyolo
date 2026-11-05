# ADR 0031: Image-To-3D Generation In LibreGen

- Status: Proposed
- Date: 2026-10-30
- Scope: New canonical task (`generate3d`), new result payload
  (`Results.gaussians`), image input for `LibreGen` families, first family
  Speridlabs Idol. Extends ADR 0030, which kept 3D generation out of scope.

## Context

Open image-to-3D models take one picture of an object and return the whole
object, including the sides the picture never showed. Each one ships its own
setup: a repo to clone, gated encoders to request, a CLI with a dozen flags,
an output written somewhere next to three side files. None of that is the
model; it is the ceremony around it.

No existing contract fits. `mesh` (ADR 0013) recovers a body that is in the
image, in camera metres, aligned with person boxes. `generate` (ADR 0030)
returns an image. A generated 3D object has no source canvas, no boxes and no
validation metric that `val()` could report today.

The de-facto YOLO API has no 3D generation surface, so there is no name to
mirror. The contract below reuses what users already know from it
(`model(source)`, `Results`, `save`, `stream`, `device`, `compile`) and from
ADR 0030 (`seed`, `n`, family knobs), and adds the minimum on top.

## Decision

Image-to-3D models are `LibreGen` families. The task `generate3d` names their
output, `Results.gaussians` carries it, and the first positional argument of
a call is the family's input: a prompt for text-to-image families, an image
for image-to-3D families.

```python
from libreyolo import LibreGen

model = LibreGen("speridlabs-idol")
result = model("chair.jpg")
result.save("chair.ply")
```

That is the whole setup. No repo, no encoder wiring, no output folder
conventions.

## Task contract: `generate3d`

`generate3d` is the canonical task for models whose public output is a
synthesized 3D object. Filename suffix `-generate3d`, following `detect` and
`detect3d`. Aliases `generation3d`, `3d-generation`, `image-to-3d`,
`image23d`, `img2-3d`, `img-to-3d` and `i23d` resolve to it.

The output is one object per result, in the object's own frame: z up,
centred in the cube `[-0.5, 0.5]^3`. Where the input view sits in that frame
is the payload's `camera`. A family whose model works in another frame
converts to this one. Gaussian splats are
the first representation. A later family that returns a mesh adds a mesh
payload under the same task; it does not change this one.

`Results.gaussians` holds a `Gaussians` payload, all arrays on one device:

- `xyz` `(N, 3)` float32 positions in the object frame.
- `scales` `(N, 3)` float32 standard deviations along the local axes, linear
  (not log).
- `rotations` `(N, 4)` float32 unit quaternions, scalar first `(w, x, y, z)`,
  rotating local axes into the object frame.
- `opacities` `(N,)` float32 in `[0, 1]`.
- `sh` `(N, K, 3)` float32 spherical-harmonic colour coefficients, degree 0
  first (`K = 1` for view-independent colour). `colors` returns the `(N, 3)`
  RGB in `[0, 1]` the degree-0 term encodes.
- `camera` is the input view: the shot the model inferred (`yaw` and `pitch`
  in radians, `radius` in object units, vertical `fov` in degrees) and the
  `margin` of the crop it saw (crop side over object size). The predicted
  shot describes the picture before the family's crop, so a view of the
  input frames the object by `margin`, not by `radius` and `fov`.
- `seed` reproduces the object with the same model and settings; `params`
  records the rest of the call (`model` and every family knob in effect).
- `input_image` is the RGB image the model saw after its own preprocessing
  (background removal, crop), so a user can tell a bad crop from a bad model.

`len(gaussians)` is `N`. `save(path)` writes by extension:

- `.ply`: the standard 3D Gaussian Splatting layout (`x y z nx ny nz f_dc_*
  opacity scale_* rot_*`, with log scales and logit opacities) that every
  splat viewer and trainer reads, turned to the y-up convention those tools
  expect. A `comment libreyolo {...}` header line stores seed, camera and
  params as JSON; readers ignore comments. A payload with higher-order
  colour (`K > 1`) is refused: turning it y-up needs its view-dependent
  terms rotated too, which a family that emits them adds with its own
  conversion.
- `.splat`: the compact 32-bytes-per-Gaussian web format, sorted by size and
  opacity, colour and rotation quantized to 8 bits.

On `Results`: `boxes` is `None`; `path`, `orig_shape` and `orig_img` describe
the source image; `len(result)` is 1; `summary()` returns one row with the
payload's record; `result.save(path)` writes the object; `result.plot()`
returns a preview of the object from the inferred input view, framed like
the input crop: a point render for a quick look rather than a Gaussian
rasterization.

Box-specific paths (tracking, tiling, export backends, mAP validation) do not
apply.

## Call contract

`generate(source, *, seed=None, n=1, stream=False, save=False,
output_path=None, **knobs)`; `model(...)` and `model.predict(...)` are the
same call.

- `source` is what YOLO predict takes: a path, URL, `PIL.Image`, NumPy array
  (BGR by default, like predict), tensor, a directory of images, or a list of
  any of these. A file or `PIL.Image` with a real alpha channel (RGBA,
  grayscale with alpha, or a palette with transparency) is used as the
  object's cutout; otherwise the family removes the background itself.
  `source=` may also be passed by keyword.
- `prompt=` is rejected: an image-to-3D family has no text input.
- `imgsz` is rejected: the input resolution is fixed by the model, and the
  amount of detail in the output is a family knob (`gaussians` for splats).
- `seed`, `n`, `stream`, `save` follow ADR 0030. Object `i` of a source uses
  `seed + i`. `save=True` writes `.ply` files to a new
  `runs/generate/predict*` directory per call, named after the source;
  `output_path` names the file for one source with `n=1`, and its extension
  picks the format.

One source with `n=1` returns a `Results`; several sources or `n > 1` return
a flat list in source order.

### Seeds and devices

ADR 0030 seeds a CPU generator so a text-to-image seed gives the same starting
noise on every device. A family that ports a reference implementation may
instead draw from a generator on the model's device, seeded the way the
reference seeds it, so that a seed reproduces the reference's object. That
choice is recorded in the family docstring. Idol makes it: on CUDA, a seed
gives the object the upstream SDK gives for the same seed and settings.

The port keeps the SDK's computation op for op, including one source of
run-to-run variation the SDK has too. On an H100 with PyTorch 2.15, PyTorch's
attention dispatch picks the cuDNN kernel for the octree decoder's first
step (one query cell over the 8192 latent tokens), and that kernel's result
depends on the GPU's memory history: 2 distinct outputs, 1.6e-4 apart, in 40
repeats, against 1 for the flash, memory-efficient and math kernels. For an
object whose point counts sit on a sampling boundary this moves some
Gaussians; it was observed in both the SDK and the port, in 3 of 143
port-versus-SDK object comparisons. Excluding the cuDNN kernel there would make a seed fully
repeatable at the cost of matching the SDK bit for bit; the port keeps the
SDK's behaviour.

## Families

| Alias | Upstream | License | Knobs (defaults) |
| --- | --- | --- | --- |
| `speridlabs-idol` | `speridlabs/idol` (code `speridlabs/idol-sdk`) | Apache-2.0 code and weights | `steps` (50), `guidance` (3.0), `shift` (3.0), `gaussians` (262144), `mode` (`"synthetic"`), `margin` (`None`) |
| `trellis` | `LibreYOLO/LibreTRELLISl`, a mirror of `microsoft/TRELLIS-image-large` (code `microsoft/TRELLIS`) | MIT code and weights | `steps` (25), `guidance` (5.0), `shift` (3.0), `margin` (1.2) |
| `triposplat` | `LibreYOLO/LibreTripoSplat0.6b`, a mirror of `VAST-AI/TripoSplat` (code `VAST-AI-Research/TripoSplat`) | MIT code and weights | `steps` (20), `guidance` (3.0), `shift` (3.0), `gaussians` (262144), `margin` (1.2) |

Speridlabs Idol is a 0.66B rectified-flow transformer: DINOv3 ViT-H+ and
FLUX.2 VAE tokens of the 1024px input condition a denoiser over an 8192-token
object latent plus one camera token; an octree decoder places anchors and a
Gaussian decoder turns each anchor into 32 Gaussians.

Knobs, shared names first: `steps` sampling steps, `guidance` classifier-free
guidance, `shift` the flow time shift toward high noise (diffusers' name for
it), `gaussians` the output count (32,768 to 1,048,576, rounded down to a
multiple of 32, the Gaussians per anchor),
`mode` `"synthetic"` for renders and generated images or `"real"` for photos
(how much room the crop leaves around the object), `margin` an explicit crop
margin that overrides `mode`.

Constructor: `device` (`"auto"` picks CUDA, then MPS, then CPU) and
`compile` (CUDA, default `False`), Ultralytics' name for `torch.compile`.
`compile=True` is the upstream SDK's `optimize` path: compiled denoiser
blocks, the guided and unguided passes in one batch, cuDNN attention and TF32
matmuls while sampling. On an H100 it takes an object from 9.8 s to 6.4 s
(defaults, after a first call that compiles). It reproduces the SDK's
compiled path bit for bit; against the eager path it is the same object,
not the same Gaussians, because TF32 and fused kernels move the sampled
latent slightly and the octree places its points from it.
Precision is the published mixed precision (fp16 transformer blocks, bf16
encoders), so `dtype` accepts only `"auto"`.

TripoSplat is the model Idol builds on: the same 0.66B architecture, the
same encoders, the same octree and Gaussian decoders, up to 262,144
Gaussians. Its knobs are upstream's: `steps` (upstream recommends 10 to
20), `guidance` with upstream's diffusers convention (1 or less runs the
conditional pass only; above 1 blends `s * v(cond) - (s - 1) * v(uncond)`,
where Idol's `w` equals `s - 1`), `shift`, `gaussians` (32,768 to 262,144)
and `margin` (upstream's fixed 1.2 crop, exposed). No `mode`, no `compile`.
The networks run in fp16 on CUDA and MPS as upstream loads them, and in
fp32 on CPU. Upstream draws the octree sampling from the global generator,
so its decoder stage is not seed-reproducible; the port draws it from the
seeded generator, so a seed reproduces the whole object. The sampled latent
and camera follow upstream for a seed; the Gaussians are the same
distribution, not the same draw.

The shared steps (background removal, crop, the two encoders, the camera
decoding) live in `libreyolo/models/gen/image3d.py`; each family keeps its
own networks and sampler. TripoSplat upstream ships its own DINOv3 ViT and
FLUX.2 VAE encoder implementations; the port reads the same weights (VAST's
DINOv3 file equals LibreYOLO's mirror tensor for tensor) through
transformers and diffusers, as Idol does. The VAE tokens are bitwise equal;
the DINOv3 tokens differ at bf16 noise level (mean absolute difference 0.006
after the layer norm, isolated tokens up to 1.0), because the two
implementations order their bf16 arithmetic differently. The parity gate
(`weights/parity_triposplat.py`) checks the encoders at that tolerance and
the rest of the pipeline from upstream's own tokens.

TRELLIS (Microsoft, `TRELLIS-image-large`) has a different architecture: a
dense flow transformer denoises a 16^3 occupancy latent, a 3D CNN decodes it
to the active voxels of a 64^3 grid, a sparse flow transformer denoises an
8-channel structured latent on those voxels, and a shifted-window sparse
transformer turns each voxel into 32 Gaussians. Both flow stages read DINOv2
ViT-L/14 tokens of the 518px input. Upstream runs the sparse stages on
spconv and xformers or flash-attn, CUDA-only extensions; the port replaces
them with PyTorch (`libreyolo/models/gen/trellis/sparse.py`): the
submanifold convolution as a gathered GEMM over the 27 neighbours (equal to
spconv in fp32 to 2.4e-7, with spconv's weight layout unchanged), the
variable-length attention as SDPA per batch element and over padded windows
with a key mask. So the family runs on CUDA, MPS and CPU. Knobs follow
upstream's samplers, shared by the two stages: `steps`, `guidance` (applied
inside upstream's guidance interval t in [0.5, 1]), `shift`, plus `margin`.
The Gaussian count is 32 per occupied voxel, not a knob. TRELLIS infers no
camera, so `camera` holds only `margin`. Noise is drawn on the CPU as
upstream does, so a seed gives the same starting noise on every device. The
torso runs in fp16 on CUDA and MPS as upstream, and in fp32 on CPU. Two
deliberate differences: background removal uses LibreYOLO's BiRefNet
instead of rembg's U2Net, and DINOv2 is read through transformers
(`facebook/dinov2-with-registers-large`) instead of `torch.hub`, the
pre-norm last-layer tokens as upstream takes them.

## Weights

Nothing is gated and no login is needed. Idol's own weights (denoiser,
decoders, BiRefNet) are the pinned snapshot of `speridlabs/idol`, downloaded
unchanged into `weights/LibreSperidlabsIdol0.6b/`; they are deliberately not
mirrored, so the downloads count for Speridlabs. Its two image encoders load
at pinned revisions through the Hugging Face cache:

- DINOv3 ViT-H+/16 from LibreYOLO's mirror `LibreYOLO/LibreDINOv3vith16plus`
  (revision `e849a3492905c6dabc89758042bf1402770fb4e6`), under Meta's DINOv3
  License, which permits redistribution with a copy of the agreement. Meta's
  own repo, `facebook/dinov3-vith16plus-pretrain-lvd1689m`, is gated behind a
  manually reviewed form. The mirror is that checkpoint cast to bf16, the
  precision Idol runs it in: all 615 tensors equal the upstream ones loaded
  in bf16, so outputs do not change, the download halves (1.7 GB) and users
  meet no gate.
- The FLUX.2 VAE from `black-forest-labs/FLUX.2-klein-4B` (Apache-2.0, not
  gated). Upstream reads it from the gated `FLUX.2-dev` repo; the two repos'
  VAE checkpoints are identical tensor for tensor once cast to the bf16 the
  model runs in, so the output does not change and users skip one gate. The
  file is the one `LibreGen("flux2-klein-4b")` already uses.

The background remover is LibreYOLO's own BiRefNet network
(`libreyolo/models/birefnet`), loading the BiRefNet weights from the Idol
snapshot.

`LibreGen(path)` loads a local directory holding the snapshot files
(`config.json` with `image_encoders`, `denoiser.safetensors`,
`decoder.safetensors`, `decoder_gs.safetensors`, `birefnet.safetensors`).

TripoSplat's own weights (denoiser, decoder, BiRefNet) are redistributed
under MIT as `LibreYOLO/LibreTripoSplat0.6b`, byte-identical to the pinned
`VAST-AI/TripoSplat` snapshot and in its folder layout, downloaded into
`weights/LibreTripoSplat0.6b/`; upstream's DINOv3 and FLUX.2 VAE copies are
not mirrored, the two encoders above are used instead. `LibreGen(path)` loads a directory holding those three files.

TRELLIS's weights are redistributed under MIT as `LibreYOLO/LibreTRELLISl`:
upstream's `pipeline.json` and network files byte-identical to
`microsoft/TRELLIS-image-large` (the sparse-structure flow and decoder, the
structured-latent flow, the Gaussian and mesh decoders), plus the BiRefNet
weights. The library downloads only the files it loads into
`weights/LibreTRELLISl/`. DINOv2 loads from
`facebook/dinov2-with-registers-large` (Apache-2.0) at a pinned revision.

## Reserved for later

As in ADR 0030, each raises `NotImplementedError` with its reason:
`train(data=...)` (fine-tuning on image and object pairs), `val(data=...)`
(rendered-view metrics on a held-out set), `prompt=` on image-to-3D families
(text-to-3D), and `export()`.

## Out of scope

- Meshing, texturing and rendering beyond the preview. Viewers and trainers
  that read the standard PLY do that.
- Scenes. One result is one object.

## Consequences

- One call shape covers text-to-image and image-to-3D: `LibreGen(name)(input)`.
- A YOLO user already knows how to feed it: the source rules are predict's.
- Adding the next image-to-3D family is a `LibreGen3DModel` subclass with
  `_load` and `_generate_3d`, its knobs and its frame conversion.
