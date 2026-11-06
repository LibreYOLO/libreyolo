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

### Meshes

A family whose output is a triangle mesh returns it in `Results.meshes`, the
existing `Meshes` payload of ADR 0013, with no person boxes, instead of
adding a payload type:

- one row: `len(meshes)` is 1, `body_model` is `"object"`;
- `global_orient`, `body_pose` and `betas` are empty `(1, 0)` and `transl`
  is zero `(1, 3)`: a generated object has no body parameters;
- `vertices` `(1, V, 3)` float32 in the object frame above (z up, inside
  `[-0.5, 0.5]^3`), `faces` `(F, 3)` int64 wound counter-clockwise seen from
  outside;
- `extras["vertex_colors"]` `(1, V, 3)` float32 RGB in `[0, 1]`, when the
  family colours its vertices.

`result.save(path)` writes Wavefront `.obj`, with vertex colours as the
common `v x y z r g b` extension; `save=True` writes `.obj` files. A mesh
family declares `SAVE_SUFFIXES = (".obj",)`; `LibreGen3DModel` picks the
`Results` slot from the payload type. `summary()` returns one row with the
vertex and face counts. `plot()` has no mesh preview yet. `Meshes` has no
`seed`, `params`, `camera` or `input_image`; a mesh family that samples, and
so needs `seed` recorded on the result, adds that to `Meshes` rather than a
second payload.

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
| `trellis2` | `LibreYOLO/LibreTRELLIS2-4b`, a mirror of the geometry files of `microsoft/TRELLIS.2-4B` (code `microsoft/TRELLIS.2`) | MIT code and weights; DINOv3 encoder under Meta's DINOv3 License | `steps` (12), `guidance` (7.5), `resolution` (1024), `margin` (1.0), `max_tokens` (49152) |
| `triposr` | `LibreYOLO/LibreTripoSR`, a mirror of `stabilityai/TripoSR` (code `VAST-AI-Research/TripoSR`) | MIT code and weights | `resolution` (256), `threshold` (25.0), `margin` (1/0.85) |
| `triposplat` | `LibreYOLO/LibreTripoSplat0.6b`, a mirror of `VAST-AI/TripoSplat` (code `VAST-AI-Research/TripoSplat`) | MIT code and weights | `steps` (20), `guidance` (3.0), `shift` (3.0), `gaussians` (262144), `margin` (1.2) |
| `lgm` | `LibreYOLO/LibreLGM0.4b`, a mirror of `ashawkey/LGM`, and `LibreYOLO/LibreImageDream`, a mirror of `ashawkey/imagedream-ipmv-diffusers` (code `3DTopia/LGM`) | MIT code; MIT LGM weights; OpenRAIL ImageDream weights | `steps` (30), `guidance` (5.0), `elevation` (0) |
| `crm` | `LibreYOLO/LibreCRM`, a mirror of `Zhengyi/CRM` (code `thu-ml/CRM`) | MIT code; MIT weights carrying SD 2.1's OpenRAIL++-M use restrictions | `steps` (50), `guidance` (5.0), `texture` (`False`) |

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

LGM (Tang et al., ECCV 2024) is two-stage. ImageDream, a multi-view
diffusion UNet fine-tuned from Stable Diffusion 2.1 with an image-prompt
branch over CLIP ViT-H/14 tokens, samples four 256 px views at azimuths 0,
90, 180 and 270 with DDIM; LGM, an asymmetric UNet with cross-view attention,
reads the four views with their Plücker ray embeddings and returns one
Gaussian per pixel of four 128 x 128 maps (65,536). The port keeps the
Gaussians at opacity 0.005 and above, as upstream's saved files do, and moves
them from LGM's frame (y up, `[-1, 1]^3`) into the object frame. Upstream
runs every attention through xformers (CUDA only) and renders through
diff-gaussian-rasterization (Inria, non-commercial); the port uses SDPA and
does not render. Knobs: `steps` and `guidance` of the ImageDream sampler
(`guidance` above 1 enables classifier-free guidance; upstream fails at 1,
the port then runs the conditional pass only) and `elevation`, upstream's
input-camera elevation in degrees, which also sets `camera` (`yaw` pi,
`pitch` minus the elevation, `radius` 0.75, `fov` 49.1, `margin` 1.25 from
the 80 % recentring). LGM infers no camera; the camera is the one it
assumes. Noise is drawn on the model's device from a generator seeded with
the seed, in upstream's order (the VAE posterior sample of the input, then
the starting latents), so on CUDA a seed reproduces upstream's demo seeded
with `kiui.seed_everything`. The networks run in fp16 on CUDA and MPS (LGM
under autocast, as upstream), fp32 on CPU. Background removal uses
BiRefNet instead of rembg; a real alpha channel is used as is, where
upstream always re-cuts the image. The empty prompt's CLIP text tokens are
computed once at load and the text encoder is released.

CRM (Tsinghua, ECCV 2024) returns a mesh. An ImageDream-style multi-view
diffusion model (SD 2.1's UNet with cross-view self-attention, camera
embeddings and a 16-token OpenCLIP image prompt) samples six orthographic
views; a second one samples their canonical coordinate maps; a diffusers
UNet2DModel turns views and maps into a 32-channel triplane; MLPs read the
SDF, vertex offsets and FlexiCubes weights on an 80^3 grid; FlexiCubes
extracts the surface and a colour MLP colours it. The result goes in
`Results.meshes` as one `Meshes` row with `body_model="object"`, empty body
parameters, `vertices` z-up in `[-0.5, 0.5]^3` (CRM's `[-1, 1]` frame halved),
faces wound counter-clockwise seen from outside and
`extras["vertex_colors"]`. Because CRM samples, `Meshes` gains `seed` and
`generation_params` (the model and knobs), which `summary()` reports.
`texture=True` adds `extras["uv"]`, `extras["uv_faces"]` and
`extras["texture"]` (1024 x 1024 uint8, row 0 at v = 0); `.obj` then comes
with a `.mtl` and a `.png`. Upstream's CUDA-only and non-permissive pieces are
replaced: SDPA for xformers, NVIDIA's Apache-2.0 FlexiCubes for CRM's
proprietary-header copy, a PyTorch UV rasterizer for nvdiffrast, a NumPy
cleanup (same operations) for kiui/pymeshlab, LibreYOLO's BiRefNet for rembg.
The diffusion models run in fp16 on CUDA and MPS as upstream and in fp32 on
CPU. Every random draw comes from one CPU generator. The parity gate
(`weights/parity_crm.py`) feeds both sides the same image and noise: the
views and CCMs differ by fp16 kernel noise (mean 0.05 and 0.11 of 255), the
triplane features, the FlexiCubes grid and the raw mesh are equal, the
colours bitwise; the cleanup merges vertices in a different order than
upstream's library (Chamfer distance 0.001 in CRM units).

TRELLIS.2 (Microsoft, `TRELLIS.2-4B`) returns a mesh in `Results.meshes`.
A dense flow transformer with 3D rotary positions denoises a 16^3 occupancy
latent, TRELLIS's 3D CNN decodes it to the occupied voxels of a 32^3 grid,
a sparse flow transformer denoises a 32-channel shape latent on them at 512,
and for `resolution` 1024 or 1536 a second one denoises it again on the
voxels the first latent's decoder predicts (fewer when they exceed
`max_tokens`, as upstream). A sparse ConvNeXt decoder upsamples the latent
16 times into a flexible dual grid, one vertex per voxel and a quad around
every crossed voxel edge. Both flow stages read DINOv3 ViT-L/16 tokens (512px,
and 1024px for the second stage). Upstream's CUDA-only extensions (FlexGEMM
sparse convolution, flash-attn, o-voxel's hash map) are replaced by the
TRELLIS family's PyTorch operations, with neighbours found by binary search
in sorted voxel keys on grids too large for a dense table; so the family
runs on CUDA, MPS and CPU. `steps` and `guidance` apply to every sampling
stage; each keeps upstream's guidance interval, guidance rescale and time
shift. Noise is drawn on the CPU as upstream does. The networks run in their
published precision (bf16 transformers, fp16 decoders) on CUDA and MPS and in
fp32 on CPU. The mesh is the dual grid's output before upstream's CuMesh
post-processing: it keeps one vertex per voxel, some unreferenced, and its
faces are not consistently wound (the dual grid carries no inside or
outside), so the `Meshes` winding rule above does not hold for this family.
`Meshes.seed` and `Meshes.generation_params` record the seed and the call
(model and knobs), as for CRM.
Upstream's texture stage is not ported: it bakes PBR voxels into a UV texture
with nvdiffrast, a non-commercial NVIDIA library. Background removal uses
LibreYOLO's BiRefNet instead of upstream's RMBG-2.0 (non-commercial).
The parity gate (`weights/parity_trellis2.py`, end to end
`weights/parity_trellis2_e2e.py`) runs each stage from upstream's inputs:
the preprocessing, the DINOv3 tokens, the structure stage with upstream's
SDPA attention backend, the structure decoder and the meshing of upstream's
decoder fields are bitwise equal; the stages that run on flash-attn and
FlexGEMM upstream differ by kernel noise no larger than upstream's own
distance to an fp32 run of the same weights.

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

LGM's own network (`model_fp16_fixrot.safetensors`, upstream's checkpoint
after its rotation-normalisation fix) and the BiRefNet weights are
redistributed under MIT as `LibreYOLO/LibreLGM0.4b`, the LGM file
byte-identical to the pinned `ashawkey/LGM`, downloaded into
`weights/LibreLGM0.4b/`. ImageDream is redistributed as
`LibreYOLO/LibreImageDream`, byte-identical to the pinned
`ashawkey/imagedream-ipmv-diffusers` weights and configs (without upstream's
remote-code `.py` files), loaded through the Hugging Face cache. Upstream
declares it `openrail`; it descends from Stable Diffusion 2.1, so the mirror
ships the CreativeML OpenRAIL++-M License and its use restrictions.
`LibreGen(path)` loads a directory holding `model_fp16_fixrot.safetensors`,
optionally with `birefnet.safetensors` and an `imagedream/` copy.

TRELLIS.2's geometry networks (the sparse-structure flow, the two shape
flows, the shape decoder) and `pipeline.json` are redistributed under MIT as
`LibreYOLO/LibreTRELLIS2-4b`, byte-identical to `microsoft/TRELLIS.2-4B` in
upstream's layout, downloaded into `weights/LibreTRELLIS2-4b/`; the texture
networks are not mirrored. The sparse-structure decoder TRELLIS.2 reuses and
the BiRefNet weights come from `LibreYOLO/LibreTRELLISl`. DINOv3 ViT-L/16
comes from LibreYOLO's mirror `LibreYOLO/LibreDINOv3vitl16`, byte-identical
(fp32) to Meta's gated `facebook/dinov3-vitl16-pretrain-lvd1689m`, with the
agreement. `LibreGen(path)` loads a local copy of upstream's TRELLIS.2-4B
layout.

## Reserved for later

As in ADR 0030, each raises `NotImplementedError` with its reason:
`train(data=...)` (fine-tuning on image and object pairs), `val(data=...)`
(rendered-view metrics on a held-out set), `prompt=` on image-to-3D families
(text-to-3D), and `export()`.

## Out of scope

- Meshing Gaussians, texturing and rendering beyond the preview. Viewers and
  trainers that read the standard PLY do that. Families whose network
  outputs a surface return it as a mesh (see Meshes above).
- Scenes. One result is one object.

## Consequences

- One call shape covers text-to-image and image-to-3D: `LibreGen(name)(input)`.
- A YOLO user already knows how to feed it: the source rules are predict's.
- Adding the next image-to-3D family is a `LibreGen3DModel` subclass with
  `_load` and `_generate_3d`, its knobs and its frame conversion.
