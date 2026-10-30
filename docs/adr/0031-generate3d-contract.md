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

## Families

| Alias | Upstream | License | Knobs (defaults) |
| --- | --- | --- | --- |
| `speridlabs-idol` | `speridlabs/idol` (code `speridlabs/idol-sdk`) | Apache-2.0 code and weights | `steps` (50), `guidance` (3.0), `shift` (3.0), `gaussians` (262144), `mode` (`"synthetic"`), `margin` (`None`) |

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

## Weights

Base weights follow ADR 0030: nothing is converted or redistributed. Idol's
own weights (denoiser, decoders, BiRefNet) are the pinned snapshot of
`speridlabs/idol`, downloaded into `weights/LibreSperidlabsIdol0.6b/`. Its two
image encoders load from their publishers at pinned revisions through the
Hugging Face cache:

- DINOv3 ViT-H+/16 from `facebook/dinov3-vith16plus-pretrain-lvd1689m`, gated
  under Meta's DINOv3 License. The loader checks access before downloading
  anything and says exactly which page to accept and how to log in.
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
