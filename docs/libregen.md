# LibreGen

`LibreGen` runs generative models behind the same call as every other
LibreYOLO task: text in and an image out, or an image in and a 3D object
out. The contracts live in `docs/adr/0030-libregen-contract.md` (images)
and `docs/adr/0031-generate3d-contract.md` (3D objects).

## Install

```bash
pip install "libreyolo[gen]"
```

The extra pulls the Apache-2.0 `diffusers` package. Importing `LibreGen`
never imports it; it loads on first use.

## Generate

```python
from libreyolo import LibreGen

model = LibreGen("flux2-klein-4b")                 # autodownloads
result = model("a red fox sleeping in fresh snow")
result.save("fox.png")
```

Arguments every model understands:

```python
results = model(
    ["a red fox", "a blue jay"],   # one prompt or a list
    imgsz=1024,                    # int, or (height, width)
    seed=0,                        # same seed, same image
    n=2,                           # images per prompt
    save=True,                     # writes runs/generate/predict/*.png
)
```

One prompt with `n=1` returns a `Results`; a list of prompts or `n > 1`
returns a flat list. `stream=True` returns a generator.

- A string is always a prompt, never a path.
- Image `i` of a prompt uses `seed + i`. Without a seed a fresh one is drawn
  and recorded, so any image can be made again.
- `imgsz` sides are rounded up to the model's size multiple with a warning.

## The result

```python
result.generated.array      # (H, W, 3) uint8 RGB
result.generated.image      # PIL.Image
result.generated.prompt     # the text
result.generated.seed       # the seed that reproduces it
result.generated.params     # model, imgsz and the knobs in effect
result.save("fox.png")      # a PNG also stores prompt, seed and params
result.summary()            # one JSON-ready row
```

## Family knobs

Settings that only some models have are per-family. Each family has
defaults, so they are optional. A knob the family does not have raises and
lists the supported ones.

```python
model("a red fox", steps=8, guidance=1.0)
```

| Alias | Parameters | Knobs (defaults) | Download | License |
| --- | --- | --- | --- | --- |
| `flux2-klein-4b` (default) | 4B | `steps` (4), `guidance` (1.0, fixed: the distilled checkpoint runs without guidance) | about 16 GB | Apache-2.0 |
| `sana-sprint-0.6b` | 0.6B | `steps` (2), `guidance` (4.5) | about 7.7 GB | Apache-2.0; Gemma 2 text encoder under the Gemma Terms of Use |
| `prx-1024-beta` | 1.3B | `steps` (28), `guidance` (5.0), `negative` ("") | about 21 GB | Apache-2.0; T5Gemma text encoder under the Gemma Terms of Use |
| `prx-512-sft-distilled` | 1.3B | `steps` (8), `guidance` (1.0), `negative` ("") | about 16 GB | as above |
| `prx-512-sft`, `prx-512` | 1.3B | `steps` (28), `guidance` (5.0), `negative` ("") | about 16 GB | as above |
| `prx-512-dc-ae-sft-distilled` | 1.3B | `steps` (8), `guidance` (1.0), `negative` ("") | about 16 GB | as above; DC-AE autoencoder MIT |
| `prx-512-dc-ae-sft`, `prx-512-dc-ae` | 1.3B | `steps` (28), `guidance` (5.0), `negative` ("") | about 16 GB | as above; DC-AE autoencoder MIT |
| `prx-256-sft`, `prx-256` | 1.3B | `steps` (28), `guidance` (5.0), `negative` ("") | about 16 GB | as above |
| `prxpixel-7b` | 7B | `steps` (28), `guidance` (5.0), `negative` ("") | about 18 GB | Apache-2.0 (Qwen3-VL text encoder, no autoencoder) |

`flux2-klein` and `sana-sprint` are short aliases; `prx` is `prx-1024-beta`
and `prxpixel` is `prxpixel-7b`.

The nine PRX checkpoints are Photoroom's whole PRX release. The number is
the native resolution; `-dc-ae` uses the 32x DC-AE autoencoder instead of the
FLUX.1 VAE, so its sides round up to a multiple of 32 instead of 16; `-sft`
is the supervised fine-tune, ready to use, and `-sft-distilled` the 8-step
distillation of it. Checkpoints without `-sft` are base models meant for
fine-tuning. `prx-1024-beta` is Photoroom's 1024 preview. PRXPixel denoises
pixels directly and needs `diffusers>=0.39`. All ten download from
byte-identical LibreYOLO mirrors of Photoroom's repos.

## Devices and memory

```python
model = LibreGen("sana-sprint-0.6b", device="mps")
model = LibreGen("flux2-klein-4b", offload=True)     # CUDA: idle components off the GPU
model = LibreGen("flux2-klein-4b", dtype="fp16")
```

`device="auto"` picks CUDA, then MPS, then CPU. `dtype="auto"` is bf16 on
CUDA and MPS and fp32 on CPU (and on MPS before macOS 14, which has no bf16).

## Local directories

```python
model = LibreGen("path/to/pipeline_dir")
```

A directory in a supported family's upstream format loads directly: an
offline copy, or a fine-tune made with upstream tooling.

## CLI

```bash
libreyolo generate prompt="a red fox" model=sana-sprint-0.6b seed=0 n=4
libreyolo generate prompt="a red fox" imgsz=768x1024
libreyolo generate prompt="a red fox" model=prx-512-sft negative="blurry"
libreyolo generate prompt_file=prompts.txt steps=8 --json
```

`prompt` is always literal text. `prompt_file` reads a text file with one
prompt per line; pass one or the other. `imgsz` is `1024` for a square image
or `HEIGHTxWIDTH` such as `768x1024`. Images are saved by default; pass
`--no-save` to skip.

## Image to 3D

One picture of an object in, the whole object out as 3D Gaussian splats,
including the sides the picture never showed:

```python
from libreyolo import LibreGen

model = LibreGen("speridlabs-idol")      # autodownloads
result = model("chair.jpg")              # Results
result.save("chair.ply")                 # opens in any Gaussian splat viewer
```

The source is anything YOLO predict takes: a path, URL, `PIL.Image`, NumPy
array (BGR), tensor, a folder of images or a list of them. A PNG or
`PIL.Image` with real transparency is used as the cutout; anything else gets
its background removed first.

```python
results = model(["chair.jpg", "lamp.png"], seed=0, n=2, save=True)
# writes runs/generate/predict/*.ply

model("photo.jpg", mode="real")          # photographs: a looser crop
model("render.png", gaussians=1_048_576) # more detail, from 32,768 to 1,048,576
model("chair.jpg", steps=25)             # faster, a little less detail
```

The result:

```python
g = result.gaussians
g.xyz, g.scales, g.rotations             # (N, 3), (N, 3), (N, 4) wxyz
g.opacities, g.sh, g.colors              # (N,), (N, 1, 3), (N, 3) RGB
g.camera                                 # the input view: yaw, pitch, radius, fov, margin
g.seed, g.params                         # how to make it again
g.input_image                            # what the model saw after its crop
g[g.opacities > 0.1]                     # select Gaussians
result.save("chair.splat")               # compact web format
result.plot()                            # a quick preview from the input view
```

The object is z-up and centred in `[-0.5, 0.5]^3`. The `.ply` is the
standard 3DGS layout turned y-up, the way viewers and trainers expect it,
and records seed, camera and settings in a header comment.

| Alias | Parameters | Knobs (defaults) | Download | License |
| --- | --- | --- | --- | --- |
| `speridlabs-idol` | 0.66B (plus DINOv3 ViT-H+ and the FLUX.2 VAE encoder) | `steps` (50), `guidance` (3.0), `shift` (3.0), `gaussians` (262144), `mode` (`"synthetic"` or `"real"`), `margin` (crop override) | about 3.1 GB, plus 1.7 GB DINOv3 and 0.2 GB VAE | Apache-2.0; the DINOv3 encoder under Meta's DINOv3 License |
| `trellis` | 1.3B in four networks (plus DINOv2 ViT-L/14) | `steps` (25), `guidance` (5.0), `shift` (3.0), `margin` (1.2) | about 2.7 GB, plus 1.2 GB DINOv2 | MIT; DINOv2 Apache-2.0 |
| `trellis2` | 4B in five networks (plus DINOv3 ViT-L/16) | `steps` (12), `guidance` (7.5), `resolution` (1024: 512, 1024 or 1536), `margin` (1.0), `max_tokens` (49152) | about 8.7 GB, plus 1.2 GB DINOv3 and 1.3 GB from the TRELLIS mirror | MIT; the DINOv3 encoder under Meta's DINOv3 License |
| `triposr` | 0.42B | `resolution` (256), `threshold` (25.0), `margin` (1/0.85) | about 1.7 GB, plus 0.4 GB BiRefNet | MIT |
| `direct3d-s2` | 2.0B in eight networks (plus DINOv2 ViT-L/14) | `resolution` (1024), `steps` ((50, 30, 15)), `guidance` (7.0) | about 4.2 GB, plus 1.2 GB DINOv2 and 0.4 GB BiRefNet | MIT; DINOv2 Apache-2.0 |
| `triposplat` | 0.66B (plus the same two encoders) | `steps` (20), `guidance` (3.0), `shift` (3.0), `gaussians` (262144), `margin` (1.2) | about 1.8 GB, plus the same DINOv3 and VAE | MIT; the DINOv3 encoder under Meta's DINOv3 License |
| `lgm` | 0.42B (plus the 1.3B ImageDream multi-view stage and its two CLIP encoders) | `steps` (30), `guidance` (5.0), `elevation` (0) | about 0.8 GB, plus 4.0 GB ImageDream | MIT; ImageDream under OpenRAIL (CreativeML OpenRAIL++-M) |
| `crm` | 4.1B: two 2.0B multi-view diffusion models (0.94B UNet, OpenCLIP ViT-H/14, SD VAE each) and a 0.12B reconstruction UNet | `steps` (50), `guidance` (5.0), `texture` (`False`) | about 12.8 GB, plus 0.4 GB BiRefNet | MIT code; weights MIT with Stable Diffusion 2.1's CreativeML OpenRAIL++-M use restrictions |

Speridlabs Idol by [Speridlabs](https://speridlabs.com/research/idol) runs
in under 8 GB of GPU memory. On CUDA, a seed gives the same object as the
Speridlabs SDK. Rarely, a run places part of the Gaussians differently
while the object stays the same: like the SDK, it runs one attention step on
a cuDNN kernel whose result can depend on GPU memory history (ADR 0031). `compile=True` (CUDA) compiles the sampler: on an H100
an object takes 6.4 s instead of 9.8 s, after a first call that compiles.

```python
model = LibreGen("speridlabs-idol", compile=True)
```

No setup: nothing is gated and no Hugging Face login is needed. The DINOv3
encoder Idol reads its input through comes from LibreYOLO's bf16 mirror,
[`LibreYOLO/LibreDINOv3vith16plus`](https://huggingface.co/LibreYOLO/LibreDINOv3vith16plus),
under Meta's DINOv3 License, which restricts some uses (military, weapons,
sanctioned parties among them); read it before use.

TripoSplat by [TripoAI](https://www.tripo3d.ai/research/triposplat) is the
model Idol builds on, under MIT; its weights come from LibreYOLO's mirror
[`LibreYOLO/LibreTripoSplat0.6b`](https://huggingface.co/LibreYOLO/LibreTripoSplat0.6b). Same call, same output, same encoders; its
own knobs are `steps` (upstream recommends 10 to 20), `guidance` (1 or less
turns it off), `shift`, `gaussians` (32,768 to 262,144) and `margin` (the
crop, 1.2 as upstream). It runs in fp16 on CUDA and Apple MPS and in fp32 on
CPU, with no compiled path.

```python
model = LibreGen("triposplat")
model("chair.jpg", steps=20, gaussians=131072).save("chair.ply")
```

TRELLIS by [Microsoft](https://microsoft.github.io/TRELLIS) builds the
object as a sparse voxel structure first, then 32 Gaussians per occupied
voxel, so the number of Gaussians follows the object (often 200,000 to
600,000) rather than a knob. Upstream needs CUDA-only extensions (spconv,
xformers or flash-attn); LibreYOLO runs the same weights in plain PyTorch,
on NVIDIA GPUs, Apple silicon and CPU. A seed gives the same starting noise
on every device. `steps` and `guidance` apply to both of its sampling
stages; `margin` is the crop around the object (1.2 as upstream).

```python
model = LibreGen("trellis")
model("chair.jpg", steps=12).save("chair.ply")
```

LGM by [Jiaxiang Tang et al.](https://me.kiui.moe/lgm/) works in two
stages: ImageDream samples four views around the object from the picture,
then LGM turns the four views into 65,536 Gaussians (those above opacity
0.005 are kept, as upstream saves them). Upstream needs xformers on CUDA;
LibreYOLO runs it in plain PyTorch on NVIDIA GPUs, Apple silicon and CPU, in
fp16 on GPUs and fp32 on CPU. `steps` and `guidance` drive ImageDream's DDIM
sampler (`guidance` 1 turns guidance off); `elevation` is the camera height
the picture was taken from, in degrees (negative is above the object). A
seed gives upstream's object on CUDA; other devices draw other noise.
ImageDream's weights carry the OpenRAIL use restrictions of their Stable
Diffusion lineage; read them on
[`LibreYOLO/LibreImageDream`](https://huggingface.co/LibreYOLO/LibreImageDream).

```python
model = LibreGen("lgm")
model("chair.jpg", elevation=-10).save("chair.ply")
```

TripoSR by [Tripo AI and Stability AI](https://github.com/VAST-AI-Research/TripoSR)
returns a mesh instead of Gaussians: a transformer turns the picture into a
triplane NeRF in one pass, and the mesh is the isosurface of its density. It
takes about a second on a GPU and runs in fp32 on NVIDIA GPUs, Apple silicon
and CPU. There is no sampling, so `seed` changes nothing.

```python
model = LibreGen("triposr")
result = model("chair.png")
result.save("chair.obj")                 # vertex colours as `v x y z r g b`
m = result.meshes
m.vertices[0], m.faces                   # (V, 3) z-up in [-0.5, 0.5]^3, (F, 3)
m.extras["vertex_colors"][0]             # (V, 3) RGB in [0, 1]
model("chair.png", resolution=384)       # finer marching-cubes grid
```

`resolution` is the marching-cubes grid (32 to 1024), `threshold` the
density at the surface, `margin` the crop around the object (upstream's
foreground ratio 0.85). The front of the object faces +x. Marching cubes runs
through scikit-image instead of upstream's torchmcubes: the same surface,
triangulated differently.

CRM by [Tsinghua University](https://ml.cs.tsinghua.edu.cn/~zhengyi/CRM)
returns a coloured triangle mesh instead of Gaussians. Two multi-view
diffusion models generate six views of the object and their canonical
coordinate maps, a convolutional UNet turns them into a triplane, and
FlexiCubes extracts the surface. Upstream needs xformers, nvdiffrast (NVIDIA,
non-commercial) and a GPL mesh library; LibreYOLO runs the same weights in
plain PyTorch on NVIDIA GPUs, Apple silicon and CPU, in fp16 on the GPUs as
upstream. A seed gives the same starting noise on every device; on CUDA a
repeat call gives the same views and triangles, with vertices that can move
by about 1e-7 (FlexiCubes accumulates with atomic adds, as upstream). An
object takes about 12 s and 12 GB on an RTX 3090.

```python
model = LibreGen("crm")
result = model("chair.png", seed=0)
result.save("chair.obj")                 # vertex colours as `v x y z r g b`
m = result.meshes
m.vertices[0], m.faces                   # (V, 3) z-up in [-0.5, 0.5]^3, (F, 3)
m.extras["vertex_colors"][0]             # (V, 3) RGB in [0, 1]
model("chair.png", texture=True).save("chair.obj")   # + chair.mtl, chair.png
```

`steps` and `guidance` apply to both diffusion stages. `texture=True` bakes a
1024 x 1024 UV texture as upstream does; it needs the MIT-licensed `xatlas`
package (`pip install xatlas`) for the UV unwrap. The input is padded to a
square on grey, as upstream, without a crop, so a photo works best with the
object filling the frame. The diffusion weights descend from Stable
Diffusion 2.1, so the CreativeML OpenRAIL++-M use restrictions shipped with
[`LibreYOLO/LibreCRM`](https://huggingface.co/LibreYOLO/LibreCRM) apply.

TRELLIS.2 by [Microsoft](https://microsoft.github.io/TRELLIS.2) returns
the object's shape as a mesh, up to a 1536^3 grid: a structure stage places
the occupied voxels, shape stages fill in a latent on them (at 512, then
again at 1024 or 1536), and a decoder turns it into a dual-grid mesh.
Upstream needs CUDA-only extensions (FlexGEMM, flash-attn, o-voxel);
LibreYOLO runs the same weights in plain PyTorch on NVIDIA GPUs, Apple
silicon and CPU. A seed gives the same starting noise on every device.

```python
model = LibreGen("trellis2")
result = model("chair.png", seed=0)
result.save("chair.obj")
m = result.meshes
m.vertices[0], m.faces, m.seed           # (V, 3) z-up in [-0.5, 0.5]^3, (F, 3)
model("chair.png", resolution=512)       # one shape stage, faster
```

Geometry only: upstream's texture stage bakes with nvdiffrast, a
non-commercial NVIDIA library, and is not part of LibreYOLO. The mesh is the
raw dual grid, before upstream's CuMesh clean-up: faces are not consistently
wound and some vertices are unused. On an A100 a 1024 object took 19.5 s and
14.3 GB of GPU memory at peak (512: 5.5 s, 11.6 GB). The DINOv3 encoder comes from
LibreYOLO's mirror
[`LibreYOLO/LibreDINOv3vitl16`](https://huggingface.co/LibreYOLO/LibreDINOv3vitl16)
under Meta's DINOv3 License; read it before use.

Direct3D-S2 by [DreamTech](https://www.neural4d.com/research/direct3d-s2)
returns a detailed mesh: it generates a signed distance field on sparse
voxels, first at 512^3 and then at 1024^3, with three flow transformers and
spatial sparse attention, and the mesh is the field's zero crossing. Upstream
needs CUDA-only extensions (torchsparse, flash-attn, Triton kernels, a CUDA
voxeliser); LibreYOLO runs the same weights in plain PyTorch. A seed gives
the same object: on CUDA the starting noise equals upstream's for
`torch.manual_seed(seed)`. It is slow: on an RTX A6000 one object at
1024^3 takes about 18 minutes and 23 GB of GPU memory, where upstream's
CUDA kernels take 2.5 minutes; `resolution=512` stops after the first pass.

```python
model = LibreGen("direct3d-s2")
result = model("chair.png", seed=0)
result.save("chair.obj")
model("chair.png", resolution=512)       # faster, coarser (stops after the first pass)
model("chair.png", steps=(25, 15, 8))    # fewer steps in the three stages
```

`resolution` is 1024 (upstream's default) or 512, `steps` the sampling steps
of the dense, 512 and 1024 stages (one integer sets all three), `guidance`
the classifier-free guidance of all three. Upstream's optional remeshing is
not included (it depends on GPL mesh tools); meshes come as marching cubes
returns them.

```bash
libreyolo generate source=chair.jpg
libreyolo generate source=chair.png model=trellis2
libreyolo generate source=chair.png model=triposr
libreyolo generate source=chair.png model=crm texture=true
libreyolo generate source=photos/ mode=real gaussians=1048576 seed=0
libreyolo generate source=chair.jpg model=triposplat
libreyolo generate source=chair.jpg model=trellis
libreyolo generate source=chair.jpg model=lgm
libreyolo generate source=chair.png model=direct3d-s2
```

With `source=` the model defaults to `speridlabs-idol` and objects are saved
as `.ply` (`.obj` for `triposr`, `crm` and `trellis2`).

as `.ply` (`.obj` for `triposr`, `crm` and `direct3d-s2`).

## Not supported yet

`train()`, `val()`, `export()` and, on text-to-image models, `source=`
(image-conditioned generation) raise `NotImplementedError`. Their names are
reserved by ADRs 0030 and 0031.

## Licensing

LibreYOLO ships only its own adapter and payload code, plus the Speridlabs
Idol networks ported from the Apache-2.0 Idol SDK, the TripoSplat
networks ported from the MIT TripoSplat repository, the TRELLIS networks
adapted from the MIT TRELLIS repository, the TRELLIS.2 networks adapted from
the MIT TRELLIS.2 repository, the TripoSR networks adapted from the MIT
TripoSR repository and the CRM networks adapted from the MIT CRM
repository with FlexiCubes from NVIDIA's Apache-2.0 release (see each
family's `NOTICE`). LGM's networks and its ImageDream sampler are adapted
from the MIT LGM repository. Text-to-image models load through the
Apache-2.0 `diffusers` API.
Weights are downloaded from the upstream Hub repos at pinned revisions. The
exceptions are the DINOv3 encoders Idol and TripoSplat (ViT-H+, mirrored in
bf16) and TRELLIS.2 (ViT-L, mirrored unchanged) read, which LibreYOLO
redistributes under Meta's DINOv3 License. LibreYOLO adds no content filter;
upstream usage policies apply.
