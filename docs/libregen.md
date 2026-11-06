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

`flux2-klein` and `sana-sprint` are short aliases.

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
| `triposplat` | 0.66B (plus the same two encoders) | `steps` (20), `guidance` (3.0), `shift` (3.0), `gaussians` (262144), `margin` (1.2) | about 1.8 GB, plus the same DINOv3 and VAE | MIT; the DINOv3 encoder under Meta's DINOv3 License |
| `lgm` | 0.42B (plus the 1.3B ImageDream multi-view stage and its two CLIP encoders) | `steps` (30), `guidance` (5.0), `elevation` (0) | about 0.8 GB, plus 4.0 GB ImageDream | MIT; ImageDream under OpenRAIL (CreativeML OpenRAIL++-M) |

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

```bash
libreyolo generate source=chair.jpg
libreyolo generate source=photos/ mode=real gaussians=1048576 seed=0
libreyolo generate source=chair.jpg model=triposplat
libreyolo generate source=chair.jpg model=trellis
libreyolo generate source=chair.jpg model=lgm
```

With `source=` the model defaults to `speridlabs-idol` and objects are saved
as `.ply`.

## Not supported yet

`train()`, `val()`, `export()` and, on text-to-image models, `source=`
(image-conditioned generation) raise `NotImplementedError`. Their names are
reserved by ADRs 0030 and 0031.

## Licensing

LibreYOLO ships only its own adapter and payload code, plus the Speridlabs
Idol networks ported from the Apache-2.0 Idol SDK, the TripoSplat
networks ported from the MIT TripoSplat repository and the TRELLIS networks
adapted from the MIT TRELLIS repository (see each family's `NOTICE`). LGM's networks
and its ImageDream sampler are adapted from the MIT LGM repository. Text-to-image models load through the Apache-2.0 `diffusers` API.
Weights are downloaded from the upstream Hub repos at pinned revisions. The
one exception is the DINOv3 encoder Idol and TripoSplat read, which
LibreYOLO mirrors in bf16 under Meta's DINOv3 License. LibreYOLO adds no content filter;
upstream usage policies apply.
