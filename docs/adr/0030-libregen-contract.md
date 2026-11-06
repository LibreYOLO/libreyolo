# ADR 0030: LibreGen Contract For Generative Image Models

- Status: Proposed
- Date: 2026-10-10
- Scope: New sibling tier (`LibreGen`), new canonical task (`generate`), new
  result payload (`Results.generated`), new CLI verb (`libreyolo generate`)

## Context

A generative image model takes text and returns a synthesized image. None of
LibreYOLO's existing contracts fit it:

- `LibreYOLO(...)` sniffs a `.pt` state dict and maps an image to a
  prediction about that image. A generator has no input image, no labels and
  no validation metric that `val()` could report today.
- `restore` is paired image in, image out on the source canvas. A generated
  image has no source canvas.
- `LibreVLM(...)`, `LibreVLA(...)` and `LibreLLM(...)` return boxes, action
  chunks and text.

The documented de-facto YOLO API has no image generation surface, so there
is no name to mirror. The contract below reuses the vocabulary that API
already gave users (`imgsz`, `device`, `save`, `stream`, `Results`) and adds
the minimum on top.

Current open generators are published as multi-file pipeline directories on
the Hugging Face Hub and load through `diffusers` (Apache-2.0).

## Decision

Add `LibreGen` as a sibling factory, the task `generate`, and
`Results.generated`. The library owns the call shape, the result payload,
the seeding rule, the weights contract and the user-facing API. The
networks, schedulers and text encoders come from the upstream runtime
through its public API. No upstream source is ported, adapted or vendored.

```python
from libreyolo import LibreGen

model = LibreGen("flux2-klein-4b")
result = model("a red fox sleeping in fresh snow")
result.save("fox.png")
```

## Task contract: `generate`

`generate` is the canonical task for models whose public output is a
synthesized image. Filename suffix `-generate`. Aliases `generation`,
`image-generation`, `text-to-image`, `text2image`, `txt2img` and `t2i`
resolve to `generate` at the API boundary.

`Results.generated` holds a `GeneratedImage` payload:

- `data` is `(H, W, 3)` uint8 RGB. `array` returns the ndarray, `image` a
  `PIL.Image`.
- `prompt` is the text that produced the image and `seed` the integer that
  reproduces it. `params` records the rest of the call: `model`, `imgsz` and
  every family knob in effect, defaults included.
- `save(path)` writes the image. A PNG also stores prompt, seed and params as
  JSON in a `libreyolo` text chunk.
- `Results.boxes` is `None`, `Results.path` is `None`, `Results.orig_shape`
  is the image's `(H, W)`, `len(result)` is 1. `summary()` returns one row
  with the payload's record. `result.plot()` returns the image and
  `result.save(path)` writes it.

Box-specific paths (tracking, tiling, export backends, mAP validation) do
not apply to `generate` results.

## Call contract

`generate(prompt, *, imgsz=None, seed=None, n=1, stream=False, save=False,
output_path=None, **knobs)`. `model(...)` and `model.predict(...)` are the
same call.

Arguments every generator understands:

- `prompt` is a string or a list of strings. A string is always a prompt,
  never a path, so `source=` stays free for image-conditioned generation.
- `imgsz` is an int (square) or `(height, width)`, default the model's
  native size. Sides are rounded up to the family's size multiple with a
  logged warning.
- `seed` in `[0, 2**32)`. Image `i` of a prompt uses `seed + i`, wrapping.
  Each prompt in a list starts from the same `seed`, so a prompt and a seed
  name one image wherever the prompt sits. `seed=None` draws a fresh seed per
  prompt and records it. Seeds drive CPU generators so the starting noise
  does not depend on the device; results can still differ across devices and
  dtypes by numerical error.
- `n` is images per prompt.
- `save=True` writes PNGs to a new `runs/generate/predict*` directory per
  call. `output_path` names the file for one prompt with `n=1`.
- `stream=True` returns a generator.

One prompt with `n=1` returns a `Results`. A list of prompts or `n > 1`
returns a flat list in prompt order.

Family knobs:

- Everything else is a property of one kind of model, not of the task. A
  family declares the knobs it accepts (`KNOBS`) and its defaults
  (`DEFAULTS`), so a bare `model(prompt)` always works.
- A knob the family does not have raises `TypeError` naming the supported
  arguments. It is never ignored.
- A shared concept keeps a shared name: `steps` for iterative sampling,
  `guidance` for prompt guidance, `negative` for a negative prompt. A family
  invents a name only for something that is its own.

## Families

| Alias | Upstream | License | Knobs (defaults) |
| --- | --- | --- | --- |
| `flux2-klein-4b` (default) | `black-forest-labs/FLUX.2-klein-4B` | Apache-2.0 | `steps` (4), `guidance` (1.0; the distilled checkpoint runs without guidance, so any other value raises) |
| `sana-sprint-0.6b` | `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers` | Apache-2.0; bundled Gemma 2 text encoder under the Gemma Terms of Use | `steps` (2), `guidance` (4.5) |
| `prx-<size>` (nine checkpoints, amendment below) | `Photoroom/prx-*` | Apache-2.0; bundled T5Gemma text encoder under the Gemma Terms of Use; FLUX.1 VAE Apache-2.0 or DC-AE MIT | `steps` (28, or 8 distilled), `guidance` (5.0, or 1.0 distilled), `negative` ("") |
| `prxpixel-7b` | `Photoroom/prxpixel-t2i` | Apache-2.0; bundled Qwen3-VL text encoder Apache-2.0 | `steps` (28), `guidance` (5.0), `negative` ("") |

All load through `diffusers` (`Flux2KleinPipeline`, `SanaSprintPipeline`,
`PRXPipeline`, `PRXPixelPipeline`).
A family that needs a vendored architecture implements the same two hooks
(`_load`, `_generate`) without changing the call shape.

## Weights

Base weights are snapshot directories, not `.pt` files: the upstream
pipeline folder downloaded from the Hub at a pinned commit into
`weights/Libre<Family><size>/`, never converted and not redistributed by
LibreYOLO. Only the component folders are fetched; a repo's root-level
single-file checkpoint is skipped.

`LibreGen(path)` loads a local directory in a supported family's upstream
format, matched by the pipeline class recorded in `model_index.json`. That is
how an offline copy or a fine-tune made with upstream tooling loads today.

## Reserved for later

These names are part of the contract now so the surface does not move when
they land. Each raises `NotImplementedError` with its reason until then.

- `train(data=...)`: fine-tuning on captioned images, returning the standard
  results dict with `save_dir`, `best`, `last`, loadable with
  `LibreGen(path)`. The dataset format, the adapter format and the checkpoint
  contract file are a later amendment.
- `val(data=...)`: generation metrics on a captioned image set.
- `source=`: image-conditioned generation (editing, image-to-image).
- `export()`: needs a graph contract for a sampling loop over several
  networks.

## Public API

```python
from libreyolo import LibreGen

model = LibreGen("flux2-klein-4b")                  # or "sana-sprint-0.6b"
result = model("a red fox", seed=0)                 # Results
result.generated.array                              # (1024, 1024, 3) uint8 RGB
result.generated.prompt, result.generated.seed
result.save("fox.png")

results = model(["a red fox", "a blue jay"], n=2, imgsz=(768, 1024), save=True)
for result in model(prompts, stream=True):
    ...

model("a red fox", steps=8)                         # a family knob
model = LibreGen("flux2-klein-4b", offload=True)    # idle components off the GPU
model = LibreGen("path/to/pipeline_dir")            # local directory
```

```bash
libreyolo generate prompt="a red fox" model=sana-sprint-0.6b seed=0
```

Constructor arguments: `device` (`"auto"` picks CUDA, then MPS, then CPU),
`dtype` (`"auto"` is bf16 on CUDA and MPS, fp16 on CUDA without bf16, fp32
on CPU and on MPS before macOS 14; or `"bf16"`, `"fp16"`, `"fp32"`), `offload` (CUDA only).
`model.pipeline` is the loaded upstream object, the raw escape hatch.

Install: `pip install "libreyolo[gen]"`.

## Out of scope

- Content filtering. LibreYOLO adds no safety checker; upstream usage
  policies apply to the weights.
- Video, audio and 3D generation. (Image-to-3D was added by ADR 0031.)
- LoRA loading and prompt rewriting.

## Consequences

- Image generation gets one stable call shape that does not depend on how a
  model samples.
- `diffusers` is an optional extra, not a core dependency.
- Adding a diffusers-backed family is a class with a repo, a pinned commit,
  a pipeline class and its knobs.

## Amendment: PRX and PRXPixel (2026-11)

Photoroom's PRX release is nine checkpoints of one 1.3B architecture.
`LibrePRX` serves all of them as size codes of one family: `1024-beta`,
`512-sft-distilled`, `512-sft`, `512`, `512-dc-ae-sft-distilled`,
`512-dc-ae-sft`, `512-dc-ae`, `256-sft` and `256`, each pinned to a Hub
commit. The size code also picks two things a single class constant cannot:
the size multiple (16 for the FLUX.1 VAE, 32 for DC-AE) and the model card's
sampling defaults (28 steps at guidance 5.0, or 8 steps at guidance 1.0 for
the distilled checkpoints). A local PRX directory with no `size=` gets the
`-sft` code matching its `default_sample_size` and autoencoder, or
`1024-beta` at 1024.

`LibrePRXPixel` (`prxpixel-7b`) is the 7B pixel-space variant. Its pipeline
first shipped in diffusers 0.39, so it requires that release and names it in
the import error; the `gen` extra keeps its 0.38 floor so it stays
installable next to `marigold`, which pins 0.38.

Unlike FLUX.2 klein and SANA-Sprint, the PRX and PRXPixel snapshots are
mirrored, unmodified, at `LibreYOLO/LibrePRX<size>` and
`LibreYOLO/LibrePRXPixel7b` (Apache-2.0, redistribution allowed; the PRX
mirrors carry Photoroom's LICENSE and NOTICE and the Gemma terms), so the
family keeps working if the upstream repos move. The classes pin the mirror
commits in `HF_REVISIONS` and record Photoroom's repos and commits in
`UPSTREAM_REPOS` and `UPSTREAM_REVISIONS`; every mirrored file is checked
byte for byte against upstream.
