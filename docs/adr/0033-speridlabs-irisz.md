# ADR 0033: Speridlabs Iris-Z, a LibreGen family with a depth task

Status: implemented
Date: 2026-11-11

## Context

Speridlabs Iris-Z (https://speridlabs.com/research/iris-z) is FLUX.2
fine-tuned to denoise an RGB latent grid and a depth latent grid in one
sequence. The same weights generate images from a prompt and estimate depth
from a photo (4 steps, beating specialist depth models on the MoGe and
Marigold protocols). It comes in two sizes: Iris-Z-small (9.1B, from FLUX.2
[klein] Base 9B, Qwen3 8B text encoder) and Iris-Z-big (32.2B, from FLUX.2
[dev], Mistral Small 3.2 24B text encoder). Code is Apache-2.0
(`speridlabs/iris-z-sdk`); weights are under the FLUX Non-Commercial License.

Depth families normally live in the `LibreYOLO(...)` checkpoint factory as
one `.pt` per size. Iris-Z does not fit that tier: the weights are a 18 GB or
64 GB sharded transformer that LibreYOLO does not mirror, the model is a
sampling loop over two networks, and the same weights also generate images.

## Decision

Add `LibreSperidlabsIrisZ` (`speridlabs_irisz`, sizes `9b` and `32b`) to
the `LibreGen` tier with `SUPPORTED_TASKS = ("generate", "depth")`:

```python
LibreGen("speridlabs-iris-z")("a lighthouse at dusk")            # generate (default)
LibreGen("speridlabs-iris-z", task="depth")("street.jpg")        # depth
```

- `generate` follows ADR 0030 unchanged: prompts, `imgsz`, `seed`, `n`,
  knobs `steps` (50) and `guidance` (4.0 for `9b`, 3.5 for `32b`).
- `depth` takes a source as predict does and returns `Results.depth_map`, a
  `DepthMap` with `encoding="log_depth"` (ADR 0025) at the source's size and
  `valid_mask=False` on sky. `imgsz` is the working resolution, 512 or 1024
  (default: 512 under half a megapixel, else 1024, as upstream). Knobs
  `steps` (4) and `ensemble` (1, the per-pixel median over that many seeds).
  The default seed is 0, as upstream, so depth repeats; `n` does not apply.
  `save=True` writes the coloured map under `runs/depth/predict`.
- The task is fixed at construction, so a depth model never loads a text
  encoder; it conditions on the empty-prompt embedding Speridlabs ships.
- CLI: `libreyolo generate source=... task=depth`; the model defaults to
  `speridlabs-iris-z` for that task.
- Names keep the maker, as for Speridlabs Idol: aliases
  `speridlabs-iris-z` (`9b`), `-9b`, `-small`, `-32b`, `-big`; no bare
  `iris-z`.

## Weights

Not redistributed. The size's folder (`Iris-Z-small/` or `Iris-Z-big/`) of
`speridlabs/iris-z` downloads at a pinned revision into
`weights/LibreSperidlabsIrisZ<size>/`, with a one-time license notice. The
FLUX.2 VAE is the Apache-2.0 bfloat16 copy in `FLUX.2-klein-4B` (upstream
reads the float32 copy in the gated FLUX.2-dev repo), upcast to float32 and
run under bfloat16 autocast as upstream does. Text encoders download from
Mistral AI and Qwen (Apache-2.0). Nothing is gated. A local snapshot or
variant folder loads with `LibreGen(path)`; the size comes from its
`config.json`.

## Consequences

- `val()`, `train()` and `export()` raise, as for every LibreGen family;
  export is blocked for both tasks in the support matrix.
- Image-conditioned generation (`source=` on `generate`, reserved by ADR
  0030) stays reserved. Iris-Z's depth-to-image and joint image-plus-depth
  modes are not exposed yet; they would land on that reserved surface and a
  generated-depth payload respectively.
- Depth through `LibreGen` is the first non-generative task in the tier; it
  reuses the `DepthMap` payload and plotting, so depth consumers see the same
  result type as from `LibreYOLO(...)` depth families.
