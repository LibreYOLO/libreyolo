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
- Iris-Z's depth-to-image and joint image-plus-depth modes land on the
  `generate` task (amendment below), not as new tasks.
- Depth through `LibreGen` is the first non-generative task in the tier; it
  reuses the `DepthMap` payload and plotting, so depth consumers see the same
  result type as from `LibreYOLO(...)` depth families.

## Amendment: depth to image and image plus depth (2026-11-14)

Both remaining upstream modes are generation, so they stay on the
`generate` task and the same loaded model:

```python
model = LibreGen("speridlabs-iris-z")
model("a sunken ruin under the sea", source="photo.jpg")   # depth -> image
model("a grand old library", depth=True)                   # image + depth
```

- `source=` on `generate` (reserved by ADR 0030 for image-conditioned
  generation) is the depth the image is painted on. It takes one of: a photo
  (its depth is estimated first with the depth defaults, seed 0, at `imgsz`
  512 or 1024); a `Results` or `DepthMap` from any depth model (`log_depth`
  is used as is, `depth` and `inverse_depth` are min-max scaled and taken to
  log depth); or a grayscale depth map image with `depth_format="depth"`
  (bright = far) or `"disparity"` (bright = near). A photo is never guessed
  to be a depth map: without `depth_format` an image is a photo.
- With `source=`, `imgsz` is one side (default 1024) and the output keeps the
  depth's aspect ratio, sized as upstream's `depth_to_image`. Prompts, `n`,
  `seed`, `save`, `steps` and `guidance` work as for text to image.
  `Results.depth_map` is the depth the image was painted on;
  `generated.params["source"]` records the source.
- `depth=True` (a `generate` knob, default `False`) samples image and depth
  jointly as upstream's `text_to_rgbd`, then, as upstream's default,
  re-estimates the depth from the finished image (sharper than the joint
  depth) with seed `seed` and 4 steps. The result carries both
  `Results.generated` and `Results.depth_map`. The joint depth itself is not
  exposed.
- Not included: upstream's coloured depth-map reader (`depth_format=
  "colours"`, which needs SciPy) and the CLI (`prompt=` and `source=` stay
  mutually exclusive there).
