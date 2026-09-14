# ADR 0026: Albedo task contract

Status: implemented
Date: 2026-09-10

## Decision

Add task `albedo`, suffix `-albedo`, and aliases `albedo-estimation` and
`albedo_estimation`. `Results.albedo` is an `AlbedoMap`: finite float32
`(H,W,3)` linear-RGB diffuse reflectance in `[0,1]` on the original canvas.
It represents surface colour without illumination, not a restored photograph.
The existing uint8 `RestoredImage` contract would lose the numeric precision
and colour-space meaning needed for reflectance evaluation.

`AlbedoMap.array` and `.data` retain linear values. `.to_rgb()`, `.save()` and
`Results.plot()` render an 8-bit preview using the standard sRGB transfer
function. A preview is not a quantitative target. NumPy/device moves and
Results slicing preserve the full dense map. JSON reports its shape and
`color_space: linear_rgb`.

Labels pair `images/<split>/<name>.<image extension>` with
`albedo/<split>/<name>.npy`. Targets are floating-point HWC linear RGB in
`[0,1]`, with exactly the input dimensions. Non-finite, integer, out-of-range
or mismatched targets are rejected. YAML retains the standard `path`, `train`
and `val` keys; `input_dir` and `albedo_dir` optionally rename those folders.

Validation stretches each pair to the selected canvas, then reports per-image
linear-RGB `metrics/PSNR` and `metrics/SSIM`, averaged over images. SSIM reuses
the existing 11x11 Gaussian window, sigma 1.5. PSNR is capped at 100 dB for a
perfect match, matching the existing image metric helper. Fitness is PSNR.
This is a library evaluation protocol, not a claim of upstream benchmark parity.

Marigold V2 is the first producer. Its decoder values are mapped from `[-1,1]`
to linear RGB, resized to the original canvas, and clipped to `[0,1]`.
Its upstream demo uses a gamma-2.2 preview; that display transform is not
applied to the numeric LibreYOLO payload.

Training, export, tracking, tiling and test-time augmentation are not supported
by the initial albedo family. They must fail explicitly. No dataset is bundled
or automatically downloaded by this task.
