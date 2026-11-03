# SEA-RAFT parity

Upstream princeton-vl/SEA-RAFT at commit 9137517ba24e628442aec097d3afe71d03503b75
against LibreSEARAFT through the public API (`LibreYOLO(pt)(frames)`), produced
by `weights/parity_searaft.py`.

- Hardware: NVIDIA GeForce RTX 3090, torch 2.11.0+cu128, fp32, TF32 disabled
  for matmul and cuDNN.
- Input: the LibreFlow one-second clip, 24 consecutive 640x360 PNG frames, so
  23 pairs (frame i to frame i + 1). 360 is a multiple of 8, so no padding.
- Upstream: `RAFT(args)` from `config/eval/sintel-{S,M,L}.json` (scale 0,
  native resolution), weights loaded from
  `MemorySlices/Tartan-C-T-TSKH-spring540x960-{S,M}` `model.safetensors` with
  `strict=False` as upstream does (the 16 missing keys are the `downsample.1`
  aliases of the shared `bn3` BatchNorm), `forward(..., test_mode=True)`, last
  flow and info.
- LibreYOLO: converted `LibreSEARAFT{s,m,l}-flow.pt`, flow and confidence from
  `Results.flow`. Upstream confidence is computed with the same formula from
  its `info` output (`custom.py::get_heatmap`, then `1 - heatmap / var_max`).

| Size | Iterations | Max abs diff (px) | Mean abs diff (px) | Mean EPE between (px) | Max confidence diff | Mean flow magnitude (px) |
|---|---|---|---|---|---|---|
| s | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 26.30 |
| m | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 27.24 |
| l | 12 | 0.0 | 0.0 | 0.0 | 0.0 | 27.34 |

The outputs are bit-identical. This depends on the preprocess multiplying by
the float32 reciprocal of 255, which is what upstream's `image / 255.0`
computes on CUDA. A first run that divided by 255 on the CPU (one ulp off on
some pixels) gave max abs diffs of 0.015 px (s), 0.023 px (m) and 1.17 px (l)
on the largest motions, with mean abs diffs of 2.7e-5, 1.8e-5 and 6.5e-4 px:
the iterative refinement amplifies input rounding, the more so with 12
iterations.

On CPU, upstream divides with torch's CPU kernel instead, so a CPU comparison
can show the same one-ulp-scale drift. It is not a model difference.

A synthetic check (the Guggenheim asset translated by (2.5, 1.25) px per frame,
three pairs, 640x498 so padded to 504) with the earlier CPU division gave max
abs diffs of 5.6e-6 (s), 1.8e-5 (m) and 3.4e-5 px (l).
