# MEMFOF parity: upstream vs LibreYOLO

Measured 2026-11-03 on an NVIDIA GeForce RTX 3090, torch 2.11.0+cu128, with
`weights/parity_memfof.py`.

- Upstream: `msu-video-group/memfof` at commit
  `a51de9fc59c6fe20ba08e079372c7b583d58a712`, loaded with
  `MEMFOF.from_pretrained` from `egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH`
  at revision `6c6c9aa3ad64f93aee8efbc2f7a6e4535814ee96`, called as its README
  shows: float `[0, 255]` input `[1, 3, 3, H, W]`, default 8 iterations,
  forward flow `model(x)["flow"][-1][:, 1]`.
- LibreYOLO: `LibreYOLO("LibreMEMFOFm-flow.pt")` converted from the same
  checkpoint, run on the whole frame list through the flow runner
  (`FLOW_CONTEXT = (1, 0)`, so result t is computed from frames t-1, t, t+1).
- Input: one second of the Sintel trailer (Blender Foundation, CC BY 3.0) from
  00:00:30, 24 frames at 640x360. Compared: the 22 interior frames, 2 to 23,
  each with its two neighbours. Mean upstream flow magnitude on these frames:
  28.7 px.

| Precision | Frames | Max abs diff (px) | Mean abs diff (px) | Mean EPE between the two (px) |
|---|---|---|---|---|
| fp32, TF32 off (upstream's inference default) | 22 | 0.0 | 0.0 | 0.0 |
| bf16 autocast on both sides | 22 | 0.0 | 0.0 | 0.0 |

Upstream's inference code and demo run in fp32; only its training script uses
`bf16-mixed`. LibreYOLO therefore defaults to fp32. The bf16 row autocasts both
sides identically and is reported for completeness. The match is bit-exact
because the vendored network is upstream's code with inference-neutral edits
(imports, no hub mixin, no ImageNet download, `einops.rearrange` written as
`reshape`/`permute`) and the converted checkpoint holds the same tensors.

Reproduce on a CUDA box:

```
PYTHONPATH=<upstream memfof checkout> python weights/parity_memfof.py \
    --frames <dir of 24 PNG frames> \
    --upstream-dir <snapshot of egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH> \
    --ckpt LibreMEMFOFm-flow.pt [--bf16]
```
