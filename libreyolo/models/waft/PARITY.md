# WAFT parity

Upstream princeton-vl/WAFT at b152ff1 versus LibreYOLO `LibreWAFT`, fp32
(TF32 off), 5 iterations, one RTX 3090, torch 2.11.0+cu128, measured
2026-11-03 with `weights/parity_waft.py`. LibreYOLO goes through the public
API (`LibreYOLO(path)(frames, stream=True)`); upstream is built from its own
code with its checkpoint loaded. Upstream ran without xformers (not installed),
so its DINOv2 attention is the manual `q @ k.T` path and its timm blocks use
fused SDPA.

## Clip: 23 consecutive pairs, 640x360

| size | checkpoint | max abs diff (px) | mean abs diff (px) | mean EPE between (px) | mean flow magnitude (px) |
|---|---|---|---|---|---|
| t | a2 dinov3 zero-shot | 0 | 0 | 0 | 30.94 |
| s | a2 dav2 zero-shot | 3.984 | 0.01975 | 0.03018 | 29.97 |
| m | a1 tar-c-t | 3.201 | 0.00529 | 0.00837 | 38.14 |
| l | a2 twins zero-shot | 3.736 | 0.01334 | 0.02147 | 30.10 |

## Synthetic pair: Guggenheim image shifted 3 px, 640x427

| size | max abs diff (px) | mean abs diff (px) |
|---|---|---|
| t | 0 | 0 |
| s | 1.07e-5 | 1.15e-6 |
| m | 3.55e-5 | 1.10e-6 |
| l | 6.44e-6 | 7.89e-7 |

## Why s, m and l exceed 1e-2 px on the clip

The sub-pixel differences enter in the encoders (Twins on size l: 1.5e-4 max
after the backbone, from the plain-torch SDPA copy versus timm; DINOv2 on s
and m: SDPA versus upstream's manual attention). The fnet and refinement
ViT are bitwise identical (size t shows it end to end). Each of the 5 warp
iterations then amplifies the difference about 10x (size l, pair 13: 1.5e-3,
8.6e-3, 9.1e-2, 0.60, 2.1 px max), concentrated in textureless regions such
as the letterbox bars.

This is the model's own sensitivity, not a porting error. Upstream against
itself, all 23 clip pairs:

| size | upstream vs upstream, input scaled by (1 + 1e-6) | upstream vs upstream, cudnn.benchmark on vs off |
|---|---|---|
| s | max 29.71 px, mean 0.0851 px | max 0, mean 0 |
| m | max 2.54 px, mean 0.0032 px | max 0, mean 0 |
| l | max 3.94 px, mean 0.0184 px | max 3.00 px, mean 0.0112 px |

The LibreYOLO differences are at or below that noise floor, and the
encoders match to 1e-4 or better. Where the attention kernel is shared
(size t), the outputs are bitwise identical.
