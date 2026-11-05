# FlowSeek parity

Upstream: [mattpoggi/flowseek](https://github.com/mattpoggi/flowseek) at
`ba6407370bceeb96244cad4d60d37489f6eb471e`, checkpoints
`flowseek_T_TartanCT_TSKH.pth` (sha256 `7913a917...312f`, sizes t and s) and
`flowseek_M_TartanCT_TSKH.pth` (sha256 `42bc1388...85c6`, sizes m and l), run
through the upstream `FlowSeek` class with `config/eval/flowseek-[T,S,M,L].json`
and `test_mode=True`.

LibreYOLO: the converted `LibreFlowSeek{t,s,m,l}-flow.pt` through the public
predict path, `model([frame_001, ..., frame_024])`.

Input: a one-second clip, 24 consecutive PNG frames at 640x360, giving 23
consecutive pairs. Hardware: NVIDIA RTX 3090, torch 2.11.0+cu128, fp32 with
TF32 disabled. Script: `weights/parity_flowseek.py`.

Differences are taken over every pixel of every pair, in pixels.

| size | iterations | pairs | max abs diff | mean abs diff | mean EPE between | mean flow magnitude |
|------|-----------:|------:|-------------:|--------------:|-----------------:|--------------------:|
| t    | 4          | 23    | 0.0          | 0.0           | 0.0              | 29.77               |
| s    | 12         | 23    | 0.0          | 0.0           | 0.0              | 30.22               |
| m    | 4          | 23    | 0.0          | 0.0           | 0.0              | 29.01               |
| l    | 12         | 23    | 0.0          | 0.0           | 0.0              | 29.40               |

The outputs are bitwise identical.

## Attention kernel

The shared Depth Anything V2 vendor (`libreyolo/models/depth_anything/_vendor`)
runs DINOv2 attention through `scaled_dot_product_attention`. FlowSeek's DAv2
instance is pinned to upstream's manual `q * scale @ k.T`, softmax, `@ v` math
(`_vendor/flowseek.py`). With the fused kernel instead, the same run gave:

| size | max abs diff | mean abs diff | mean EPE between |
|------|-------------:|--------------:|-----------------:|
| t    | 0.473        | 3.8e-4        | 6.3e-4           |
| s    | 2.604        | 2.0e-3        | 3.3e-3           |
| m    | 1.171        | 8.1e-4        | 1.3e-3           |
| l    | 41.094       | 3.2e-2        | 5.0e-2           |

The kernel's rounding differences in the encoder features are amplified by the
recurrent refinement at a few ambiguous pixels, so FlowSeek keeps the manual
math.

A synthetic check on three Guggenheim frames shifted by (3, 2) px per frame
recovers a mean flow magnitude of 3.60 px for every size (expected 3.61).

No ground-truth EPE is reported here: the clip has no ground-truth flow, and
the flow8 fixture was not available when these numbers were taken.
