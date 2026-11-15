# DPFlow parity

Upstream: ptlflow `DPFlow` (https://github.com/hmorimitsu/ptlflow, commit
`43cdc2b5fc7228e7090009a66e8e28e0b48e31ba`), loaded with
`ptlflow.get_model("dpflow", ckpt_path=...)` in its own venv.
LibreYOLO: the converted checkpoints through `LibreYOLO(...).predict([...])`.
Script: `weights/parity_dpflow.py`. Hardware: 1x RTX 3090, torch 2.11.0+cu128.

Input: the 24-frame, one-second clip (640x360 PNG), 23 consecutive pairs,
fp32, TF32 off, `cudnn.benchmark = False`, `cudnn.deterministic = True`.
Upstream reads frames with OpenCV (BGR in [0, 1]); LibreYOLO reads the same
PNGs through its loader (RGB in [0, 1]). Both run at native resolution
(3 pyramid levels, output stride 32, `allpairs` correlation, 4 iterations
per level, no warm start).

| Variant | File | Pairs | Max abs diff (px) | Mean abs diff (px) | Mean EPE between the two (px) | Upstream mean flow magnitude (px) |
|---|---|---|---|---|---|---|
| things | `LibreDPFlowm-flow.pt` | 23 | 0.0 | 0.0 | 0.0 | 28.943 |
| sintel | `LibreDPFlowm-flow-sintel.pt` | 23 | 0.0 | 0.0 | 0.0 | 30.773 |
| kitti | `LibreDPFlowm-flow-kitti.pt` | 23 | 0.0 | 0.0 | 0.0 | 30.725 |
| spring | `LibreDPFlowm-flow-spring.pt` | 23 | 0.0 | 0.0 | 0.0 | 25.608 |

The outputs are bitwise identical.

## Why cuDNN must be pinned for the comparison

With cuDNN autotuning (`benchmark = True`), each process may pick different
convolution kernels. The recurrent refinement amplifies those last-bit
differences at a few ambiguous pixels. On the things checkpoint:

| Setting | Comparison | Max abs diff (px) | Mean over pairs of per-pair max (px) |
|---|---|---|---|
| benchmark, two processes | upstream vs LibreYOLO | 0.423 | n/a (mean abs diff 4.9e-4, mean EPE 8.0e-4) |
| benchmark, one process | upstream vs vendored network | 0.206 | 0.021 |
| benchmark, one process | upstream vs upstream (second run) | 0.183 | 0.021 |
| deterministic, one process | upstream vs vendored network | 0.0 | 0.0 |

The second and third rows show that upstream run twice against itself
differs by as much as the port does, so the difference in the first row is
the kernel noise floor, not a port defect.

## Synthetic known shift

- 640x498, content moved by (-3, -2) px (crop offset), things checkpoint,
  benchmark mode: upstream vs LibreYOLO max abs diff 7.0e-5 px, mean 3.9e-7 px.
- 3840x2160, content moved by (-18, -12) px: LibreYOLO median flow
  (-17.980, -11.976) px. The all-pairs volume needs 63.5 GiB at this size,
  so the model falls back to the pure-PyTorch `torch` correlation (as
  upstream does) with the window sampled in row chunks: 5 pyramid levels,
  7.3 s for the pair (including the failed all-pairs attempt), 10.5 GB peak
  GPU memory.
- `tests/unit/test_dpflow.py::test_torch_corr_mode_matches_allpairs`: the
  `torch` engine matches `allpairs` within 1e-3 px on a random-weight
  48x64 pair (CPU).
