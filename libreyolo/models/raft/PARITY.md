# LibreRAFT parity

Upstream: princeton-vl/RAFT at commit 2888e15, checkpoints from the upstream
`download_models.sh` (`models.zip`). LibreYOLO side: the converted
`LibreRAFT*-flow*.pt` files, run through the public call
`LibreYOLO(weights)(frames)` (sequence runner, `_forward_flow`, postprocess).
Upstream side: `demo.py` recipe (`InputPadder` sintel mode, `iters=20`,
`test_mode=True`, then `unpad`).

Box: RTX 3090, torch 2.11.0+cu128, fp32, TF32 disabled, cudnn.benchmark off.
Script: `weights/raft_parity.py` (imports upstream code from a local clone).

Iteration count: 20 for both sizes, the upstream `demo.py` value for arbitrary
frame pairs. Upstream `evaluate.py` uses 32 on Sintel and 24 on KITTI and
Chairs for benchmark numbers; LibreRAFT does not change iteration count per
dataset.

## Sintel trailer clip, 23 consecutive pairs, 640x360

| Weights | Upstream file | Max abs diff (px) | Mean abs diff (px) | Mean EPE between (px) | Mean flow magnitude (px) |
|---|---|---|---|---|---|
| LibreRAFTl-flow.pt | raft-things.pth | 0.0 | 0.0 | 0.0 | 38.06 |
| LibreRAFTs-flow.pt | raft-small.pth | 0.0 | 0.0 | 0.0 | 38.09 |
| LibreRAFTl-flow-sintel.pt | raft-sintel.pth | 0.0 | 0.0 | 0.0 | 44.09 |
| LibreRAFTl-flow-kitti.pt | raft-kitti.pth | 0.0 | 0.0 | 0.0 | 43.69 |

## Upstream demo frames (Sintel), 9 pairs, 1024x436 (exercises padding: 436 pads to 440)

| Weights | Upstream file | Max abs diff (px) | Mean abs diff (px) | Mean EPE between (px) | Mean flow magnitude (px) |
|---|---|---|---|---|---|
| LibreRAFTl-flow.pt | raft-things.pth | 0.0 | 0.0 | 0.0 | 4.94 |
| LibreRAFTs-flow.pt | raft-small.pth | 0.0 | 0.0 | 0.0 | 5.04 |
| LibreRAFTl-flow-sintel.pt | raft-sintel.pth | 0.0 | 0.0 | 0.0 | 4.89 |
| LibreRAFTl-flow-kitti.pt | raft-kitti.pth | 0.0 | 0.0 | 0.0 | 5.69 |

The outputs are bit-identical: the vendored code keeps upstream's computation
and the converter only strips the `module.` prefix.
