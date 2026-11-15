# NeuFlow v2 parity

Measured 2026-11-03 on an NVIDIA GeForce RTX 3090, torch 2.11.0+cu128, fp32,
TF32 off, BatchNorm unfused, with `weights/parity_neuflowv2.py` against
upstream `neufieldrobotics/NeuFlow_v2` at commit
`204b5e3744461d90303b9ff82caa7a1bb56a2ca2`.

Input: the 24-frame, 640x360 LibreFlow test clip, 23 consecutive pairs.
Both sides get the same tensors (LibreYOLO preprocess: RGB, stretch-resized to
the 432x768 canvas). "Canvas" compares the raw `(2, 432, 768)` network output;
"e2e" compares `LibreYOLO(ckpt).predict(frames)` `Results.flow` on the 640x360
original canvas with the upstream output resized and rescaled the same way.
EPE is the mean L2 norm of the vector difference between the two fields.

| Checkpoint | Upstream file | Pairs | Canvas max / mean abs (px) | Canvas EPE (px) | e2e max / mean abs (px) | e2e EPE (px) | Upstream mean flow magnitude (px) |
|---|---|---|---|---|---|---|---|
| `LibreNeuFlowv2s-flow.pt` | `neuflow_mixed.pth` | 23 | 0.0 / 0.0 | 0.0 | 0.0 / 0.0 | 0.0 | 36.55 |
| `LibreNeuFlowv2s-flow-sintel.pt` | `neuflow_sintel.pth` | 23 | 0.0 / 0.0 | 0.0 | 0.0 / 0.0 | 0.0 | 35.90 |
| `LibreNeuFlowv2s-flow-things.pt` | `neuflow_things.pth` | 23 | 0.0 / 0.0 | 0.0 | 0.0 / 0.0 | 0.0 | 36.73 |

The port is bit-exact. The same holds on the nine 1280x720 pairs of upstream
`test_images/` (mixed checkpoint: max abs diff 0.0 px).

Upstream `infer.py` differs from this contract in two ways that are not part
of the parity claim: it feeds OpenCV BGR frames (training and evaluation read
RGB through PIL; LibreYOLO feeds RGB), and it fuses Conv and BatchNorm and runs
in fp16. Neither changes the learned parameters.

## ONNX pair graph

A standalone `torch.onnx.export` (opset 17, inputs `image1` and `image2`
`(1, 3, 432, 768)`, output `flow` `(1, 2, 432, 768)`, iterations 1 at 1/16 and
8 at 1/8 baked in) checks clean and contains only standard ops (Conv,
BatchNormalization, LayerNormalization, MatMul, Softmax, GridSample, Resize,
AveragePool, elementwise). onnxruntime 1.31.1 CPU vs PyTorch CPU on the 23 clip
pairs: max abs 0.216 px, mean abs 3.1e-4 px, EPE 5.0e-4 px. The graph is
66 MB because the per-pixel correlation-lookup offsets are folded in as
constants. The exporter route is not wired: flow export waits for the ADR 0032
graph contract in `libreyolo/export/`.
