# Compiled training (`compile`)

`train(compile=...)` compiles the training network with `torch.compile`
(Inductor). It is off by default.

```python
model.train(data="data.yaml", compile=True)
```

```bash
libreyolo train model=LibreRFDETRn.pt data=data.yaml compile=true
libreyolo train --model LibreYOLO9t.pt --data data.yaml --compile max-autotune-no-cudagraphs
```

| Value | Meaning |
| --- | --- |
| `False` | Eager training (default) |
| `True`, `"default"` | Inductor's default mode |
| `"reduce-overhead"` | Also replays CUDA graphs |
| `"max-autotune"` | Autotunes kernels and replays CUDA graphs |
| `"max-autotune-no-cudagraphs"` | Autotunes kernels, no CUDA graphs |

Other values raise `ValueError` (a `config_type_error` on the CLI).

## What is compiled

The network forward and its backward, at the same network/loss boundary as
CUDA graph capture. Target assignment, the loss, the optimizer step, EMA and
the learning-rate schedule stay eager. The model object is not wrapped:
validation, EMA, checkpoints (no `_orig_mod.` prefixes), resume and export
are unchanged, and a checkpoint trained with `compile=True` loads anywhere.

Supported: single-GPU CUDA training of families with a training boundary,
including both flagships. YOLO9's default PGI auxiliary branch is compiled.
RF-DETR detection is compiled with dynamic shapes when per-batch multi-scale
is on (the default), so its sizes share one compilation; RF-DETR
segmentation, pose, OBB and classification train eager. CPU and MPS
(Inductor training measured 15 to 60 times slower than eager on CPU),
distributed training and distillation train eager after one warning.

## CUDA graphs

`reduce-overhead`, `max-autotune`, or `compile=True` together with
`cuda_graph=True`, replay the compiled kernels as CUDA graphs. The separate
eager capture manager is not used in a compiled run. Replay keeps gradients
in the graph's memory, so it is turned off, with a warning, when gradient
accumulation is on: RF-DETR's defaults (`batch=4`, `nbs=16`) accumulate four
batches; pass `nbs` equal to `batch` to replay graphs, which changes the
effective batch.

## Measured (RTX 4090, torch 2.14, steady-state epochs, 256 images)

| Workload | Eager | `compile` | `cuda_graph` | Compile time |
| --- | --- | --- | --- | --- |
| RF-DETR nano, 384 px, batch 4 | 8.86 s | 6.15 s (1.44x) | 5.90 s (1.50x) | about 5 min |
| RF-DETR nano, 384 px, batch 16 | 4.74 s | 4.39 s (1.08x) | | about 4 min |
| YOLO9-t, 640 px, batch 16 | 5.15 s | 5.22 s (1.0x) | | about 4 min |

The gain is largest where the GPU waits on kernel launches (small batches);
`cuda_graph=True` removes the same launch cost without compiling. With
RF-DETR's default per-batch multi-scale, dynamic-shape compilation takes far
longer (over 18 minutes on this host), so prefer `multi_scale=False` or
`cuda_graph=True` when you want compiled speed. Accuracy was not measured.

## Cost and failures

The first steps, and each new input shape, include compilation time
(minutes for large networks). A compiler failure logs a warning and the run
continues eagerly; errors while running compiled kernels, such as out of
memory, stop the run. Compiled kernels round differently from eager ones,
so losses are close to, not identical with, an eager run. Compare throughput,
memory and validation metrics on your own data before relying on it.

`tests/e2e/test_training_compile.py` checks that compilation engages on
CUDA for both flagships; speed and accuracy are not asserted there.
