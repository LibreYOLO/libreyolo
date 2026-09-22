# Compiling the training network

`compile` is an opt-in training argument shared by the Python API, training
YAMLs and CLI. The default is `False`. Compilation uses PyTorch Inductor and
does not change the learning rate, batch size, precision or augmentation recipe.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRn.pt")
model.train(data="data.yaml", compile=True)
```

```bash
libreyolo train model=LibreYOLO9t.pt data=data.yaml compile=default
libreyolo train --model LibreRFDETRn.pt --data data.yaml --compile max-autotune-no-cudagraphs
```

| Value | Behavior |
| --- | --- |
| `False` / `false` | Existing eager training |
| `True` / `true` / `default` | Inductor's default mode |
| `reduce-overhead` | Request Inductor CUDA graphs on eligible runs |
| `max-autotune` | Autotune supported operations and request Inductor CUDA graphs |
| `max-autotune-no-cudagraphs` | Autotune without requesting CUDA graphs |

Unknown values raise `ValueError` in Python and a structured configuration
error on the CLI. `--compile` takes a value; it is not a bare boolean switch.
`--help-json`, `--dry-run --json` and saved training configs expose the option.

## Network and loss boundary

The compiled callable runs the network forward. PyTorch compiles its autograd
backward when supported. Target preparation, Hungarian matching or dense
assignment, loss calculation, optimizer steps, gradient clipping, EMA and the
learning-rate schedule remain eager.

The trainer retains the original model object. It does not replace or rebind
the model's forward, parameters or state-dict keys. Validation uses the ordinary
model or EMA copy; checkpoints contain ordinary model tensors and optimizer
state, with no compiled wrapper prefixes or serialized compiled program. A
checkpoint can be resumed with compilation either enabled or disabled.

The initial supported boundary covers YOLO9 detection with its ordinary dense
head and RF-DETR detection. YOLO9 PGI auxiliary branches, derived YOLO9 heads,
other tasks/families, DDP, distillation, LoRA, QAT, and devices other than CPU or
CUDA log a warning and run eager. These limits describe the implemented
network/loss split, not restrictions imposed by PyTorch in general.

RF-DETR's optional native deformable-attention providers decline compiler
tracing. Compiled regions use the portable attention path and the trainer logs
that choice. A native-kernel eager run can therefore be faster than a compiled
run on a particular GPU.

## Combining compilation and CUDA graphs

`compile=True, cuda_graph=True` requests Inductor's `triton.cudagraphs` option.
The trainer never constructs the separate eager CUDA-graph manager for a run
that requests compilation. The chosen compiler mode's other options, including
autotuning, remain in effect.

Compiler CUDA graphs require a supported single-CUDA training run with one
optimizer step per batch. RF-DETR's default `nbs=16` accumulates four batches
when `batch=4`; explicitly use `nbs=4` for one step per batch:

```python
model.train(data="data.yaml", batch=4, nbs=4, compile=True, cuda_graph=True)
```

```bash
libreyolo train model=LibreRFDETRn.pt data=data.yaml batch=4 nbs=4 compile=true cuda_graph=true
```

Changing `nbs` changes the effective training batch; the trainer never does so
automatically to enable capture. On CPU or with accumulation, compiler CUDA
graphs are disabled with a warning while compilation remains enabled. The same
rule applies to modes that request capture implicitly. If the installed
PyTorch lacks its iteration-marker API, graphs are disabled the same way.

Eligible runs call `torch.compiler.cudagraph_mark_step_begin()` before each
training network forward. Inductor can still decline capture for individual
regions; the startup log reports the requested option, not proof of capture or
speedup. Validation is eager and does not share these training graph buffers.

## Compilation cost, shapes and failure behavior

Compilation and autotuning can make the first steps substantially slower.
Changing batch or image sizes can trigger additional compilation. PyTorch's
default automatic dynamic-shape behavior is retained; the trainer does not
disable multi-scale augmentation or drop partial batches to avoid recompiles.
`fullgraph=False` permits unsupported network regions to remain eager.

If compiler creation or the first compiled forward fails, the trainer logs the
failure and runs eager for the rest of the run. The first forward snapshots
registered tensor buffers and the device's PyTorch RNG state, then restores
them before that eager retry, preserving BatchNorm and dropout behavior.
Out-of-memory errors propagate. After the first successful compiled forward,
later failures, including backward compilation or execution failures, stop the
run: retrying a partially executed training step could corrupt its state. Use
`compile=False` when resuming after such a failure.

Compiled reductions and kernels can change floating-point rounding and RNG
sequences. Bitwise agreement with eager training, a speedup, and matching final
accuracy are not guaranteed. Compare the training loss, validation metrics,
steady-state throughput and memory on the actual device and dataset.

## Validation evidence

CPU unit tests cover option validation and both CLI grammars, network/loss
routing, graph ownership and step markers, unsupported configurations,
first-forward buffer/RNG restoration, optimizer parameter identities, EMA,
checkpoint keys and resume state. A small real CPU Dynamo/AOTAutograd training
test verifies forward/backward and optimizer behavior without requiring a
platform C++ toolchain.

CUDA Inductor execution, CUDA-graph numerical parity, throughput and final
training accuracy require a GPU validation run. CPU routing tests do not
establish those results.

Compiler semantics follow the PyTorch documentation for
[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
and
[`cudagraph_mark_step_begin`](https://docs.pytorch.org/docs/stable/generated/torch.compiler.cudagraph_mark_step_begin.html).
