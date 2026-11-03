"""NeuFlow v2 model code vendored from neufieldrobotics/NeuFlow_v2.

Upstream: https://github.com/neufieldrobotics/NeuFlow_v2
Commit:   204b5e3744461d90303b9ff82caa7a1bb56a2ca2 (2024-09-01)
Licence:  Apache-2.0, Copyright (c) 2024 Zhiyong Zhang, Aniket Gupta,
          Huaizu Jiang, Hanumant Singh. Full text in ``LICENSE``.

Only the inference model (``NeuFlow/*.py``) is vendored; training, datasets,
evaluation and demo scripts are not. Changes from upstream:

- package-relative imports (``from . import x`` instead of ``from NeuFlow``);
- ``neuflow.py``: the ``huggingface_hub.PyTorchModelHubMixin`` base is
  dropped (plain ``torch.nn.Module``), and the input ``/ 255`` is done out of
  place instead of mutating the caller's tensors;
- ``corr.py``: the commented-out ``RAFTCorrBlock`` is removed;
- ``utils.py``: the unused ``flow_warp`` helper (which called
  ``coords_grid`` with the wrong arity) is removed.

Learned parameters, module names and the numerical path are unchanged, so
upstream checkpoints load with ``strict=True``.
"""
