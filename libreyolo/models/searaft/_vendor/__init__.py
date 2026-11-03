"""Vendored SEA-RAFT modules (BSD-3-Clause).

Source: https://github.com/princeton-vl/SEA-RAFT (``core/``), commit
9137517ba24e628442aec097d3afe71d03503b75.
Copyright (c) 2024, Princeton Vision & Learning Lab. Licensed under the
BSD 3-Clause License, see ``LICENSE`` in this directory.

Inference subset only. Changes from upstream, none of which alter the learned
parameters or the outputs:

- ``raft.py``: the Hugging Face ``PyTorchModelHubMixin`` base, the training
  loss branch (``flow_gt`` / mixture-of-Laplace NLL) and the intermediate
  prediction lists are dropped. Normalisation to [-1, 1] and the replicate
  padding to a multiple of 8 moved to the LibreYOLO preprocess (they commute,
  replicate padding copies values), so ``forward`` takes padded, normalised
  frames and returns the final flow and the Laplace mixture parameters at the
  padded resolution.
- ``extractor.py``: the torchvision ImageNet initialisation (a download used
  only before training) is removed; the module tree and parameter names are
  unchanged.
- ``corr.py``: only ``CorrBlock`` is kept; the unused sparse helpers, the
  commented-out alternatives and the optional ``alt_cuda_corr`` import are
  dropped.
- ``update.py``, ``layer.py``, ``utils.py``: unused classes and helpers
  (gradient clipping, ``FlowHead``, dataset and reprojection utilities) are
  dropped; ``torch.meshgrid`` gets an explicit ``indexing="ij"``, its previous
  default.
"""
