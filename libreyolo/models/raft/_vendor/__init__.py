"""Vendored RAFT model code (BSD-3-Clause).

Source: https://github.com/princeton-vl/RAFT at commit 2888e15 (``core/``).
Copyright (c) 2020, princeton-vl. Licensed under the BSD 3-Clause License,
see ``LICENSE`` in this directory.

``update.py`` and ``extractor.py`` are verbatim. Changes elsewhere, none of
which alter the computation of the default (non-alternate-corr) path:

- package-relative imports;
- ``corr.py``: the ``alt_cuda_corr`` CUDA extension and ``AlternateCorrBlock``
  are removed (``raft.py`` raises if ``alternate_corr`` is requested);
- ``raft.py``: the deprecated ``torch.cuda.amp.autocast`` alias is replaced by
  an equivalent helper;
- ``utils/utils.py``: ``forward_interpolate`` (scipy, training warm start only)
  is removed;
- ``torch.meshgrid`` calls pass ``indexing='ij'``, the default they already
  used, to silence the PyTorch deprecation warning.
"""
