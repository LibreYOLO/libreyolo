"""Vendored MEMFOF inference code (msu-video-group/memfof, BSD-3-Clause).

Source: https://github.com/msu-video-group/memfof at commit
a51de9fc59c6fe20ba08e079372c7b583d58a712 (2025-12-12). See LICENSE in this
directory. Changes from upstream, all inference-neutral:

- package-relative imports;
- ``model.py``: dropped the ``huggingface_hub.PyTorchModelHubMixin`` base and
  defaulted ``backbone_weights`` to ``None`` so construction never downloads
  torchvision ImageNet weights (the converted checkpoint overwrites every
  tensor anyway);
- ``gma.py``: the four ``einops.rearrange`` calls are written as
  ``reshape``/``permute`` so the family needs no extra dependency;
- ``utils.py``: kept only ``InputPadder``, ``bilinear_sampler`` and
  ``coords_grid``.

Training code (Lightning module, datasets, losses, augmentors) is not vendored.
``gma.py`` (``Attention``, ``Aggregate``) originates in GMA
(github.com/zacjiang/GMA, ``core/gma.py``, WTFPL; text in ``LICENSE-GMA``).
MEMFOF credits VideoFlow, whose copy is GMA's file with the position-embedding
branches commented out; VideoFlow ships no licence, so the grant relied on is
GMA's.
``update.py``'s ``GMAUpdateBlock`` likewise follows GMA's ``core/update.py``;
``corr.py`` follows RAFT/SEA-RAFT (BSD-3-Clause, see LICENSE).
"""

from .model import MEMFOF

__all__ = ["MEMFOF"]
