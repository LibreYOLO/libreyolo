"""Vendored Video Depth Anything modules (Apache-2.0).

Source: https://github.com/DepthAnything/Video-Depth-Anything at commit
4f5ae23172ba60fd7bc11ef671cca678842c7072 (2025-10-07). Copyright (2025)
Bytedance Ltd. and/or its affiliates. See LICENSE in this directory.

Files and changes from upstream, all inference-neutral:

- ``dpt_temporal.py`` (``video_depth_anything/dpt_temporal.py``): imports
  ``DPTHead`` from the Depth Anything V2 vendor package, whose ``DPTHead``,
  DINOv2 encoder and transforms are byte-identical to the copies upstream
  ships; ``EasyDict`` replaced by a plain ``dict`` (it only feeds ``**kwargs``).
- ``motion_module/motion_module.py`` and ``motion_module/attention.py``
  (upstream ``video_depth_anything/motion_module/``): the import-time
  ``print("xFormers not available")`` is a debug log. ``motion_module.py``
  originates in AnimateDiff (github.com/guoyww/AnimateDiff, Apache-2.0) and
  ``attention.py`` in Hugging Face diffusers (Apache-2.0), as their headers say.
- ``util.py`` (``utils/util.py``): verbatim; the least-squares scale/shift and
  the overlap interpolation used to stitch inference windows together.

Training code, losses, benchmark scripts and the experimental streaming model
are not vendored.
"""
