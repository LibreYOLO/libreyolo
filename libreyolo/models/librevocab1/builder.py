"""LibreVocab1 — orchestrate the SAM3 + CRADIOv4 vision-encoder swap.

Mirrors ``mranzinger/sam3-radio``'s ``demo_sam3_radio.py`` flow:

    1. Build SAM3 image model (text encoder, fusion encoder, decoder, mask head).
    2. Load CRADIOv4 from torch.hub with the ``sam3`` adaptor.
    3. Replace SAM3's ViT trunk with a ``RADIO_Adaptor`` that wraps CRADIOv4.
    4. Hand back the resulting model + a ``Sam3Processor`` configured for the
       16-vs-14 patch difference.

This module is the *only* code we need to write to compose Path A — everything
else is vendored sam3 + upstream CRADIOv4 hub code.

LICENSE NOTE: the vendored ``sam3/`` subtree is under Meta's SAM License (see
``sam3/LICENSE_SAM3``). CRADIOv4 weights are NVIDIA Open Model License. This
file (the orchestration glue) is MIT, like the rest of LibreYOLO.
"""

from __future__ import annotations

from typing import Optional

import torch

from .sam3.model_builder import build_sam3_image_model
from .sam3.sam3_radio_utils import (
    create_sam3_radio_processor,
    load_radio_model,
    replace_sam3_encoder,
)


def build_librevocab1(
    radio_model_version: str = "c-radio_v4-h",
    sam3_checkpoint_path: Optional[str] = None,
    load_sam3_from_hf: bool = True,
    confidence_threshold: float = 0.5,
    vitdet_window_size: Optional[int] = None,
    device: str = "cuda",
):
    """Build a LibreVocab1 detector: SAM3 stack + CRADIOv4 vision encoder.

    Args:
        radio_model_version: CRADIOv4 variant. Either ``c-radio_v4-h`` or
            ``c-radio_v4-so400m``, or a path to a local ``.pth.tar`` file.
        sam3_checkpoint_path: Local path to a SAM3 checkpoint. If ``None``
            and ``load_sam3_from_hf=True``, the checkpoint is downloaded from
            the SAM3 HuggingFace repo.
        load_sam3_from_hf: Whether to fetch SAM3 weights from HF when
            ``sam3_checkpoint_path`` is ``None``.
        confidence_threshold: Score threshold for the ``Sam3Processor``.
        vitdet_window_size: If set, run CRADIOv4 in ViTDet mode with this
            window size (must be ≤ patch grid size). Recommended: ``8`` or
            ``16`` for high-resolution efficiency.
        device: Device to place the resulting model on.

    Returns:
        Tuple of ``(sam3_model, processor)``. ``sam3_model`` is a
        ``Sam3Image`` whose vision trunk is the wrapped CRADIOv4. ``processor``
        is a ``Sam3Processor`` configured for the 16-pixel patch CRADIOv4 uses.
    """
    # 1. Locate the BPE vocab that ships inside the vendored sam3 package.
    from importlib.resources import files as _pkg_files
    bpe_path = str(
        _pkg_files("libreyolo.models.librevocab1.sam3").joinpath(
            "assets/bpe_simple_vocab_16e6.txt.gz"
        )
    )

    # 2. Build the SAM3 image stack.
    sam3_model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=sam3_checkpoint_path,
        load_from_HF=load_sam3_from_hf and sam3_checkpoint_path is None,
        eval_mode=True,
        device=device,
    )

    # 3. Load CRADIOv4 with the sam3 adaptor head.
    radio_model = load_radio_model(
        radio_model_version,
        device=device,
        vitdet=vitdet_window_size,
    )

    # 4. Swap CRADIOv4 in for SAM3's ViT trunk.
    sam3_model = replace_sam3_encoder(sam3_model, radio_model, device=device)

    # 5. Build the processor with CRADIOv4-aware resolution defaults.
    processor = create_sam3_radio_processor(
        sam3_model,
        confidence_threshold=confidence_threshold,
        resolution=None,  # uses DEFAULT_RADIO_RESOLUTION (1152) from sam3_radio_utils
        device=device,
    )

    return sam3_model, processor
