"""GTR instance segmentation (``GTRSeg``) constants.

Source: Intellindust-AI-Lab/GTR, MIT, revision
782e737efe2e6437ac537fbdcee089673d3376c1, configs/seg/gtrseg_base.yml. The
seg model is the detector plus the per-query mask head already vendored in
``segmentation_head.py``; masks come out at a quarter of the input resolution
as logits, thresholded at zero after bilinear upsampling.
"""

SEG_MASK_DOWNSAMPLE_RATIO = 4
SEG_HEAD_PREFIX = "decoder.decoder.segmentation_head."

# Pinned ``LibreYOLO/LibreGTR{size}-seg`` revisions.
# TODO(upload): fill each SHA after the Hugging Face upload; ``None`` disables
# auto-download for that size until then.
SEG_HF_REVISIONS: dict[str, str | None] = {
    "s": None,
    "m": None,
    "l": None,
    "x": None,
}


def is_seg_state_dict(state_dict) -> bool:
    return any(key.startswith(SEG_HEAD_PREFIX) for key in state_dict)
