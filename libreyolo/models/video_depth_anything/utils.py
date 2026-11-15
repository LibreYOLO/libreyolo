"""Preprocessing and window stitching for Video Depth Anything.

``stitch_video_depth`` is a streaming rewrite of upstream
``VideoDepthAnything.infer_video_depth`` (Apache-2.0, Bytedance Ltd. and/or
its affiliates; see ``_vendor/__init__.py`` for the pinned commit). Upstream
reads the whole video, runs 32-frame windows that overlap by 10 frames, and
then aligns every window to the previous one with a least-squares scale and
shift and a linear cross-fade over the overlap. The arithmetic here is the
same, in the same order and dtype, but frames are read lazily and a depth map
is released as soon as no later window can change it, so memory stays bounded
by one window instead of the whole clip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..depth_anything.utils import preprocess_numpy
from ._vendor.util import compute_scale_and_shift, get_interpolate_frames
from .nn import INFER_LEN, INTERP_LEN, KEYFRAMES, OVERLAP

SEQUENCE_MIN_FRAMES_MESSAGE = (
    "Video Depth Anything needs at least one frame: pass an image, a list of "
    "frames, a directory of frames, a video file, or a live source with "
    "stream=True."
)


def check_not_metric(path) -> None:
    """Reject an upstream *metric* checkpoint by its file name.

    Metric checkpoints (``metric_video_depth_anything_vit*.pth``) have exactly
    the relative checkpoints' tensors but predict depth in metres and are
    stitched without the scale/shift fit, so they would load silently and be
    reported as relative inverse depth. Tensors cannot tell them apart; the
    upstream file name can. Metric depth is not part of this family yet.
    """
    if "metric" in Path(str(path)).name.lower():
        raise ValueError(
            f"{Path(str(path)).name} looks like a metric Video Depth Anything "
            "checkpoint. LibreYOLO's video_depth_anything family is relative "
            "inverse depth only; the metric weights would load but their "
            "metres would be reported as inverse depth."
        )


def effective_input_size(height: int, width: int, input_size: int) -> int:
    """Upstream's network size for a frame of ``height`` x ``width``.

    Frames wider than 16:9 run at a proportionally smaller size, rounded to the
    DINOv2 patch (upstream ``infer_video_depth``: "we recommend to process
    video with ratio smaller than 16:9 due to memory limitation").
    """
    ratio = max(height, width) / min(height, width)
    if ratio > 1.78:
        input_size = int(input_size * 1.777 / ratio)
        input_size = round(input_size / 14) * 14
    return int(input_size)


def preprocess_frame(img_rgb_hwc: np.ndarray, input_size: int) -> np.ndarray:
    """RGB uint8 frame to a ``(3, h, w)`` float32 array in ``[0, 1]``.

    Keep-aspect lower-bound resize to a multiple of 14 with cubic
    interpolation, as upstream. ImageNet normalization happens in the network.
    """
    height, width = img_rgb_hwc.shape[:2]
    chw, _ = preprocess_numpy(img_rgb_hwc, effective_input_size(height, width, input_size))
    return chw


def _clamp(depth: np.ndarray) -> np.ndarray:
    depth[depth < 0] = 0
    return depth


def stitch_video_depth(
    frames: Iterable[Tuple[torch.Tensor, Tuple[int, int]]],
    run_window: Callable[[torch.Tensor, Tuple[int, int]], Sequence[np.ndarray]],
    metric: bool = False,
) -> Iterator[np.ndarray]:
    """Yield one depth map per input frame, aligned across the whole sequence.

    Args:
        frames: ``(tensor, (orig_w, orig_h))`` per frame, in order, where
            ``tensor`` is the ``(1, 3, h, w)`` preprocessed frame. All frames
            must share one size.
        run_window: runs the network on a ``(1, 32, 3, h, w)`` window and
            returns its 32 depth maps resized to the original ``(orig_h,
            orig_w)`` canvas as float32 arrays.
        metric: metric checkpoints skip the scale/shift fit (upstream sets
            ``scale, shift = 1.0, 0.0``); the cross-fade still applies.

    Every window holds 32 frames and starts 22 frames after the previous one.
    Its first 10 slots carry keyframes of the previous window (upstream
    ``KEYFRAMES``), the sequence end is padded with copies of the last frame,
    and window ``k > 0`` is aligned to the sequence by fitting its first two
    slots to the reference depth of frame 0 and of the previous window's slot
    12. Depth map ``t`` is final, and yielded, once the window after it no
    longer overlaps it: the stream lags the input by at most one window.
    """
    step = INFER_LEN - OVERLAP
    align_len = OVERLAP - INTERP_LEN
    kf_align_list = list(KEYFRAMES[:align_len])

    iterator = iter(frames)
    tensors: dict = {}
    count = 0
    exhausted = False
    shape = None
    canvas = None

    aligned: List[np.ndarray] = []  # aligned depth of frames base .. base + len - 1
    base = 0
    ref_align: List[np.ndarray] = []
    pre_input: Optional[torch.Tensor] = None
    window_index = 0

    while True:
        start = window_index * step
        while not exhausted and count < start + INFER_LEN:
            try:
                tensor, original_size = next(iterator)
            except StopIteration:
                exhausted = True
                break
            if shape is None:
                shape, canvas = tuple(tensor.shape), tuple(original_size)
            elif tuple(tensor.shape) != shape or tuple(original_size) != canvas:
                raise ValueError(
                    "Frames of one Video Depth Anything sequence must share a "
                    f"size; frame {count} is {tuple(original_size)} -> "
                    f"{tuple(tensor.shape)}, frame 0 is {canvas} -> {shape}."
                )
            tensors[count] = tensor
            count += 1
        if count == 0:
            raise ValueError(SEQUENCE_MIN_FRAMES_MESSAGE)
        if start >= count:
            break

        window = torch.cat(
            [tensors[min(start + i, count - 1)] for i in range(INFER_LEN)], dim=0
        ).unsqueeze(0)
        if pre_input is not None:
            window[:, :OVERLAP, ...] = pre_input[:, list(KEYFRAMES), ...]
        depth = list(run_window(window, canvas))
        pre_input = window

        if window_index == 0:
            aligned = depth[:INFER_LEN]
            # Copies: the yielded maps belong to the caller from then on.
            ref_align = [depth[kf_id].copy() for kf_id in kf_align_list]
        else:
            curr_align = [depth[i] for i in range(len(kf_align_list))]
            if metric:
                scale, shift = 1.0, 0.0
            else:
                scale, shift = compute_scale_and_shift(
                    np.concatenate(curr_align),
                    np.concatenate(ref_align),
                    np.concatenate(np.ones_like(ref_align) == 1),
                )
            pre_depth_list = aligned[-INTERP_LEN:]
            post_depth_list = [
                _clamp(depth[i] * scale + shift) for i in range(align_len, OVERLAP)
            ]
            aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)
            for i in range(OVERLAP, INFER_LEN):
                aligned.append(_clamp(depth[i] * scale + shift))
            ref_align = ref_align[:1]
            for kf_id in kf_align_list[1:]:
                ref_align.append(_clamp(depth[kf_id] * scale + shift))

        window_index += 1
        # The next window rewrites the last INTERP_LEN aligned maps; every
        # frame before those is final. Frames past the input are padding.
        final_end = min(base + len(aligned) - INTERP_LEN, count)
        while base < final_end:
            yield aligned.pop(0)
            base += 1
        for index in [i for i in tensors if i < window_index * step]:
            del tensors[index]

    while base < count:
        yield aligned.pop(0)
        base += 1


__all__ = [
    "SEQUENCE_MIN_FRAMES_MESSAGE",
    "check_not_metric",
    "effective_input_size",
    "preprocess_frame",
    "stitch_video_depth",
]
