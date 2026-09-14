"""Process boundary for the optional 3D-MOOD runtime environment."""

from pathlib import Path

import numpy as np

from ..runtime_worker import (
    _PREFIX,
    decode_array,
    encode_array,
)
from ..runtime_worker import (
    RuntimeWorker as _RuntimeWorker,
)


class RuntimeWorker(_RuntimeWorker):
    """A persistent 3D-MOOD worker using the shared typed-array protocol."""

    def __init__(self, **kwargs):
        super().__init__(
            worker_path=Path(__file__).with_name("worker.py"),
            runtime_name="3D-MOOD",
            **kwargs,
        )

    def predict(self, image, intrinsics, prompt, depth=None):
        if depth is not None:
            raise ValueError("3D-MOOD predicts depth internally and takes no depth input.")
        reply = self._rpc(
            {
                "action": "predict",
                "image": encode_array(image),
                "intrinsics": np.asarray(intrinsics).tolist(),
                "text": prompt["text"],
            }
        )
        return tuple(decode_array(value) for value in reply["outputs"]) + (None,)


__all__ = ["_PREFIX", "RuntimeWorker", "decode_array", "encode_array"]
