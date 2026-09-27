"""Process boundary for the optional WildDet3D runtime environment."""

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
    """A persistent WildDet3D worker using the shared typed-array protocol."""

    def __init__(self, **kwargs):
        super().__init__(
            worker_path=Path(__file__).with_name("worker.py"),
            runtime_name="WildDet3D",
            **kwargs,
        )

    def predict(self, image, intrinsics, prompt, depth=None):
        reply = self._rpc(
            {
                "action": "predict",
                "image": encode_array(image),
                "intrinsics": np.asarray(intrinsics).tolist(),
                "prompt": prompt,
                "depth": encode_array(depth) if depth is not None else None,
            }
        )
        return tuple(
            [decode_array(value) for value in field] for field in reply["outputs"]
        ) + (None,)

__all__ = ["_PREFIX", "RuntimeWorker", "decode_array", "encode_array"]
