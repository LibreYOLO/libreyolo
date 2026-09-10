"""Typed-array process boundary for the optional DetAny3D runtime."""

from pathlib import Path

from ..runtime_worker import RuntimeWorker as _RuntimeWorker
from ..runtime_worker import decode_array, encode_array


class RuntimeWorker(_RuntimeWorker):
    def __init__(self, **kwargs):
        super().__init__(
            worker_path=Path(__file__).with_name("worker.py"),
            runtime_name="DetAny3D",
            **kwargs,
        )

    def _rpc(self, request):
        reply = super()._rpc(request)
        if request.get("action") == "load":
            self.device = reply["device"]
        return reply

    def predict(self, image, prompt):
        reply = self._rpc(
            {"action": "predict", "image": encode_array(image), "prompt": prompt}
        )
        return {
            key: decode_array(value) for key, value in reply["arrays"].items()
        }, reply["labels"]
