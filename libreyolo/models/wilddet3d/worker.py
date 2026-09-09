"""Private image-inference worker for the separately installed runtime.

Run as a script, not as an importable application API. Compatibility here
implements serial scheduling using PyTorch's stream interface. It does not
reimplement any model, tracker, preprocessing, or postprocessing algorithm.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.machinery
import importlib.util
import json
import sys
import types
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Keep Python and native library diagnostics out of the protocol responses.
_protocol = sys.stdout
sys.stdout = sys.stderr


def _warning(message, category, filename, lineno, file=None, line=None):
    # Warnings remain visible, without printing implementation source lines.
    print(f"{category.__name__}: {message}", file=sys.stderr, flush=True)


warnings.showwarning = _warning

from libreyolo.models.wilddet3d.runtime import _PREFIX, decode_array, encode_array


def _disable_unavailable_tracking_kernel():
    """Leave the unused optional tracker operation unavailable on Mac/CPU."""
    if importlib.util.find_spec("triton") is not None:
        return
    name = "sam3.model.edt"
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)

    def unavailable(*args, **kwargs):
        raise NotImplementedError(
            "The optional SAM tracking distance transform requires Triton. "
            "This worker supports WildDet3D image inference only."
        )

    module.edt_triton = unavailable
    sys.modules[name] = module


def _serial_scheduling(torch, device):
    """Execute branches in order on the single CPU execution queue.

    This worker is private and owns its torch module. CUDA tensor operators,
    device availability, and numeric computations are not replaced.
    """

    def synchronize(*args, **kwargs):
        return None

    class SerialStream:
        def __init__(self, *args, **kwargs):
            self.device = device

        def wait_stream(self, other):
            synchronize()

        def synchronize(self):
            synchronize()

        def query(self):
            synchronize()
            return True

    torch.cuda.Stream = SerialStream
    torch.cuda.current_stream = lambda *args, **kwargs: SerialStream()
    torch.cuda.stream = lambda stream: contextlib.nullcontext(stream)
    torch.cuda.synchronize = synchronize


def _predict(runtime, predictor, torch, device, request):
    import numpy as np

    image = decode_array(request["image"])
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("The runtime requires an RGB uint8 image.")
    depth = decode_array(request["depth"]) if request["depth"] is not None else None
    data = runtime.preprocess(
        image.astype(np.float32),
        np.asarray(request["intrinsics"], dtype=np.float32),
        depth=depth,
    )
    call = {
        "images": data["images"].to(device),
        "intrinsics": data["intrinsics"].to(device)[None],
        "input_hw": [data["input_hw"]],
        "original_hw": [data["original_hw"]],
        "padding": [data["padding"]],
        **request["prompt"],
    }
    if depth is not None:
        call["depth_gt"] = data["depth_gt"].to(device)
    with torch.inference_mode():
        outputs = predictor(**call)
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 7:
        raise ValueError("Expected seven WildDet3D output fields.")
    return [
        [
            encode_array(torch.as_tensor(value).detach().cpu().float().numpy())
            for value in field
        ]
        for field in outputs[:6]
    ]


def main():
    import torch

    runtime = predictor = device = None
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            action = request["action"]
            if action == "load":
                if predictor is not None:
                    raise ValueError("The worker already has a model.")
                config = request["config"]
                device = torch.device(config["device"])
                if device.type not in {"cpu", "cuda"}:
                    raise ValueError("Unsupported runtime device.")
                if device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError(
                        "CUDA is unavailable in the runtime interpreter."
                    )
                if device.type != "cuda":
                    _disable_unavailable_tracking_kernel()
                runtime = importlib.import_module("wilddet3d")
                if device.type != "cuda":
                    _serial_scheduling(torch, device)
                predictor = runtime.build_model(**config)
                reply = {"id": request["id"], "ready": True}
            elif action == "predict" and predictor is not None:
                reply = {
                    "id": request["id"],
                    "outputs": _predict(runtime, predictor, torch, device, request),
                }
            else:
                raise ValueError("Invalid runtime request or model not loaded.")
        except Exception as exc:  # noqa: BLE001 - serialize model failures across the process boundary
            frames = []
            tb = exc.__traceback__
            while tb:
                frames.append(
                    f"{tb.tb_frame.f_code.co_filename}:{tb.tb_lineno} {tb.tb_frame.f_code.co_name}"
                )
                tb = tb.tb_next
            reply = {
                "id": request.get("id"),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc).splitlines()[0],
                    "frames": frames,
                },
            }
        _protocol.write(_PREFIX + json.dumps(reply, allow_nan=False) + "\n")
        _protocol.flush()


if __name__ == "__main__":
    main()
