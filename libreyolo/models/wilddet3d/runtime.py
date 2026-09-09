"""Local process boundary for an optional external runtime environment.

CPU scheduling compatibility stays inside its worker. The calling process's
torch module, streams and allocator are never patched. Supplying a separate
CUDA interpreter also uses this boundary without the scheduling shim.
Images and tensors cross private pipes as JSON plus typed array bytes, not
pickle. A worker keeps its loaded model until close() or owner collection.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import weakref
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
_PREFIX = "LIBREYOLO_WILDDET3D "
_ALLOWED_DTYPES = {"uint8", "float32"}


def encode_array(value):
    array = np.ascontiguousarray(value)
    if str(array.dtype) not in _ALLOWED_DTYPES:
        raise TypeError(f"Unsupported runtime transport dtype: {array.dtype}")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def decode_array(value):
    if value["dtype"] not in _ALLOWED_DTYPES:
        raise ValueError("Invalid runtime tensor dtype.")
    shape = tuple(value["shape"])
    if any(not isinstance(v, int) or v < 0 for v in shape):
        raise ValueError("Invalid runtime tensor shape.")
    data = base64.b64decode(value["data"], validate=True)
    dtype = np.dtype(value["dtype"])
    if len(data) != int(np.prod(shape, dtype=object)) * dtype.itemsize:
        raise ValueError("Runtime tensor byte count does not match its shape.")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def _read_responses(stream, replies):
    try:
        for line in stream:
            if line.startswith(_PREFIX):
                replies.put(json.loads(line[len(_PREFIX) :]))
            elif line.strip():
                logger.info("WildDet3D: %s", line.rstrip())
    except (OSError, ValueError) as exc:
        replies.put({"transport_error": str(exc)})
    finally:
        replies.put({"transport_error": "The WildDet3D worker exited."})
        stream.close()


def _read_logs(stream, logs):
    try:
        for line in stream:
            logs.append(line.rstrip())
            logger.info("WildDet3D: %s", line.rstrip())
    finally:
        stream.close()


def _shutdown(process):
    if process.stdin is not None:
        try:
            process.stdin.close()
        except (OSError, ValueError):
            pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


class RuntimeWorker:
    """One loaded external model, with serialized requests and bounded waits."""

    def __init__(self, *, config, runtime_path=None, runtime_python=None, timeout=600):
        self._lock = threading.Lock()
        self._replies = queue.Queue()
        self._timeout = timeout
        self._request_id = 0
        self._logs = deque(maxlen=40)
        environment = os.environ.copy()
        if runtime_path:
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(None, [str(runtime_path), environment.get("PYTHONPATH")])
            )
        self._process = subprocess.Popen(
            [
                str(runtime_python or sys.executable),
                "-u",
                str(Path(__file__).with_name("worker.py")),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        self._finalizer = weakref.finalize(self, _shutdown, self._process)
        threading.Thread(
            target=_read_responses,
            args=(self._process.stdout, self._replies),
            daemon=True,
        ).start()
        self._log_thread = threading.Thread(
            target=_read_logs,
            args=(self._process.stderr, self._logs),
            daemon=True,
        )
        self._log_thread.start()
        try:
            self._rpc({"action": "load", "config": config})
        except BaseException:
            self.close()
            raise

    def _rpc(self, request):
        with self._lock:
            if not self._finalizer.alive or self._process.poll() is not None:
                raise RuntimeError("The WildDet3D worker is closed.")
            self._request_id += 1
            request["id"] = self._request_id
            try:
                self._process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
                self._process.stdin.flush()
                reply = self._replies.get(timeout=self._timeout)
            except queue.Empty as exc:
                self.close()
                raise RuntimeError("WildDet3D inference worker timed out.") from exc
            except (BrokenPipeError, OSError) as exc:
                self.close()
                raise RuntimeError("Lost the WildDet3D inference worker.") from exc
            except KeyboardInterrupt:
                self.close()
                raise
            if "transport_error" in reply:
                try:
                    self._process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
                self._log_thread.join(timeout=0.5)
                details = "\n".join(self._logs)
                self.close()
                message = reply["transport_error"]
                if details:
                    message += f" Last runtime output:\n{details}"
                raise RuntimeError(message)
            if reply.get("id") != self._request_id:
                self.close()
                raise RuntimeError(
                    "WildDet3D worker response does not match the request."
                )
            if "error" in reply:
                error = RuntimeError(f"WildDet3D runtime: {reply['error']['message']}")
                error.runtime_frames = reply["error"].get("frames", [])
                raise error
            return reply

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

    def close(self):
        self._finalizer()
