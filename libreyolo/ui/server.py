"""Standard-library HTTP server backing ``libreyolo ui``.

Zero third-party dependencies: the page is served from :mod:`libreyolo.ui.page`
and inference reuses the same ``LibreYOLO`` predict path the CLI uses, writing
annotated images to ``runs/detect/predict`` exactly like ``predict --save``.

Endpoints
---------
``GET  /``                page (HTML)
``GET  /api/models``      JSON list of resolvable model names + default
``POST /api/run/new``     start a fresh ``runs/detect/predict`` output dir
``POST /api/infer``       body = raw image bytes, header ``X-Filename``,
                          query ``model`` + ``conf``; returns the rendered
                          (annotated) image as a base64 data URL + box count
``POST /api/open-folder`` open the current results dir in the OS file manager
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .page import INDEX_HTML

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename(name: str) -> str:
    """Reduce an arbitrary upload name to a safe basename for disk."""
    base = Path(name or "image").name
    base = _SAFE_NAME.sub("_", base).strip("._") or "image"
    if "." not in base:
        base += ".jpg"
    return base


def _open_in_file_manager(path: Path) -> bool:
    """Open a directory in the platform's file manager. Returns success."""
    p = str(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(p)  # type: ignore[attr-defined]  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", p], check=False)
        else:
            subprocess.run(["xdg-open", p], check=False)
        return True
    except Exception:
        logger.exception("Failed to open results folder")
        return False


class _StreamLogHandler(logging.Handler):
    """Logging handler that forwards each record's message to a callback.

    Used to stream libreyolo's own log output (weight download progress, save
    paths) into the UI terminal while inference runs in the same thread.
    """

    def __init__(self, emit):
        super().__init__()
        self._emit = emit
        # Mirror the real console format so the UI terminal looks authentic.
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S"
            )
        )

    def emit(self, record):  # noqa: A003 - logging.Handler API
        try:
            self._emit(self.format(record))
        except Exception:
            pass


class _UIState:
    """Per-server state: model cache, current run directory, upload scratch dir."""

    def __init__(self, device: str = "auto"):
        self.device = device
        self._models: dict[str, object] = {}
        self._lock = threading.Lock()  # serialize inference (model not thread-safe)
        self.run_dir: Path | None = None
        self._input_dir = Path(tempfile.mkdtemp(prefix="libreyolo-ui-"))

    def _get_model(self, name: str):
        model = self._models.get(name)
        if model is None:
            from libreyolo import LibreYOLO
            from libreyolo.cli.config import resolve_model_name

            weight = resolve_model_name(name)
            logger.info("Loading model %s (%s)", name, weight)
            model = LibreYOLO(weight, device=self.device)
            self._models[name] = model
        return model

    def new_run(self) -> Path:
        from libreyolo.utils.general import increment_path

        self.run_dir = increment_path(
            Path("runs/detect") / "predict", mkdir=True
        )
        return self.run_dir

    def infer(
        self,
        model_name: str,
        conf: float,
        filename: str,
        data: bytes,
        emit=None,
    ) -> dict:
        """Run inference. If ``emit`` is given, libreyolo log lines (model
        download, save path, ...) are forwarded to it live during the run."""
        safe = _sanitize_filename(filename)
        in_path = self._input_dir / safe
        in_path.write_bytes(data)

        with self._lock:
            if self.run_dir is None:
                self.new_run()
            lg = logging.getLogger("libreyolo")
            handler = _StreamLogHandler(emit) if emit is not None else None
            prev_level = lg.level
            if handler is not None:
                handler.setLevel(logging.INFO)
                if prev_level == logging.NOTSET or prev_level > logging.INFO:
                    lg.setLevel(logging.INFO)
                lg.addHandler(handler)
            try:
                # _get_model triggers weight download on first use; keep it
                # inside the captured block so the download streams to the UI.
                model = self._get_model(model_name)
                result = model(
                    str(in_path),
                    conf=conf,
                    save=True,
                    output_path=str(self.run_dir),
                )
            finally:
                if handler is not None:
                    lg.removeHandler(handler)
                    lg.setLevel(prev_level)

        if isinstance(result, list):
            result = result[0] if result else None
        if result is None:
            raise RuntimeError("inference returned no result")

        saved = getattr(result, "saved_path", None)
        if not saved or not Path(saved).exists():
            raise RuntimeError("annotated image was not saved")

        boxes = getattr(result, "boxes", None)
        count = len(boxes) if boxes is not None else 0
        suffix = Path(saved).suffix.lower().lstrip(".")
        mime = "jpeg" if suffix in ("jpg", "jpeg", "") else suffix
        encoded = base64.b64encode(Path(saved).read_bytes()).decode("ascii")
        return {
            "name": safe,
            "count": int(count),
            "rendered": "data:image/" + mime + ";base64," + encoded,
            "dir": str(self.run_dir),
            "saved": str(saved),
        }

    def open_folder(self) -> dict:
        if self.run_dir is None or not Path(self.run_dir).exists():
            return {"ok": False, "dir": str(self.run_dir) if self.run_dir else None}
        ok = _open_in_file_manager(Path(self.run_dir))
        return {"ok": ok, "dir": str(self.run_dir)}


class _Handler(BaseHTTPRequestHandler):
    state: _UIState  # bound on the subclass created in serve()
    server_version = "LibreYOLO-UI"

    def log_message(self, *args):  # keep the console quiet
        pass

    def _send(self, code: int, body, ctype: str = "application/json") -> None:
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, INDEX_HTML, "text/html; charset=utf-8")
        elif path == "/api/models":
            from libreyolo.cli.config import get_all_cli_names

            names = sorted(get_all_cli_names())
            default = "yolo9-t" if "yolo9-t" in names else (names[0] if names else "")
            self._send(200, {"models": names, "default": default})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path == "/api/run/new":
                self._send(200, {"dir": str(self.state.new_run())})
            elif path == "/api/infer":
                self._handle_infer_stream(qs)
                return
            elif path == "/api/open-folder":
                self._send(200, self.state.open_folder())
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:
            logger.exception("UI request failed: %s", path)
            self._send(500, {"error": str(exc)})

    def _handle_infer_stream(self, qs: dict) -> None:
        """Stream NDJSON: {"type":"log"} lines live, then a final
        {"type":"result"} or {"type":"error"} object.

        Uses a close-delimited streaming body (no Content-Length) so each
        flushed line reaches the browser's fetch ReadableStream immediately.
        """
        length = int(self.headers.get("Content-Length", 0) or 0)
        data = self.rfile.read(length) if length else b""
        model = qs.get("model", ["yolo9-t"])[0]
        try:
            conf = float(qs.get("conf", ["0.25"])[0])
        except ValueError:
            conf = 0.25
        filename = self.headers.get("X-Filename", "image.jpg")

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        write_lock = threading.Lock()

        def write_obj(obj: dict) -> None:
            line = (json.dumps(obj) + "\n").encode("utf-8")
            with write_lock:
                try:
                    self.wfile.write(line)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        if not data:
            write_obj({"type": "error", "error": "empty upload"})
            return

        def emit(msg: str) -> None:
            write_obj({"type": "log", "line": msg})

        try:
            result = self.state.infer(model, conf, filename, data, emit=emit)
            write_obj({"type": "result", **result})
        except Exception as exc:
            logger.exception("UI inference failed")
            write_obj({"type": "error", "error": str(exc)})


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "auto",
    open_browser: bool = True,
) -> tuple[ThreadingHTTPServer, str]:
    """Bind the UI server and (optionally) schedule the browser to open.

    Binding happens eagerly, so an in-use port raises ``OSError`` here and the
    caller can retry on the next port. Returns ``(httpd, url)``; the caller is
    responsible for ``httpd.serve_forever()``.
    """
    state = _UIState(device=device)
    handler = type("BoundUIHandler", (_Handler,), {"state": state})
    httpd = ThreadingHTTPServer((host, port), handler)
    url = "http://%s:%d" % (host, port)
    if open_browser:
        import webbrowser

        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    return httpd, url
