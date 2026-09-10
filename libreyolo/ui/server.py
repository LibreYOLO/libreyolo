"""Standard-library HTTP server backing ``libreyolo ui``.

Zero third-party dependencies: the page is served from :mod:`libreyolo.ui.page`
and inference reuses the same ``LibreYOLO`` predict path the CLI uses, writing
annotated images to ``runs/detect/predict`` exactly like ``predict --save``.

Endpoints
---------
``GET  /``                page (HTML)
``GET  /api/models``      JSON list of resolvable model names + default
``GET  /api/sample``      bundled sample image
``POST /api/run/new``     start a fresh ``runs/detect/predict`` output dir
``POST /api/infer``       body = raw image bytes, header ``X-Filename``,
                          query ``model`` + ``conf`` (+ optional ``classes``,
                          a comma-separated text vocabulary for open-vocab /
                          zero-shot models); returns the rendered (annotated)
                          image as a base64 data URL + a task-aware summary
``POST /api/open-folder`` open the current results dir in the OS file manager
"""

from __future__ import annotations

import base64
from io import BytesIO
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


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    word = singular if n == 1 else (plural or singular + "s")
    return f"{n} {word}"


def _summarize_result(result) -> tuple[str, str]:
    """Return (task, label): a task name and a short human summary for the UI."""
    # Gaze first: gaze results also carry the face boxes, so the generic
    # boxes branch below would mislabel them as plain detection.
    gaze = getattr(result, "gaze", None)
    if gaze is not None:
        return "gaze", _plural(len(gaze), "face")

    # Region embeddings may carry boxes; whole-image embeddings do not.
    embeddings = getattr(result, "embeddings", None)
    if embeddings is not None:
        if getattr(result, "boxes", None) is None:
            unit = "image"
        else:
            unit = (
                "face" if "face" in getattr(result, "names", {}).values() else "region"
            )
        unknown_unit = f"unknown {unit}"
        identities = getattr(result, "identities", None)
        if identities is not None and len(identities) == len(embeddings):
            known = list(dict.fromkeys(n for n in identities.name if n is not None))
            unknown = sum(1 for n in identities.name if n is None)
            if known:
                label = ", ".join(known)
                if unknown:
                    label += f" (+{_plural(unknown, unknown_unit)})"
                return "embed", label
            return "embed", _plural(len(identities), unknown_unit)
        return "embed", _plural(len(embeddings), unit)

    # OBB results also populate ``boxes`` with the axis-aligned proxy, so the
    # rotated slot has to be checked first or every OBB model reports "detect".
    obb = getattr(result, "obb", None)
    if obb is not None:
        return "obb", _plural(len(obb), "object")

    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        n = len(boxes)
        if getattr(result, "masks", None) is not None:
            return "segment", _plural(n, "instance")
        if getattr(result, "keypoints", None) is not None:
            return "pose", _plural(n, "pose")
        return "detect", _plural(n, "object")

    points = getattr(result, "points", None)
    if points is not None:
        return "point", _plural(len(points), "point")

    probs = getattr(result, "probs", None)
    if probs is not None:
        top1 = int(probs.top1)
        names = getattr(result, "names", None)
        name = str(top1)
        if names:
            name = str(names[top1])
        conf = float(probs.top1conf)
        return "classify", f"{name} {conf * 100:.0f}%"

    panoptic = getattr(result, "panoptic", None)
    if panoptic is not None:
        return "panoptic", _plural(len(panoptic.segments_info), "segment")

    semantic_mask = getattr(result, "semantic_mask", None)
    if semantic_mask is not None:
        n = len(semantic_mask.classes)
        return "semantic", _plural(n, "region")

    if getattr(result, "depth_map", None) is not None:
        return "depth", "depth map"

    if getattr(result, "normal_map", None) is not None:
        return "normal", "normal map"

    if getattr(result, "albedo", None) is not None:
        return "albedo", "albedo map"

    if getattr(result, "restored", None) is not None:
        scale = getattr(result, "restore_scale", 1)
        label = f"upscaled x{scale}" if scale and scale != 1 else "restored"
        return "restore", label

    if getattr(result, "matte", None) is not None:
        return "matte", "alpha matte"

    ocr = getattr(result, "ocr", None)
    if ocr is not None:
        return "ocr", _plural(len(ocr), "text region")

    return "detect", "0 objects"


def _result_count(result) -> int:
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        return int(len(boxes))

    obb = getattr(result, "obb", None)
    if obb is not None:
        return int(len(obb))

    points = getattr(result, "points", None)
    if points is not None:
        return int(len(points))

    panoptic = getattr(result, "panoptic", None)
    if panoptic is not None:
        return int(len(panoptic.segments_info))

    ocr = getattr(result, "ocr", None)
    if ocr is not None:
        return int(len(ocr))

    return 0


def _resolve_download_url(name: str) -> str | None:
    """Resolve the Hugging Face download URL the app would use for a model name,
    or None if no model class can build one. Mirrors the real download path."""
    from libreyolo.cli.config import resolve_model_name
    from libreyolo.models import try_ensure_rfdetr
    from libreyolo.models.base.model import BaseModel

    filename = Path(resolve_model_name(name)).name

    # The face-embedding family is ONNX-only and lives outside the model
    # registry, so it publishes its download URLs directly.
    from libreyolo.models.facerec.weights import FACEREC_WEIGHT_URLS

    if filename in FACEREC_WEIGHT_URLS:
        return FACEREC_WEIGHT_URLS[filename]

    classes = list(BaseModel._registry)
    rf = try_ensure_rfdetr()
    if rf is not None and rf not in classes:
        classes.append(rf)
    for cls in classes:
        try:
            url = cls.get_download_url(filename)
        except Exception:
            url = None
        if url:
            return url
    return None


def _self_managed_download(filename: str) -> bool:
    """Whether a weight filename belongs to a family that fetches its own
    weights (HF snapshot repos, Google Drive) and so has no per-file URL."""
    from libreyolo.models.base.model import BaseModel

    for cls in BaseModel._registry:
        prefix = getattr(cls, "FILENAME_PREFIX", "")
        if not prefix or not filename.startswith(prefix):
            continue
        hook = getattr(cls, "supports_autodownload", None)
        if callable(hook):
            return bool(hook(filename))
        if getattr(cls, "HF_REPOS", None) or hasattr(cls, "_try_autodownload"):
            return True
    return False


def _factory_openvocab_names() -> list[str]:
    """Open-vocab detectors live behind the ``LibreOpenVocab`` factory (HF
    snapshot weights, no CLI weight filename), so the UI lists them itself.
    One canonical alias per model/size; the factory resolves them."""
    import importlib.util

    # The openvocab package itself imports fine without transformers (the
    # backend is deferred to model construction), so probe for the actual
    # dependency rather than the package.
    if importlib.util.find_spec("transformers") is None:
        return []
    try:
        from libreyolo.models.openvocab import LibreOpenVocab  # noqa: F401
    except Exception:
        return []
    return [
        "grounding-dino-t",
        "grounding-dino-b",
        "owlv2-b16",
        "owlv2-l14",
        "omdet-turbo",
        "ov-deim-s",
        "ov-deim-m",
        "ov-deim-l",
    ]


def _factory_picosam_names() -> list[str]:
    """Promptable edge models with a dedicated UI inference path."""

    import importlib.util

    if importlib.util.find_spec("huggingface_hub") is None:
        return []
    return ["picosam3"]


def _openvocab_model_names(names: list[str]) -> list[str]:
    """Model names that accept a per-run text vocabulary (``set_classes``):
    the factory-tier open-vocab detectors plus zero-shot classifier families
    from the CLI registry (clip, siglip2)."""
    from libreyolo.models.base.model import BaseModel

    factory = set(_factory_openvocab_names())
    prefixes = tuple(
        f"{cls.FAMILY}-"
        for cls in BaseModel._registry
        if callable(getattr(cls, "set_classes", None)) and getattr(cls, "FAMILY", "")
    )
    return [n for n in names if n in factory or (prefixes and n.startswith(prefixes))]


# HTTP statuses that mean "this weight is definitively not there" (grey it out).
# Anything else, including timeouts/connection errors, is treated as available so
# a network hiccup never disables a model that actually works.
_UNAVAILABLE_STATUSES = frozenset({400, 401, 403, 404, 410})


def _load_banner_font(image_font, size: int):
    for font_name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return image_font.truetype(font_name, size=size)
        except OSError:
            continue
    return image_font.load_default()


def _text_box(draw, text: str, font) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return int(right - left), int(bottom - top)
    width, height = draw.textsize(text, font=font)
    return int(width), int(height)


def _overlay_classification_label(path: Path, label: str, fallback: bytes) -> bytes:
    try:
        from PIL import Image, ImageDraw, ImageFont

        with Image.open(path) as image:
            image_format = image.format or (
                "JPEG" if path.suffix.lower() in (".jpg", ".jpeg") else "PNG"
            )
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            max_size = max(14, min(48, image.width // 18))
            font = _load_banner_font(ImageFont, max_size)
            pad = max(6, max_size // 3)

            for size in range(max_size, 9, -2):
                candidate = _load_banner_font(ImageFont, size)
                text_w, text_h = _text_box(draw, label, candidate)
                if text_w + pad * 2 <= image.width or size == 10:
                    font = candidate
                    break

            text_w, text_h = _text_box(draw, label, font)
            strip_w = min(image.width, text_w + pad * 2)
            strip_h = min(image.height, text_h + pad * 2)
            draw.rectangle((0, 0, strip_w, strip_h), fill=(15, 23, 42))
            draw.text((pad, pad), label, fill=(255, 255, 255), font=font)

            output = BytesIO()
            image.save(output, format=image_format)
            rendered = output.getvalue()

        path.write_bytes(rendered)
        return rendered
    except Exception:
        logger.debug("Failed to draw classification label", exc_info=True)
        return fallback


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
        self._upload_lock = threading.Lock()
        self._used_upload_names: set[str] = set()
        self.run_dir: Path | None = None
        self._default_names: dict[str, dict] = {}  # set_classes models: original vocab
        self._input_dir = Path(tempfile.mkdtemp(prefix="libreyolo-ui-"))
        self._availability: dict[str, bool] | None = None
        self._avail_lock = threading.Lock()

    def model_availability(self) -> dict[str, bool]:
        """Map each CLI model name to whether its weights are downloadable.

        A model is unavailable when no download URL can be built or the URL
        returns a definitive not-there status. Checked once (in parallel) and
        cached for the life of the server; network errors count as available.
        """
        with self._avail_lock:
            if self._availability is not None:
                return self._availability

        from concurrent.futures import ThreadPoolExecutor

        import requests

        from libreyolo.cli.config import get_all_cli_names

        from libreyolo.cli.config import resolve_model_name

        def check(name: str) -> tuple[str, bool]:
            url = _resolve_download_url(name)
            if not url:
                filename = Path(resolve_model_name(name)).name
                return name, _self_managed_download(filename)
            try:
                resp = requests.get(
                    url,
                    stream=True,
                    headers={"Range": "bytes=0-0"},
                    timeout=8,
                    allow_redirects=True,
                )
                code = resp.status_code
                resp.close()
                return name, code not in _UNAVAILABLE_STATUSES
            except Exception:
                return name, True

        names = get_all_cli_names()
        with ThreadPoolExecutor(max_workers=16) as pool:
            result = dict(pool.map(check, names))
        # Factory-tier open-vocab models fetch HF snapshots on first use; there
        # is no single weight URL to probe, so list them as available.
        for name in _factory_openvocab_names():
            result[name] = True
        for name in _factory_picosam_names():
            result[name] = True

        with self._avail_lock:
            self._availability = result
        return result

    def _get_model(self, name: str):
        model = self._models.get(name)
        if model is None:
            if name in _factory_openvocab_names():
                from libreyolo.models.openvocab import LibreOpenVocab

                logger.info("Loading open-vocab model %s", name)
                model = LibreOpenVocab(name, device=self.device)
            elif name in _factory_picosam_names():
                from libreyolo import LibreSAM

                logger.info("Loading promptable model %s", name)
                model = LibreSAM(name, device=self.device)
            else:
                from libreyolo import LibreYOLO
                from libreyolo.cli.config import resolve_model_name

                weight = resolve_model_name(name)
                logger.info("Loading model %s (%s)", name, weight)
                model = LibreYOLO(weight, device=self.device)
            self._models[name] = model
        return model

    def new_run(self) -> Path:
        from libreyolo.utils.general import increment_path

        self.run_dir = increment_path(Path("runs/detect") / "predict", mkdir=True)
        with self._upload_lock:
            self._used_upload_names.clear()
        return self.run_dir

    def _write_upload(self, filename: str, data: bytes) -> tuple[Path, str]:
        safe = _sanitize_filename(filename)
        stem = Path(safe).stem or "image"
        suffix = Path(safe).suffix or ".jpg"
        with self._upload_lock:
            upload_name = safe
            index = 2
            while upload_name in self._used_upload_names:
                upload_name = f"{stem}_{index}{suffix}"
                index += 1
            self._used_upload_names.add(upload_name)
            in_path = self._input_dir / upload_name
            in_path.write_bytes(data)
        return in_path, safe

    def _apply_vocabulary(self, model_name: str, model, classes: list[str] | None):
        """Apply (or reset) the text vocabulary on a ``set_classes`` model.

        Models are cached for the server's life and ``set_classes`` is sticky,
        so the original vocabulary is remembered per model and restored when a
        request arrives without one.
        """
        set_classes = getattr(model, "set_classes", None)
        if not callable(set_classes):
            return
        default = self._default_names.get(model_name)
        if default is None:
            default = dict(model.names)
            self._default_names[model_name] = default
        if classes:
            logger.info("Vocabulary: %s", ", ".join(classes))
            set_classes(classes)
        elif model.names != default:
            set_classes(list(default.values()))

    def infer(
        self,
        model_name: str,
        conf: float,
        filename: str,
        data: bytes,
        emit=None,
        classes: list[str] | None = None,
        bbox: list[float] | None = None,
    ) -> dict:
        """Run inference. If ``emit`` is given, libreyolo log lines (model
        download, save path, ...) are forwarded to it live during the run."""
        with self._lock:
            if self.run_dir is None:
                self.new_run()
            in_path, safe = self._write_upload(filename, data)
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
                self._apply_vocabulary(model_name, model, classes)
                if model_name in _factory_picosam_names():
                    from PIL import Image

                    from libreyolo.models.base.inference import InferenceRunner
                    from libreyolo.utils.general import resolve_save_path

                    with Image.open(in_path) as uploaded:
                        width, height = uploaded.size
                        original = uploaded.convert("RGB")
                    prompt_box = bbox or [0.0, 0.0, float(width), float(height)]
                    result = model(str(in_path), conf=conf, bboxes=prompt_box)
                    save_path = resolve_save_path(self.run_dir, safe, ext="jpg")
                    InferenceRunner(model)._save_annotated_image(
                        result, original, save_path
                    )
                else:
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

        task, label = _summarize_result(result)
        count = _result_count(result)
        suffix = Path(saved).suffix.lower().lstrip(".")
        mime = "jpeg" if suffix in ("jpg", "jpeg", "") else suffix
        rendered = Path(saved).read_bytes()
        if task == "classify":
            rendered = _overlay_classification_label(Path(saved), label, rendered)
        encoded = base64.b64encode(rendered).decode("ascii")
        return {
            "name": safe,
            "count": int(count),
            "task": task,
            "label": label,
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

            names = sorted(
                get_all_cli_names()
                + _factory_openvocab_names()
                + _factory_picosam_names()
            )
            avail = self.state.model_availability()
            unavailable = [n for n in names if not avail.get(n, True)]
            usable = [n for n in names if avail.get(n, True)]
            default = (
                "yolo9-t" if "yolo9-t" in usable else (usable[0] if usable else "")
            )
            self._send(
                200,
                {
                    "models": names,
                    "unavailable": unavailable,
                    "openvocab": _openvocab_model_names(names),
                    "box_prompt": _factory_picosam_names(),
                    "default": default,
                },
            )
        elif path == "/api/sample":
            sample = (
                Path(__file__).resolve().parents[1] / "assets" / "guggenheim-bilbao.jpg"
            )
            try:
                data = sample.read_bytes()
            except FileNotFoundError:
                self._send(404, {"error": "sample not found"})
            except OSError as exc:
                logger.exception("Failed to load sample image")
                self._send(500, {"error": str(exc)})
            else:
                self._send(200, data, "image/jpeg")
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
        raw_classes = qs.get("classes", [""])[0]
        classes = [c.strip() for c in raw_classes.split(",") if c.strip()] or None
        raw_bbox = qs.get("bbox", [""])[0]
        bbox = None
        bbox_error = None
        if raw_bbox:
            try:
                bbox = [float(value.strip()) for value in raw_bbox.split(",")]
            except ValueError:
                bbox_error = "bbox must contain four comma-separated numbers"
            else:
                if len(bbox) != 4:
                    bbox_error = "bbox must contain x1,y1,x2,y2"

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

        if bbox_error:
            write_obj({"type": "error", "error": bbox_error})
            return

        if not data:
            write_obj({"type": "error", "error": "empty upload"})
            return

        def emit(msg: str) -> None:
            write_obj({"type": "log", "line": msg})

        try:
            result = self.state.infer(
                model,
                conf,
                filename,
                data,
                emit=emit,
                classes=classes,
                bbox=bbox,
            )
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
