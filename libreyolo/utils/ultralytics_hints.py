"""Error hints for Ultralytics names typed into LibreYOLO.

Every helper here only builds error text that names the LibreYOLO spelling to
type instead. Nothing aliases an Ultralytics name, and nothing loads or
converts an Ultralytics checkpoint.

Imported from ``libreyolo/__init__.py``'s ``__getattr__``, so this module must
stay free of torch and of other heavy imports.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from typing import NamedTuple

MODELS_URL = "https://www.libreyolo.com/docs/models"


class UltralyticsNameError(AttributeError):
    """Raised for an Ultralytics name that LibreYOLO spells differently.

    An ``AttributeError`` so ``hasattr()`` and ``getattr(obj, name, default)``
    keep working.
    """


# ---------------------------------------------------------------------------
# Top-level names: ``from libreyolo import YOLO``
# ---------------------------------------------------------------------------

_OPEN_VOCAB_CODE = (
    "from libreyolo import LibreOpenVocab  "
    '# pip install "libreyolo[openvocab]"\n'
    '    model = LibreOpenVocab("owlv2")\n'
    '    model.set_classes(["person", "bus"])'
)

_CLASS_HINTS = {
    "YOLO": (
        'from libreyolo import LibreYOLO\n    model = LibreYOLO("LibreYOLO9s.pt")'
    ),
    "RTDETR": (
        'from libreyolo import LibreYOLO\n    model = LibreYOLO("LibreRTDETRl.pt")'
    ),
    "NAS": (
        'from libreyolo import LibreYOLO\n    model = LibreYOLO("LibreYOLONASs.pt")'
    ),
    "YOLOWorld": _OPEN_VOCAB_CODE,
    "YOLOE": _OPEN_VOCAB_CODE,
    "SAM": (
        'from libreyolo import LibreSAM  # pip install "libreyolo[sam]"\n'
        '    model = LibreSAM("base")'
    ),
}


# Case-insensitive lookup, so ``Yolo``, ``yolo`` and ``Rtdetr`` get the hint
# too. No LibreYOLO export or submodule has any of these spellings.
_CLASS_HINTS_FOLDED = {key.lower(): code for key, code in _CLASS_HINTS.items()}


def class_name_hint(name: str) -> str | None:
    """Message for an Ultralytics class name looked up on ``libreyolo``."""
    code = _CLASS_HINTS_FOLDED.get(name.lower())
    if code is None:
        return None
    return f"libreyolo has no '{name}'. Use:\n    {code}"


def _executing_import_from(frame) -> bool:
    """True when ``frame`` is executing a ``from module import name``."""
    import dis

    code = frame.f_code.co_code
    index = frame.f_lasti
    return 0 <= index < len(code) and code[index] == dis.opmap["IMPORT_FROM"]


def raise_class_name_hint(name: str, module: str, module_file: str, message: str):
    """Raise ``message`` so it survives both ``from x import y`` and ``x.y``.

    ``from libreyolo import YOLO`` swallows any ``AttributeError`` the module
    ``__getattr__`` raises and prints a generic "cannot import name" instead,
    so the import statement gets an ``ImportError`` carrying the hint.
    Everything else (``libreyolo.YOLO``, ``hasattr``, ``getattr`` with a
    default) gets an ``AttributeError`` subclass. A class deriving from both
    is not an option: Python 3.10+ rejects the combination ("multiple bases
    have instance lay-out conflict"), and 3.9 swallows it like any other
    ``AttributeError``.
    """
    try:
        # Depth 2: this function, then the module __getattr__, then the
        # frame that ran the import or attribute access.
        caller = sys._getframe(2)
    except (AttributeError, ValueError):
        caller = None
    if caller is not None and _executing_import_from(caller):
        raise ImportError(message, name=module, path=module_file)
    raise UltralyticsNameError(message)


# ---------------------------------------------------------------------------
# Model methods LibreYOLO spells differently: ``model.add_callback(...)``
# ---------------------------------------------------------------------------

_METHOD_HINTS = {
    "add_callback": (
        "LibreYOLO models have no add_callback(). Pass training callbacks to "
        "train() instead:\n"
        '    model.train(data="your_data.yaml", callbacks=[fn])\n'
        "A plain function receives a TrainEpochEvent (epoch, train_loss, "
        "val_metrics, save_dir, ...) after every epoch, not a trainer object. "
        "Objects may define on_train_start, on_train_epoch_end, on_train_end "
        "and on_train_exception."
    ),
}


class UltralyticsMethodHint:
    """Class attribute that raises a hint for an Ultralytics-only method.

    Accessing it raises :class:`UltralyticsNameError`, an ``AttributeError``,
    so ``hasattr(model, name)`` stays False. A descriptor rather than a
    ``__getattr__`` on the model, which would also replace the message of any
    ``AttributeError`` raised inside the model's properties.
    """

    def __init__(self, name: str):
        self.message = _METHOD_HINTS[name]

    def __get__(self, instance, owner=None):
        if instance is None:
            return self  # class-level introspection (inspect, dir) stays quiet
        raise UltralyticsNameError(self.message)


# Export formats LibreYOLO does not produce, with what to use instead.
_EXPORT_FORMAT_HINTS = {
    "saved_model": "For TensorFlow runtimes, export format='tflite' or 'onnx'.",
    "pb": "For TensorFlow runtimes, export format='tflite' or 'onnx'.",
    "tfjs": "For the browser, export format='onnx' and run it with onnxruntime-web.",
    "edgetpu": "Edge TPU compilation is not built in; format='tflite' is the closest.",
}


def export_format_hint(format: str) -> str | None:
    """One sentence for an Ultralytics export format LibreYOLO lacks."""
    return _EXPORT_FORMAT_HINTS.get(str(format).lower())


# ---------------------------------------------------------------------------
# Weight names: ``LibreYOLO("yolov8n.pt")``
# ---------------------------------------------------------------------------


class WeightHint(NamedTuple):
    """The LibreYOLO counterpart of one Ultralytics weight name."""

    python: str  # code line for the Python API
    cli: str  # model= value for the CLI, "" when the CLI has none
    ultralytics: bool = True  # False for look-alikes such as yolov9-c
    untrained: bool = False  # True for model YAMLs: train from scratch


# Ultralytics release names: yolov3u, yolov5nu, yolov5n6u, yolov8n-seg,
# yolov9c, yolov10b, yolo11n-pose, yolo12x, yolo26n-obb, yolov8s-worldv2...
# plus common misspellings such as yolov11n and yolo8n. Versions after the
# latest known one (yolo27n, yolov30s-seg, ...) are treated as future
# releases with the same naming; 1, 2, 4, 7 and 13 to 25 are other projects'
# names and get no hint.
_ULTRALYTICS_VERSIONS = frozenset({3, 5, 6, 8, 9, 10, 11, 12, 26})
_LATEST_ULTRALYTICS_VERSION = max(_ULTRALYTICS_VERSIONS)
_YOLO_NAME = re.compile(
    r"^yolo-?v?(?P<version>[1-9][0-9]?)"
    r"(?P<size>[nsmblxtce])?6?u?"
    r"(?:-(?P<rest>[a-z0-9.-]+))?$"
)
_YOLOE_NAME = re.compile(r"^yoloe(?:-|$)")
_YOLO_NAS_NAME = re.compile(r"^yolo[_-]?nas[_-](?P<size>[sml])$")

_TASKS = {"seg": "segment", "pose": "pose", "obb": "obb", "cls": "classify"}
_SIZE_TOKENS = {"tiny": "t", "tinyu": "t", "spp": "l", "sppu": "l"}
_UPSTREAM_SUFFIXES = (".pt", ".pth", ".yaml", ".yml")

# LibreYOLO size per Ultralytics size letter, per task. Every target was
# checked to download from the LibreYOLO Hugging Face org.
_SIZE_LETTERS = "ntsmbclxe"
_SIZE_TABLE = {
    #            n    t    s    m    b    c    l    x    e
    "detect": ("t", "t", "s", "m", "c", "c", "c", "c", "c"),  # LibreYOLO9
    "segment": ("n", "n", "s", "m", "l", "l", "l", "x", "x"),  # LibreDFINE
    "pose": ("s", "s", "s", "m", "l", "l", "l", "x", "x"),  # LibreGTR
    "obb": ("s", "s", "s", "s", "x", "x", "x", "x", "x"),  # LibreGTR
    "classify": ("s", "s", "s", "m", "l", "l", "l", "l", "l"),  # LibreMobileNetV4
}


# Per task: LibreYOLO class, weight file and CLI model name.
_TASK_MODELS = {
    "detect": ("LibreYOLO9", "LibreYOLO9{s}.pt", "yolo9-{s}"),
    "segment": ("LibreDFINE", "LibreDFINE{s}-seg.pt", "dfine-{s}-seg"),
    "pose": ("LibreGTR", "LibreGTR{s}-pose.pt", "gtr-{s}-pose"),
    "obb": ("LibreGTR", "LibreGTR{s}-obb.pt", "gtr-{s}-obb"),
    "classify": (
        "LibreMobileNetV4",
        "LibreMobileNetV4{s}-cls.pt",
        "mobilenetv4-{s}-cls",
    ),
}


def _libre_size(task: str, size: str) -> str:
    return _SIZE_TABLE[task][_SIZE_LETTERS.index(size)]


def _split_suffix(name: str) -> tuple[str, str]:
    lower = name.lower()
    for suffix in _UPSTREAM_SUFFIXES:
        if lower.endswith(suffix):
            return lower[: -len(suffix)], suffix
    return lower, ""


def ultralytics_weight_hint(model_path: str) -> WeightHint | None:
    """Return the LibreYOLO counterpart of an Ultralytics weight name.

    ``None`` when ``model_path`` does not look like an Ultralytics name.
    """
    stem, suffix = _split_suffix(Path(str(model_path)).name)

    nas = _YOLO_NAS_NAME.match(stem)
    if nas:
        size = nas.group("size")
        return WeightHint(f'LibreYOLO("LibreYOLONAS{size}.pt")', f"yolonas-{size}")

    if _YOLOE_NAME.match(stem):
        return WeightHint(_OPEN_VOCAB_CODE, "")

    match = _YOLO_NAME.match(stem)
    if match is None:
        return None
    version = int(match.group("version"))
    if version not in _ULTRALYTICS_VERSIONS and version <= _LATEST_ULTRALYTICS_VERSION:
        return None
    size = match.group("size")
    tokens = (match.group("rest") or "").split("-")
    if "world" in tokens or "worldv2" in tokens:
        return WeightHint(_OPEN_VOCAB_CODE, "")
    ultralytics = True
    if size is None and tokens[0] in _SIZE_TOKENS:
        size = _SIZE_TOKENS[tokens[0]]  # yolov3-tinyu, yolov3-sppu
    elif size is None and len(tokens[0]) == 1:
        # yolov9-c (MultimediaTechLab) and the CLI-style yolo9-t are not
        # Ultralytics names, but still deserve the hint.
        size = tokens[0]
        ultralytics = False
    if size is None or size not in _SIZE_LETTERS:
        size = "s"
    task = next((_TASKS[t] for t in tokens if t in _TASKS), "detect")
    s = _libre_size(task, size)
    cls_name, filename, cli = _TASK_MODELS[task]
    filename = filename.format(s=s)
    cli = cli.format(s=s)

    if suffix in (".yaml", ".yml"):
        # A model YAML means "build it untrained", whatever the task.
        task_arg = "" if task == "detect" else f', task="{task}"'
        return WeightHint(
            f"from libreyolo import {cls_name}; "
            f'model = {cls_name}(None, size="{s}"{task_arg}); '
            # pretrained=False rebuilds the model for scratch training (some
            # families freeze parts of a randomly initialised network otherwise).
            f'model.train(data="your_data.yaml", pretrained=False)',
            cli,
            ultralytics,
            untrained=True,
        )
    return WeightHint(f'LibreYOLO("{filename}")', cli, ultralytics)


def weight_name_message(model_path: str, hint: WeightHint) -> str:
    """Python-API error text for an Ultralytics weight name."""
    name = Path(str(model_path)).name
    if hint.python.startswith("from "):
        code = hint.python
    else:
        code = f"model = {hint.python}"
    if hint.ultralytics:
        what = (
            f"'{name}' is an Ultralytics model name. LibreYOLO does not load "
            f"Ultralytics weights. The closest LibreYOLO model:"
        )
    else:
        what = f"'{name}' is not a LibreYOLO weight name. Did you mean:"
    return f"{what}\n    {code}\nAll models: {MODELS_URL}"


# ---------------------------------------------------------------------------
# Checkpoint files: ``LibreYOLO("best.pt")`` trained with Ultralytics
# ---------------------------------------------------------------------------

_ULTRALYTICS_GLOBAL = "ultralytics.nn"
# The pickle of a torch zip checkpoint holds only the object graph (tensors
# live in other members), so a few MB covers any real one. The cap bounds
# what an invalid or hostile file can make this error path read.
_MAX_PICKLE_BYTES = 4 * 1024 * 1024


def _is_checkpoint_pickle(member: str) -> bool:
    """True for ``<archive>/data.pkl``, the pickle of a torch zip checkpoint."""
    parts = member.split("/")
    return len(parts) == 2 and parts[0] != "" and parts[1] == "data.pkl"


def is_ultralytics_checkpoint(model_path: str, error: BaseException) -> bool:
    """True when a failed safe load points at an Ultralytics checkpoint.

    Reads the load error and, for zip checkpoints, at most the first
    ``_MAX_PICKLE_BYTES`` of ``<archive>/data.pkl`` as plain text. Nothing is
    unpickled.
    """
    texts = [str(error)]
    if error.__cause__ is not None:
        texts.append(str(error.__cause__))
    if any(_ULTRALYTICS_GLOBAL in text for text in texts):
        return True
    try:
        with zipfile.ZipFile(model_path) as archive:
            info = next(
                (i for i in archive.infolist() if _is_checkpoint_pickle(i.filename)),
                None,
            )
            if info is None or info.file_size > _MAX_PICKLE_BYTES:
                return False
            # read(n) also bounds a member whose header understates its size.
            with archive.open(info) as member:
                data = member.read(_MAX_PICKLE_BYTES)
            return _ULTRALYTICS_GLOBAL.encode() in data
    except (OSError, zipfile.BadZipFile, KeyError, RuntimeError, ValueError):
        return False


def checkpoint_message(model_path: str) -> str:
    """Error text for an Ultralytics-trained checkpoint file."""
    hint = ultralytics_weight_hint(model_path)
    start = hint.python if hint and not hint.python.startswith("from ") else None
    start = start or 'LibreYOLO("LibreYOLO9s.pt")'
    return (
        f"{model_path} is an Ultralytics checkpoint. LibreYOLO does not load "
        f"Ultralytics weights. Start from a LibreYOLO model and train it on "
        f"your dataset instead:\n"
        f"    model = {start}\n"
        f'    model.train(data="your_data.yaml", epochs=100)'
    )
