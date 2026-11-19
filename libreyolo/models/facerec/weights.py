"""Named weights and auto-download for the face-embedding (``embed``) task.

The family ships two ONNX artifacts on the LibreYOLO Hugging Face org:

- ``librefacerec-l.onnx``   — iResNet100 recognition head, 512-d embeddings
  (mirrored single-file from an Apache-2.0 upstream release).
- ``librefacerec-det.onnx`` — lightweight face detector with 5 landmarks
  (MIT-licensed artifact from the OpenCV zoo), used as the default detector.

Any other ArcFace-convention ONNX (aligned 112x112 in, ``(N, D)`` out) can be
used by passing its file path directly (bring-your-own-weights).
"""

from __future__ import annotations

from pathlib import Path

_HF_ORG = "LibreYOLO"
_HF_BASE = f"https://huggingface.co/{_HF_ORG}"

#: Canonical downloadable artifacts: filename -> HF resolve URL.
FACEREC_WEIGHT_URLS = {
    "librefacerec-l.onnx": f"{_HF_BASE}/librefacerec-l/resolve/main/librefacerec-l.onnx",
    "librefacerec-det.onnx": f"{_HF_BASE}/librefacerec-det/resolve/main/librefacerec-det.onnx",
}

#: Embedder sizes the factory accepts as ``librefacerec-<size>``.
FACEREC_SIZES = ("l",)


def is_facerec_weight_name(model_path: str) -> bool:
    """True for ``librefacerec-*`` names/paths (with or without ``.onnx``)."""
    return Path(model_path).name.lower().startswith("librefacerec-")


def resolve_facerec_weight(model_path: str) -> str:
    """Resolve a ``librefacerec-*`` name to a local path, downloading if needed.

    Bare names resolve into the standard ``weights/`` directory. Existing
    file paths are returned unchanged.
    """
    path = Path(model_path)
    name = path.name.lower()
    if not name.endswith(".onnx"):
        name += ".onnx"

    if path.exists():
        return str(path)

    if name not in FACEREC_WEIGHT_URLS:
        known = ", ".join(sorted(FACEREC_WEIGHT_URLS))
        raise FileNotFoundError(
            f"Unknown face-embedding weight '{path.name}'. Known downloadable "
            f"names: {known}. For third-party recognition models, pass the "
            f"path to a local ArcFace-convention ONNX file instead."
        )

    # Hub id form (``LibreYOLO/librefacerec-l``): fetch into the shared
    # huggingface_hub cache, never into a ``LibreYOLO/`` folder in the working
    # directory, which would shadow every later ``LibreYOLO/<repo>`` Hub id.
    if path.parent.as_posix().lower() == _HF_ORG.lower():
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=f"{_HF_ORG}/{name[:-5]}", filename=name)

    # Bare name (or weights/-prefixed name from the factory resolver).
    dest = path if path.parent != Path(".") else Path("weights") / name
    dest = dest.with_name(name)
    if not dest.exists():
        from ...utils.download import download_url_to_path

        download_url_to_path(FACEREC_WEIGHT_URLS[name], dest)
    return str(dest)


def default_face_detector():
    """Build the default face detector, downloading its weights if needed."""
    from .model import OpenCVFaceDetector

    det_path = resolve_facerec_weight("librefacerec-det.onnx")
    return OpenCVFaceDetector(det_path)
