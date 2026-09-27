"""Hugging Face Hub integration: load checkpoints from any repo, push your own.

Two directions, both optional (``pip install libreyolo[hf]``):

- Loading: ``LibreYOLO("owner/repo")`` or ``LibreYOLO("hf://owner/repo")``
  downloads a LibreYOLO checkpoint from any Hub repository into the shared
  huggingface_hub cache and loads it through the normal metadata-validated
  path. ``hf://owner/repo@revision/file.pt`` pins a revision and file.
- Pushing: :func:`push_model_to_hub` / :func:`push_checkpoint_to_hub` upload a
  schema v1.0 checkpoint as ``model.pt`` plus an auto-generated model card
  derived from the checkpoint metadata.

``huggingface_hub`` is imported lazily so ``import libreyolo`` never pays for
it and torch-free deployments are unaffected.
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("libreyolo")

HF_URI_PREFIX = "hf://"

# The canonical checkpoint filename inside a pushed repo. Loading does not
# depend on it (any single .pt in a repo is picked up), but a fixed name keeps
# repeated pushes to the same repo as updates rather than accumulating files.
HUB_CHECKPOINT_FILENAME = "model.pt"

_REPO_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# A bare ``owner/name`` string whose last segment has a file extension keeps
# its historical meaning: a (possibly missing) local path that the legacy
# auto-download flow may create (``weights/model.pt``, ``ckpts/best.pth.tar``,
# ``cfg/yolov4.weights``). A dot followed by a digit is a version
# (``owner/model-v1.5``), not an extension, so such repo ids stay on the Hub.
_FILE_EXTENSION_RE = re.compile(r"\.[A-Za-z][A-Za-z0-9]*$")

# Names the factory routes by name and auto-downloads itself, with or without
# an extension (``models/librefacerec-l``).
_LOCAL_NAME_PREFIXES = ("librefacerec-",)


def _names_local_artifact(model_path: str) -> bool:
    name = model_path.rsplit("/", 1)[-1].lower()
    return bool(_FILE_EXTENSION_RE.search(name)) or name.startswith(
        _LOCAL_NAME_PREFIXES
    )


# File types inside a Hub repo that LibreYOLO(...) can load directly.
_WEIGHT_CANDIDATE_SUFFIXES = (".pt", ".safetensors")

# Canonical LibreYOLO task -> Hub pipeline_tag. Tasks without a good Hub
# equivalent are simply omitted from the card front matter.
_PIPELINE_TAGS = {
    "detect": "object-detection",
    "obb": "object-detection",
    "point": "object-detection",
    "segment": "image-segmentation",
    "semantic": "image-segmentation",
    "panoptic": "image-segmentation",
    "pose": "keypoint-detection",
    "classify": "image-classification",
    "depth": "depth-estimation",
    "edge": "image-to-image",
    "normal": "image-to-image",
    "restore": "image-to-image",
    "matte": "image-to-image",
    "ocr": "image-to-text",
    "embed": "image-feature-extraction",
}


@dataclass(frozen=True)
class HubRef:
    """A parsed Hugging Face Hub model reference."""

    repo_id: str
    filename: str | None = None
    revision: str | None = None


def _require_hub():
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError(
            "This model reference points to the Hugging Face Hub, which "
            "requires the optional huggingface_hub package. Install it with: "
            "pip install libreyolo[hf]"
        ) from exc
    return huggingface_hub


def _is_valid_repo_id(repo_id: str) -> bool:
    parts = repo_id.split("/")
    return len(parts) == 2 and all(_REPO_SEGMENT_RE.match(part) for part in parts)


def parse_hub_reference(model_path: str) -> HubRef | None:
    """Parse a model path into a :class:`HubRef`, or None for local paths.

    Recognized forms:

    - ``hf://owner/repo`` (optionally ``@revision``, optionally a trailing
      ``/path/to/file`` inside the repo)
    - bare ``owner/repo`` when it cannot be a local path: exactly one forward
      slash, valid Hub characters, no file extension, not a LibreYOLO
      auto-download name, and nothing with that name on disk.
    """
    if not isinstance(model_path, str) or not model_path:
        return None

    if model_path.startswith(HF_URI_PREFIX):
        remainder = model_path[len(HF_URI_PREFIX) :].strip("/")
        parts = remainder.split("/")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid Hugging Face reference {model_path!r}. Expected "
                "hf://owner/repo, hf://owner/repo@revision, or "
                "hf://owner/repo/path/to/file.pt."
            )
        owner, name = parts[0], parts[1]
        revision = None
        if "@" in name:
            name, _, revision = name.partition("@")
            if not revision:
                raise ValueError(
                    f"Invalid Hugging Face reference {model_path!r}: empty "
                    "revision after '@'."
                )
        repo_id = f"{owner}/{name}"
        if not _is_valid_repo_id(repo_id):
            raise ValueError(
                f"Invalid Hugging Face repository id {repo_id!r} in "
                f"{model_path!r}."
            )
        filename = "/".join(parts[2:]) or None
        return HubRef(repo_id=repo_id, filename=filename, revision=revision)

    # Bare owner/repo form: conservative, never claims anything that could be
    # (or become) a local path.
    if "\\" in model_path or model_path.count("/") != 1:
        return None
    if model_path.startswith((".", "~", "/")) or ":" in model_path:
        return None
    if _names_local_artifact(model_path):
        return None
    if not _is_valid_repo_id(model_path):
        return None
    if Path(model_path).exists() or Path(model_path.split("/")[0]).is_dir():
        return None
    return HubRef(repo_id=model_path)


def looks_like_repo_id(model_path: str) -> bool:
    """Pure-syntax check: could this string name a Hub repo (``owner/name``)?

    Unlike :func:`parse_hub_reference` this never consults the filesystem, so
    it can explain a failed local load that shadowed a plausible Hub id.
    """
    return (
        isinstance(model_path, str)
        and model_path.count("/") == 1
        and "\\" not in model_path
        and ":" not in model_path
        and not _names_local_artifact(model_path)
        and _is_valid_repo_id(model_path)
    )


def _auth_help(repo_id: str, *, write: bool = False) -> str:
    lines = [
        "Authenticate with the Hugging Face Hub in one of these ways:",
        "  1. Run `hf auth login` once (stores a token on this machine).",
        "  2. Set the HF_TOKEN environment variable to a token.",
        "  3. Pass token='hf_...' to this call.",
        "Create tokens at https://huggingface.co/settings/tokens.",
    ]
    if write:
        lines.append(
            f"Pushing to '{repo_id}' needs a token with WRITE scope "
            "(fine-grained tokens must include this repository)."
        )
    return "\n".join(lines)


def _select_repo_weight_file(repo_files: list[str], repo_id: str) -> str:
    candidates = [
        f for f in repo_files if f.lower().endswith(_WEIGHT_CANDIDATE_SUFFIXES)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No loadable checkpoint (*.pt or *.safetensors) found in Hugging "
            f"Face repo '{repo_id}'. Files present: {sorted(repo_files)}"
        )
    if len(candidates) == 1:
        return candidates[0]
    pt_files = [f for f in candidates if f.lower().endswith(".pt")]
    if len(pt_files) == 1:
        return pt_files[0]
    raise ValueError(
        f"Hugging Face repo '{repo_id}' contains multiple checkpoints: "
        f"{sorted(candidates)}. Select one explicitly with "
        f"LibreYOLO('hf://{repo_id}/<filename>')."
    )


def resolve_hub_checkpoint(ref: HubRef, *, token: str | None = None) -> str:
    """Download the checkpoint for ``ref`` and return its local cache path."""
    hub = _require_hub()
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    filename = ref.filename
    try:
        if filename is None:
            api = hub.HfApi(token=token)
            repo_files = api.list_repo_files(ref.repo_id, revision=ref.revision)
            filename = _select_repo_weight_file(list(repo_files), ref.repo_id)
        local_path = hub.hf_hub_download(
            repo_id=ref.repo_id,
            filename=filename,
            revision=ref.revision,
            token=token,
        )
    except GatedRepoError as exc:
        raise PermissionError(
            f"Hugging Face repo '{ref.repo_id}' is gated: request access on "
            f"its model page, then authenticate.\n{_auth_help(ref.repo_id)}"
        ) from exc
    except RepositoryNotFoundError as exc:
        raise FileNotFoundError(
            f"Hugging Face repo '{ref.repo_id}' was not found. Either the id "
            "is wrong, or the repo is private and you are not authenticated.\n"
            + _auth_help(ref.repo_id)
        ) from exc
    except RevisionNotFoundError as exc:
        raise FileNotFoundError(
            f"Revision '{ref.revision}' does not exist in Hugging Face repo "
            f"'{ref.repo_id}'."
        ) from exc
    except LocalEntryNotFoundError as exc:
        # Subclass of EntryNotFoundError, so it must be caught first. It means
        # "could not reach the Hub and it is not cached", which has nothing to
        # do with the file being absent from the repo.
        raise ConnectionError(
            f"Could not download '{filename or ref.repo_id}' from the Hugging "
            f"Face Hub and it is not in the local cache. Check your network "
            f"connection, or unset HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE if you "
            f"are in offline mode."
        ) from exc
    except EntryNotFoundError as exc:
        raise FileNotFoundError(
            f"File '{filename}' does not exist in Hugging Face repo "
            f"'{ref.repo_id}'."
        ) from exc
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise PermissionError(
                f"Access to Hugging Face repo '{ref.repo_id}' was denied "
                f"(HTTP {status}).\n" + _auth_help(ref.repo_id)
            ) from exc
        raise

    logger.info(
        "Resolved Hugging Face model '%s' to %s", ref.repo_id, local_path
    )
    return local_path


def maybe_resolve_hub_reference(
    model_path: str, *, token: str | None = None
) -> str | None:
    """Return a local checkpoint path if ``model_path`` is a Hub reference."""
    ref = parse_hub_reference(model_path)
    if ref is None:
        return None
    return resolve_hub_checkpoint(ref, token=token)


# =============================================================================
# Pushing
# =============================================================================


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _card_names(names: Any) -> list[str]:
    """Return class names in class-index order.

    The checkpoint schema accepts ``names`` as a list or as a dict whose keys
    may be ints or their string forms, so normalize all three here. Sorting
    raw dict keys would order string keys lexicographically ("10" before
    "2") and indexing a list with them would raise.
    """
    if isinstance(names, (list, tuple)):
        return [str(value) for value in names]
    if not isinstance(names, dict):
        return []
    try:
        ordered = sorted(names.items(), key=lambda item: int(item[0]))
    except (TypeError, ValueError):
        ordered = list(names.items())
    return [str(value) for _, value in ordered]


def _yaml_scalar(value: Any) -> str:
    """Render a checkpoint-derived value as a safe single-line YAML scalar.

    Checkpoint metadata is attacker-controlled for any file you did not write
    yourself, and it is rendered into the card's front matter. Without this, a
    ``model_family`` carrying a newline could inject its own keys (forging,
    say, a permissive ``license:`` on non-permissive weights).
    """
    text = str(value)
    if _REPO_SEGMENT_RE.match(text):
        # Plain token (the normal case): emit it bare so cards stay readable.
        return text
    for char in ("\r", "\n", "\t"):
        text = text.replace(char, " ")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').strip() + '"'


def build_model_card(
    metadata: dict[str, Any],
    repo_id: str,
    *,
    license_id: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Render a model card (README.md) from LibreYOLO checkpoint metadata."""
    family = str(metadata.get("model_family", "unknown"))
    size = str(metadata.get("size", ""))
    task = str(metadata.get("task", "detect"))
    nc = metadata.get("nc")
    imgsz = metadata.get("imgsz")
    names = metadata.get("names") or {}
    version = metadata.get("libreyolo_version", "unknown")

    front: list[str] = ["---", "library_name: libreyolo"]
    pipeline_tag = _PIPELINE_TAGS.get(task)
    if pipeline_tag:
        front.append(f"pipeline_tag: {pipeline_tag}")
    if license_id:
        front.append(f"license: {_yaml_scalar(license_id)}")
    front.append("tags:")
    for tag in ("libreyolo", family, task):
        front.append(f"- {_yaml_scalar(tag)}")
    front.append("---")

    body: list[str] = [
        "",
        f"# {family}{size} ({task})",
        "",
        "LibreYOLO checkpoint (metadata schema v1.0). Load it directly:",
        "",
        "```python",
        "from libreyolo import LibreYOLO",
        "",
        f'model = LibreYOLO("{repo_id}")',
        'results = model.predict("image.jpg")',
        "```",
        "",
        "## Model details",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Family | `{family}` |",
        f"| Size | `{size}` |",
        f"| Task | `{task}` |",
        f"| Classes (nc) | {nc} |",
        f"| Input size (imgsz) | {imgsz} |",
        f"| LibreYOLO version | `{version}` |",
    ]
    if metadata.get("quant"):
        body.append("| Quantized | yes |")

    ordered_names = _card_names(names)
    if ordered_names:
        shown = ordered_names[:50]
        suffix = ", ..." if len(ordered_names) > 50 else ""
        body += [
            "",
            "## Classes",
            "",
            ", ".join(shown) + suffix,
        ]

    if metrics:
        body += ["", "## Metrics", "", "| Metric | Value |", "| --- | --- |"]
        for key in sorted(metrics):
            body.append(f"| {key} | {_format_metric(metrics[key])} |")

    body += [
        "",
        "---",
        "",
        "Trained and published with [LibreYOLO]"
        "(https://github.com/LibreYOLO/libreyolo).",
        "",
    ]
    return "\n".join(front + body)


def _validate_push_repo_id(repo_id: str) -> None:
    if not isinstance(repo_id, str) or not _is_valid_repo_id(repo_id):
        raise ValueError(
            f"Invalid Hugging Face repository id {repo_id!r}. Expected "
            "'owner/name', e.g. 'someuser/my-yolo9-finetune'."
        )


def assert_can_push(repo_id: str, *, private: bool = True, token: str | None = None):
    """Verify now that a later push to ``repo_id`` can succeed.

    Checking that *a* token exists proves nothing: a token scoped to another
    namespace passes that test and then fails after the run it was supposed to
    protect. So this resolves the identity behind the token, refuses a
    namespace the user cannot write to, and finally creates the repo, which is
    the only real proof of write access. The repo is therefore created up
    front rather than at the end of training.
    """
    hub = _require_hub()
    from huggingface_hub.errors import HfHubHTTPError

    _validate_push_repo_id(repo_id)
    api = hub.HfApi(token=token)
    try:
        identity = api.whoami()
    except Exception as exc:
        raise PermissionError(
            f"Could not verify Hugging Face credentials for '{repo_id}'.\n"
            + _auth_help(repo_id, write=True)
        ) from exc

    owner = repo_id.split("/")[0]
    namespaces = {identity.get("name")}
    namespaces.update(
        org.get("name") for org in identity.get("orgs", []) or [] if org
    )
    namespaces.discard(None)
    if owner not in namespaces:
        raise PermissionError(
            f"You cannot push to '{repo_id}': you are signed in as "
            f"'{identity.get('name')}' and '{owner}' is not your username or "
            f"one of your organizations ({sorted(namespaces)}).\n"
            f"{_auth_help(repo_id, write=True)}"
        )

    try:
        api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise PermissionError(
                f"Your token cannot write to '{repo_id}' (HTTP {status}). A "
                f"fine-grained token must grant write access to this "
                f"repository.\n{_auth_help(repo_id, write=True)}"
            ) from exc
        raise
    return api


def push_checkpoint_to_hub(
    checkpoint_path: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    token: str | None = None,
    commit_message: str | None = None,
    license_id: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Upload a LibreYOLO checkpoint file plus a generated model card.

    The checkpoint must carry schema v1.0 metadata (anything written by
    ``model.save`` or the trainer qualifies). Returns the repo URL.
    """
    from .serialization import (
        load_untrusted_torch_file,
        validate_checkpoint_metadata,
    )

    _validate_push_repo_id(repo_id)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaded = load_untrusted_torch_file(
        checkpoint_path, map_location="cpu", context="hub upload inspection"
    )
    errors = validate_checkpoint_metadata(loaded, strict=False)
    if errors:
        raise ValueError(
            f"Refusing to push '{checkpoint_path}': it is not a LibreYOLO "
            "schema v1.0 checkpoint (" + "; ".join(errors) + "). Load it and "
            "re-save with model.save() first."
        )
    metadata = {k: v for k, v in loaded.items() if k != "model"}

    hub = _require_hub()
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    card = build_model_card(
        metadata, repo_id, license_id=license_id, metrics=metrics
    )
    if commit_message is None:
        commit_message = (
            f"Upload {metadata.get('model_family', 'model')}"
            f"{metadata.get('size', '')} {metadata.get('task', '')} "
            "checkpoint with LibreYOLO"
        )

    try:
        api = hub.HfApi(token=token)
        repo_url = api.create_repo(
            repo_id, private=private, exist_ok=True, repo_type="model"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            readme_path = Path(tmp_dir) / "README.md"
            readme_path.write_text(card, encoding="utf-8")
            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=HUB_CHECKPOINT_FILENAME,
                repo_id=repo_id,
                commit_message=commit_message,
            )
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Update model card",
            )
    except RepositoryNotFoundError as exc:
        raise PermissionError(
            f"Could not access Hugging Face repo '{repo_id}'.\n"
            + _auth_help(repo_id, write=True)
        ) from exc
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise PermissionError(
                f"Pushing to Hugging Face repo '{repo_id}' was denied "
                f"(HTTP {status}).\n" + _auth_help(repo_id, write=True)
            ) from exc
        raise

    url = str(getattr(repo_url, "url", None) or repo_url)
    logger.info("Pushed checkpoint to %s", url)
    return url


def push_model_to_hub(
    model,
    repo_id: str,
    *,
    private: bool = False,
    token: str | None = None,
    commit_message: str | None = None,
    license_id: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Save ``model`` as a v1.0 checkpoint and upload it to the Hub."""
    _validate_push_repo_id(repo_id)
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = model.save(str(Path(tmp_dir) / HUB_CHECKPOINT_FILENAME))
        return push_checkpoint_to_hub(
            checkpoint_path,
            repo_id,
            private=private,
            token=token,
            commit_message=commit_message,
            license_id=license_id,
            metrics=metrics,
        )
