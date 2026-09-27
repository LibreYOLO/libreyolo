"""Wrap unchanged Apache-2.0 Marigold trainables and fixed prompt tensors."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import torch

from ...utils.serialization import wrap_libreyolo_checkpoint
from .config import PROMPTS, UPSTREAM_REPO, UPSTREAM_REVISION, VARIANTS
from .nn import BASE_REPO, BASE_REVISION

VARIANT_KEY = "_marigold_variant"
PROMPT_KEYS = ("_marigold_prompt_embeds", "_marigold_prompt_mask")
TRAINING_ONLY_KEYS = {
    "iREPAStudentProjector.out__qwen_dit_hidden_state_-1.weight",
    "iREPAStudentProjector.out__qwen_dit_hidden_state_-8.weight",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def variant_from_state(state):
    code = state.get(VARIANT_KEY)
    if (
        not isinstance(code, torch.Tensor)
        or code.numel() != 1
        or code.dtype != torch.int64
    ):
        return None
    index = int(code.item())
    return tuple(VARIANTS)[index] if 0 <= index < len(VARIANTS) else None


def convert_checkpoint(source, destination, *, variant, prompt_dir=None, verify=True):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    info = VARIANTS[variant]
    if verify and sha256(source) != info.sha256:
        raise ValueError(f"Marigold V2 {variant} checkpoint checksum mismatch.")
    state = load_file(str(source), device="cpu")
    dropped = sorted(set(state) & TRAINING_ONLY_KEYS)
    state = {
        key: value for key, value in state.items() if key not in TRAINING_ONLY_KEYS
    }
    if not state or not all(key.startswith(("Diffuser.", "VAE.")) for key in state):
        raise ValueError("Expected an upstream Marigold V2 trainables checkpoint.")
    prefix = PROMPTS[info.task]
    for key, ending in zip(PROMPT_KEYS, ("prompt_embeds", "prompt_mask")):
        filename = f"{prefix}_{ending}.pt"
        path = (
            Path(prompt_dir) / filename
            if prompt_dir is not None
            else Path(
                hf_hub_download(
                    UPSTREAM_REPO,
                    f"qwen_text_embeddings/{filename}",
                    revision=UPSTREAM_REVISION,
                )
            )
        )
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Marigold prompt file is not a tensor: {filename}")
        state[key] = tensor
    state[VARIANT_KEY] = torch.tensor(tuple(VARIANTS).index(variant), dtype=torch.int64)
    checkpoint = wrap_libreyolo_checkpoint(
        state,
        model_family="marigold_v2",
        size="b",
        task=info.task,
        nc=1,
        names={0: info.task},
        imgsz=1024,
        variant=variant,
        base_model=BASE_REPO,
        base_revision=BASE_REVISION,
        upstream_repo=UPSTREAM_REPO,
        upstream_revision=UPSTREAM_REVISION,
        depth_encoding=info.encoding,
        omitted_training_tensors=dropped,
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".pt", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint
