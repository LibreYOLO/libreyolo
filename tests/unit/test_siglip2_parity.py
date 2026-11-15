"""LibreSigLIP2 <-> reference SiglipModel parity.

Skipped unless ``transformers`` + ``sentencepiece`` are installed (the
``libreyolo[siglip2-convert]`` extra) and the converted LibreSigLIP2 checkpoint is
present. These tests download/load real Apache-2.0 google/siglip2-* weights, so
they are marked ``external_data`` + ``network``.

Parity bar (see ``weights/parity_siglip2.py`` for the rationale):

* the text tower and the vision **encoder** are bit-identical (max_abs_diff == 0);
* the MAP-pooled image feature and the final logits differ by < 1e-5, an
  irreducible ``torch.nn.MultiheadAttention`` fused-kernel residual in the
  reference's pooling head.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("sentencepiece")

from libreyolo.models.siglip2.tokenizer import SigLIP2Tokenizer  # noqa: E402

pytestmark = [pytest.mark.external_data, pytest.mark.network, pytest.mark.siglip2]

_REPO_ROOT = Path(__file__).resolve().parents[2]

# size -> (libreyolo checkpoint, upstream HF repo)
_SPECS = {
    "b16": ("LibreSigLIP2b16-cls.pt", "google/siglip2-base-patch16-256"),
    "so400m": ("LibreSigLIP2so400m-cls.pt", "google/siglip2-so400m-patch14-384"),
}

_CORPUS = [
    "a photo of a cat",
    "an empty aisle",
    "un gato en la mesa",
    "eine Katze",
    "n01440764",
]

_EXACT_TOL = 0.0
_POOL_TOL = 1e-5


def test_tokenizer_parity():
    """Vendored SentencePiece must produce byte-exact token IDs vs the reference."""
    from transformers import AutoTokenizer

    ref = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-256")
    mine = SigLIP2Tokenizer()
    hf_ids = ref(_CORPUS, padding="max_length", max_length=64, return_tensors="pt")["input_ids"]
    assert torch.equal(mine(_CORPUS), hf_ids)


@pytest.mark.parametrize("size", list(_SPECS))
def test_forward_parity(size):
    ckpt_name, repo = _SPECS[size]
    ckpt = _REPO_ROOT / "weights" / ckpt_name
    if not ckpt.exists():
        pytest.skip(f"{ckpt_name} not converted (run weights/convert_siglip2_weights.py)")

    from transformers import AutoModel, AutoProcessor

    from libreyolo import LibreSigLIP2

    ref = AutoModel.from_pretrained(repo, attn_implementation="eager").eval()
    proc = AutoProcessor.from_pretrained(repo)
    model = LibreSigLIP2(str(ckpt), size=size, device="cpu")

    labels = ["a forklift", "an empty aisle", "a spill", "a cat"]
    ids = model.tokenizer([f"a photo of a {lbl}." for lbl in labels])
    import numpy as np
    from PIL import Image

    img = Image.fromarray((np.random.RandomState(0).rand(340, 500, 3) * 255).astype("uint8"))
    pix = proc(images=img, return_tensors="pt")["pixel_values"]

    with torch.no_grad():
        my_txt = model.model.encode_text(ids)
        ref_txt = ref.get_text_features(input_ids=ids).pooler_output
        my_img = model.model.encode_image(pix)
        ref_img = ref.get_image_features(pixel_values=pix).pooler_output

    assert (my_txt - ref_txt).abs().max().item() <= _EXACT_TOL
    assert (my_img - ref_img).abs().max().item() <= _POOL_TOL


def test_multilingual_spanish_smoke():
    """Spanish class names classify the parkour fixture correctly (b16)."""
    ckpt = _REPO_ROOT / "weights" / "LibreSigLIP2b16-cls.pt"
    if not ckpt.exists():
        pytest.skip("LibreSigLIP2b16-cls.pt not converted")

    from libreyolo import LibreSigLIP2

    model = LibreSigLIP2(str(ckpt), size="b16", device="cpu")
    labels = ["una persona haciendo parkour", "un perro", "un coche", "una calle vacia"]
    model.set_classes(labels, templates=["una foto de {}."])
    fixture = _REPO_ROOT / "tests" / "data" / "parkour.jpg"
    r = model.predict(str(fixture), verbose=False)[0]
    assert model.names[r.probs.top1] == "una persona haciendo parkour"


def test_imagenette_zero_shot_gate():
    """b16 zero-shot imagenette top-1 must be >= 0.99 via model.val().

    Needs a local imagenette ImageFolder; point IMAGENETTE_DIR at its root
    (containing train/ and val/ wnid folders).
    """
    import os

    data = os.environ.get("IMAGENETTE_DIR")
    if not data or not Path(data).is_dir():
        pytest.skip("Set IMAGENETTE_DIR to an imagenette ImageFolder root")
    ckpt = _REPO_ROOT / "weights" / "LibreSigLIP2b16-cls.pt"
    if not ckpt.exists():
        pytest.skip("LibreSigLIP2b16-cls.pt not converted")

    from libreyolo import LibreSigLIP2

    model = LibreSigLIP2(str(ckpt), size="b16", device="auto")
    res = model.val(data=data, imgsz=256, batch=32, workers=0, verbose=False)
    assert float(res["metrics/accuracy_top1"]) >= 0.99
