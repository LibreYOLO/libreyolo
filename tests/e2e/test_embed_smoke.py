"""Published-weight smoke coverage for the generic embedding contract."""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.slow,
    pytest.mark.clip,
]


def test_clip_embed_text_gallery_and_batch_smoke(tmp_path):
    from libreyolo import Gallery, LibreYOLO, SAMPLE_IMAGE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed", device=device)

    result = model.predict(SAMPLE_IMAGE)
    image_vector = result.embeddings.data
    text_vectors = model.embed_text(
        ["a person doing a flip in front of a museum", "a close-up photograph of a dog"]
    )

    assert result.boxes is None
    assert image_vector.shape == (1, 512)
    assert image_vector.dtype == torch.float32
    torch.testing.assert_close(
        image_vector.norm(dim=-1),
        torch.ones(1),
        rtol=0,
        atol=1e-5,
    )
    assert float(text_vectors[0] @ image_vector[0]) > float(
        text_vectors[1] @ image_vector[0]
    )
    assert model.embed([SAMPLE_IMAGE, SAMPLE_IMAGE]).shape == (2, 512)
    assert result.summary() == [{"embedding_dim": 512}]
    assert "embedding" in result.summary(embeddings=True)[0]

    gallery = Gallery(model)
    assert gallery.enroll("guggenheim", SAMPLE_IMAGE) == 1
    gallery_path = tmp_path / "images.gallery.npz"
    gallery.save(gallery_path)
    restored = Gallery.load(gallery_path, model=model)

    identified = model.predict(
        SAMPLE_IMAGE,
        gallery=restored,
        threshold=0.99,
    )
    assert identified.identities.name == ["guggenheim"]
