"""Structural unit tests for the V-JEPA 2 port.

These run offline with no checkpoints and no Transformers oracle. The exact
``max_abs_diff == 0.0`` parity gate against pinned ``transformers==5.1.0``
lives in ``weights/parity_vjepa2.py`` because it needs both the network and a
pinned out-of-tree oracle install.

Each test here locks in an invariant that was found to be load-bearing while
porting: getting any of them wrong still builds a model, but breaks the strict
load of a released checkpoint or silently changes the numerics.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.vjepa2.nn import (
    VJEPA2_CONFIGS,
    LibreVJEPA2Classifier,
    LibreVJEPA2Encoder,
    VJEPA2Config,
)

pytestmark = [pytest.mark.unit, pytest.mark.vjepa2]


# Transcribed from the pinned Hugging Face config.json of each released
# snapshot. Depth/heads/mlp_ratio are not inferable from the size label.
EXPECTED_ARCH = {
    "l256": (1024, 16, 24, 4.0, 256),
    "h256": (1280, 16, 32, 4.0, 256),
    "g256": (1408, 22, 40, 4.363636363636363, 256),
    "g384": (1408, 22, 40, 4.363636363636363, 384),
}


@pytest.mark.parametrize("size", sorted(EXPECTED_ARCH))
def test_architecture_table_matches_pinned_configs(size):
    hidden, heads, layers, ratio, crop = EXPECTED_ARCH[size]
    cfg = VJEPA2_CONFIGS[size]
    assert cfg["hidden_size"] == hidden
    assert cfg["num_attention_heads"] == heads
    assert cfg["num_hidden_layers"] == layers
    assert cfg["mlp_ratio"] == pytest.approx(ratio)
    assert cfg["crop_size"] == crop


def test_g_sizes_differ_only_by_crop():
    """g256 and g384 share a width; only the crop distinguishes them."""
    g256, g384 = VJEPA2_CONFIGS["g256"], VJEPA2_CONFIGS["g384"]
    assert g256["hidden_size"] == g384["hidden_size"] == 1408
    assert g256["crop_size"] != g384["crop_size"]


def test_unknown_size_rejected():
    with pytest.raises(ValueError, match="unknown V-JEPA 2 size"):
        VJEPA2Config.for_size("vitl")


def _tiny(**overrides) -> VJEPA2Config:
    params = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        mlp_ratio=4.363636363636363,
        crop_size=32,
        patch_size=16,
        tubelet_size=2,
        frames_per_clip=8,
    )
    params.update(overrides)
    return VJEPA2Config(**params)


def test_token_count_and_order():
    """(F/2) * (H/16) * (W/16) tokens, time-major flattening."""
    cfg = _tiny()
    model = LibreVJEPA2Encoder(cfg).eval()
    x = torch.randn(2, 8, 3, 32, 32)
    with torch.no_grad():
        tokens = model(x)
    # 8/2 * 32/16 * 32/16 = 4 * 2 * 2 = 16
    assert tokens.shape == (2, 16, 64)


@pytest.mark.parametrize("size,frames,expected", [("l256", 64, 8192), ("g384", 64, 18432)])
def test_released_token_counts(size, frames, expected):
    cfg = VJEPA2_CONFIGS[size]
    grid = cfg["crop_size"] // 16
    assert (frames // 2) * grid * grid == expected


def test_pooler_mlp_ratio_is_four_not_encoder_ratio():
    """The probe's MLP ratio is fixed at 4.0 even when the encoder's is not.

    Upstream builds the pooler MLP without passing ``mlp_ratio``, so it keeps
    the 4.0 default. Using the encoder's 4.3636 for a g-size probe would build
    fc1 wider than every released checkpoint provides.
    """
    cfg = _tiny()  # encoder ratio 4.3636...
    model = LibreVJEPA2Classifier(cfg, nc=5)
    assert model.encoder.layer[0].mlp.fc1.out_features == int(64 * 4.363636363636363)
    assert model.pooler.cross_attention_layer.mlp.fc1.out_features == int(64 * 4.0)


def test_pooler_cross_attention_has_no_output_projection():
    """Upstream's pooler cross-attention omits out_proj; adding one breaks load."""
    cfg = _tiny()
    model = LibreVJEPA2Classifier(cfg, nc=5)
    cross = model.pooler.cross_attention_layer.cross_attn
    assert not hasattr(cross, "out_proj")
    assert hasattr(model.pooler.self_attention_layers[0].self_attn, "out_proj")


def test_pooler_depth_is_three():
    model = LibreVJEPA2Classifier(_tiny(), nc=5)
    assert len(model.pooler.self_attention_layers) == 3


def test_classifier_logit_shape():
    model = LibreVJEPA2Classifier(_tiny(), nc=7).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 8, 3, 32, 32))
    assert logits.shape == (2, 7)


def test_temporal_order_changes_tokens():
    """A clip model that ignores frame order is a failed port."""
    torch.manual_seed(0)
    model = LibreVJEPA2Encoder(_tiny()).eval()
    x = torch.randn(1, 8, 3, 32, 32)
    with torch.no_grad():
        forward = model(x)
        backward = model(x.flip(dims=[1]))
    assert (forward - backward).abs().max().item() > 1e-4


def test_sdpa_and_eager_agree_closely():
    """Both attention paths implement the same math."""
    torch.manual_seed(0)
    x = torch.randn(1, 8, 3, 32, 32)
    sdpa = LibreVJEPA2Encoder(_tiny(attn_implementation="sdpa")).eval()
    eager = LibreVJEPA2Encoder(_tiny(attn_implementation="eager")).eval()
    eager.load_state_dict(sdpa.state_dict(), strict=True)
    with torch.no_grad():
        assert (sdpa(x) - eager(x)).abs().max().item() < 1e-5


def test_public_layout_is_batch_frames_channels():
    """Channels must be dim 2; a (B, C, F, H, W) input is a different picture."""
    model = LibreVJEPA2Encoder(_tiny()).eval()
    with torch.no_grad():
        ok = model(torch.randn(1, 8, 3, 32, 32))
    assert ok.shape[-1] == 64
    with pytest.raises(RuntimeError):
        model(torch.randn(1, 3, 8, 32, 32))


# ---------------------------------------------------------------------------
# Family / artifact-matrix behaviour
# ---------------------------------------------------------------------------

from libreyolo.models.vjepa2.model import LibreVJEPA2  # noqa: E402
from libreyolo.models.vjepa2.preprocess import clip_frame_indices  # noqa: E402


class TestArtifactMatrix:
    """Parsing a filename is not the same as the artifact existing."""

    @pytest.mark.parametrize(
        "name,size,task",
        [
            ("LibreVJEPA2l256-embed.pt", "l256", "embed"),
            ("LibreVJEPA2h256-embed.pt", "h256", "embed"),
            ("LibreVJEPA2g384-cls-ssv2.pt", "g384", "classify"),
            ("LibreVJEPA2l256-cls-diving48.pt", "l256", "classify"),
        ],
    )
    def test_published_names_resolve(self, name, size, task):
        assert LibreVJEPA2.detect_size_from_filename(name) == size
        assert LibreVJEPA2.detect_task_from_filename(name) == task

    def test_bare_family_name_is_not_canonical(self):
        # REQUIRE_TASK_SUFFIX: every published artifact carries a task suffix.
        assert LibreVJEPA2.detect_size_from_filename("LibreVJEPA2l256.pt") is None

    def test_embed_rejects_a_dataset_variant(self):
        with pytest.raises(ValueError, match="carry no dataset variant"):
            LibreVJEPA2.validate_artifact_name("l256", "embed", "ssv2")

    def test_published_classify_requires_a_variant(self):
        with pytest.raises(ValueError, match="require a dataset variant"):
            LibreVJEPA2.validate_artifact_name("l256", "classify", None)

    @pytest.mark.parametrize("size,variant", [("h256", "ssv2"), ("g256", "ssv2"), ("l256", "kinetics")])
    def test_unpublished_probe_combinations_rejected(self, size, variant):
        with pytest.raises(ValueError):
            LibreVJEPA2.validate_artifact_name(size, "classify", variant)

    @pytest.mark.parametrize("size,variant", [("l256", "ssv2"), ("l256", "diving48"), ("g384", "ssv2"), ("g384", "diving48")])
    def test_published_probe_combinations_accepted(self, size, variant):
        LibreVJEPA2.validate_artifact_name(size, "classify", variant)

    def test_unknown_size_rejected(self):
        with pytest.raises(ValueError, match="unknown V-JEPA 2 size"):
            LibreVJEPA2.validate_artifact_name("vitl", "embed", None)


class TestDiscriminators:
    def test_can_load_needs_a_5d_patch_embedding(self):
        five_d = {"embeddings.patch_embeddings.proj.weight": torch.zeros(8, 3, 2, 16, 16)}
        assert LibreVJEPA2.can_load(five_d)
        # A 4D Conv2d patch embed is an image ViT, not V-JEPA 2.
        four_d = {"embeddings.patch_embeddings.proj.weight": torch.zeros(8, 3, 16, 16)}
        assert not LibreVJEPA2.can_load(four_d)
        assert not LibreVJEPA2.can_load({"backbone.conv1.weight": torch.zeros(4, 3, 7, 7)})

    def test_can_load_accepts_both_checkpoint_roots(self):
        nested = {"encoder.embeddings.patch_embeddings.proj.weight": torch.zeros(8, 3, 2, 16, 16)}
        assert LibreVJEPA2.can_load(nested)

    @pytest.mark.parametrize("hidden,expected", [(1024, "l256"), (1280, "h256")])
    def test_detect_size_from_width(self, hidden, expected):
        assert LibreVJEPA2.detect_size({"layernorm.weight": torch.zeros(hidden)}) == expected

    def test_g_sizes_are_not_guessed_from_width(self):
        """g256 and g384 share a width, so width must not decide between them."""
        assert LibreVJEPA2.detect_size({"layernorm.weight": torch.zeros(1408)}) is None


class TestClipSampler:
    def test_exact_length_is_centered(self):
        # 4 frames at stride 2 spans 7 source frames.
        assert clip_frame_indices(7, 4, 2) == [0, 2, 4, 6]

    def test_long_video_is_centered(self):
        idx = clip_frame_indices(107, 4, 2)
        assert idx == [50, 52, 54, 56]
        assert len(idx) == 4

    def test_short_video_uses_real_frames_before_repeating(self):
        idx = clip_frame_indices(3, 4, 2)
        # Real frames first (0, 2), then hold the last rather than repeating early.
        assert idx == [0, 2, 2, 2]
        assert idx[0] == 0

    def test_released_default_spans_127_frames(self):
        idx = clip_frame_indices(1000, 64, 2)
        assert len(idx) == 64
        assert idx[-1] - idx[0] == 126

    def test_empty_video_is_an_error_not_a_black_clip(self):
        with pytest.raises(ValueError, match="no frames"):
            clip_frame_indices(0, 64, 2)

    @pytest.mark.parametrize("frames,stride", [(0, 2), (64, 0), (-1, 2)])
    def test_invalid_geometry_rejected(self, frames, stride):
        with pytest.raises(ValueError):
            clip_frame_indices(100, frames, stride)


class TestTrainingGate:
    def test_embed_training_rejects_before_dataset_construction(self):
        model = LibreVJEPA2(size="l256", task="embed")
        with pytest.raises(NotImplementedError, match="self-supervised"):
            model.train(data="anything.yaml")


class TestVideoMode:
    def test_family_declares_clip_mode(self):
        assert LibreVJEPA2.VIDEO_EMBED_MODE == "clip"

    def test_other_families_keep_frame_mode(self):
        """No-regression: existing families must keep per-frame video results."""
        from libreyolo.models.base.model import BaseModel

        assert getattr(BaseModel, "VIDEO_EMBED_MODE", "frames") == "frames"
        for family in ("dinov2", "clip", "siglip2"):
            cls = next(
                (c for c in BaseModel._registry if c.FAMILY == family), None
            )
            if cls is not None:
                assert getattr(cls, "VIDEO_EMBED_MODE", "frames") == "frames"


class TestInferenceContracts:
    """Regressions for the three review findings on the public paths."""

    def test_preprocess_returns_the_shared_four_tuple(self):
        """The shared runner unpacks four values; a bare tensor breaks predict()."""
        import numpy as np

        model = LibreVJEPA2(size="l256", task="embed")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out = model._preprocess(frame, "rgb")
        assert isinstance(out, tuple) and len(out) == 4
        tensor, _original, original_size, ratio = out
        # One frame, stackable by the shared whole-clip runner, which
        # concatenates frames into (1, F, C, H, W) itself.
        assert tensor.shape == (1, 3, 256, 256)
        assert original_size == (320, 240)         # (w, h) of the source
        assert ratio == 1.0

    def test_forward_expands_a_still_frame_into_a_static_clip(self):
        """A 4D still must become a clip, with every frame identical."""
        model = LibreVJEPA2(size="l256", task="embed")
        model._requested_clip_frames = 2
        frame = torch.zeros(1, 3, 256, 256, device=model.device)
        with torch.no_grad():
            row = model._forward(frame)
        assert row.shape == (1, 1024)

    def test_explicit_5d_clip_passes_through_preprocess(self):
        model = LibreVJEPA2(size="l256", task="embed")
        clip = torch.zeros(1, 4, 3, 256, 256)
        tensor, _, size, ratio = model._preprocess(clip)
        assert tensor.shape == (1, 4, 3, 256, 256)
        assert size == (256, 256) and ratio == 1.0

    def test_wrong_crop_in_explicit_clip_is_rejected(self):
        model = LibreVJEPA2(size="l256", task="embed")
        with pytest.raises(ValueError, match="requires 256x256"):
            model._preprocess(torch.zeros(1, 4, 3, 224, 224))

    def test_channels_last_clip_is_rejected(self):
        """(B, F, H, W, C) is a different picture; it must not be accepted."""
        model = LibreVJEPA2(size="l256", task="embed")
        with pytest.raises(ValueError, match=r"C=3 at dim 2"):
            model._preprocess(torch.zeros(1, 4, 256, 256, 3))

    def test_clip_mode_is_actually_consumed_by_the_runner(self):
        """Declaring VIDEO_EMBED_MODE is not enough; the runner must honour it."""
        from libreyolo.models.base.inference import InferenceRunner

        assert hasattr(InferenceRunner, "_predict_video_clip")

    def test_family_overrides_the_shared_uniform_sampler(self):
        """Temporal sampling is family-local; uniform sampling is wrong here.

        The shared runner spreads frames across the whole video. These
        checkpoints were trained on a fixed window, so the family supplies its
        own centered, strided sampler and the runner must prefer it.
        """
        import inspect

        from libreyolo.models.base.inference import InferenceRunner

        assert callable(getattr(LibreVJEPA2, "sample_clip_frames", None))
        source = inspect.getsource(InferenceRunner._predict_video_clip)
        assert 'getattr(self.model, "sample_clip_frames", None)' in source

    def test_centered_window_spans_the_trained_extent(self):
        """64 frames at stride 2 is a 127-frame span, centered."""
        from libreyolo.models.vjepa2.preprocess import clip_frame_indices

        idx = clip_frame_indices(1000, 64, 2)
        assert len(idx) == 64
        assert idx[-1] - idx[0] == 126
        # Centered, not anchored at the start.
        assert idx[0] > 0

    def test_embed_tokens_unpacks_the_preprocess_tuple(self):
        """Regression: embed_tokens fed the whole 4-tuple to the encoder."""
        model = LibreVJEPA2(size="l256", task="embed")
        model._requested_clip_frames = 2
        tokens = model.embed_tokens(torch.zeros(1, 2, 3, 256, 256))
        # (B, T', H', W', D) with T' = frames / tubelet, H' = W' = 256/16
        assert tokens.shape == (1, 1, 16, 16, 1024)

    @pytest.mark.parametrize("source", ["still_image", "frame_tensor"])
    def test_embed_tokens_accepts_a_still_frame(self, source):
        """Regression: _preprocess returns a 4D frame, the encoder is 5D-only.

        The earlier test only passed an explicit 5D clip, which takes the
        passthrough branch, so it never exercised the still-image path that
        actually broke.
        """
        import numpy as np

        model = LibreVJEPA2(size="l256", task="embed")
        model._requested_clip_frames = 2
        if source == "still_image":
            payload = np.zeros((240, 320, 3), dtype=np.uint8)
        else:
            payload = torch.zeros(1, 3, 256, 256, device=model.device)
        tokens = model.embed_tokens(payload)
        assert tokens.shape == (1, 1, 16, 16, 1024)

    def test_every_encoder_entry_point_promotes_to_a_clip(self):
        """_forward and embed_tokens must share one promotion path.

        They diverged once already: _forward expanded a 4D still into a clip
        while embed_tokens called the encoder directly and hit a rank error.
        """
        import inspect

        for method in (LibreVJEPA2._forward, LibreVJEPA2.embed_tokens):
            assert "_as_clip" in inspect.getsource(method), method.__name__

    def test_embed_tokens_rejects_a_classify_model(self):
        model = LibreVJEPA2(size="l256", task="classify", nb_classes=174)
        with pytest.raises(ValueError, match="requires task='embed'"):
            model.embed_tokens(torch.zeros(1, 2, 3, 256, 256))

    def test_clip_mode_video_options_are_handled_not_ignored(self):
        """Whole-clip inference must answer for save/show/vid_stride.

        The shared route (introduced by the Perception Encoder port) saves an
        annotated frame, and rejects show and vid_stride with reasons, rather
        than accepting a flag and quietly doing nothing.
        """
        import inspect

        from libreyolo.models.base.inference import InferenceRunner

        source = inspect.getsource(InferenceRunner.__call__)
        assert "vid_stride does not apply" in source
        assert "show=True displays annotated frames" in source
        assert "_predict_video_clip" in source

    def test_only_one_whole_clip_route_exists(self):
        """This family must extend the shared route, not add a competing one.

        Both this port and the Perception Encoder introduced a clip hook with
        the same names. Two definitions would silently shadow each other.
        """
        import inspect

        from libreyolo.models.base.inference import InferenceRunner

        source = inspect.getsource(InferenceRunner)
        assert source.count("def _predict_video_clip(") == 1

    def test_classify_shares_the_whole_clip_route(self):
        """A video probe collapses a video to one result, exactly like embed."""
        from libreyolo.models.base.inference import _WHOLE_CLIP_TASKS

        assert {"embed", "classify"} <= set(_WHOLE_CLIP_TASKS)

    def test_unknown_frame_count_decode_is_bounded(self):
        """A video with no header frame count must not be buffered whole.

        Memory has to stay proportional to the clip, not the video: the
        counting pass uses grab() (no pixel decode) and only the frames the
        clip actually needs are retained.
        """
        import inspect

        from libreyolo.models.base.inference import InferenceRunner

        source = inspect.getsource(LibreVJEPA2.sample_clip_frames)
        assert "capture.grab()" in source
        # The old unbounded accumulator must not come back.
        assert "frames_all" not in source
