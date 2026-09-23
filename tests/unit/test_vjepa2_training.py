"""Dataset validation and frozen-probe trainer tests for V-JEPA 2.

Everything here is offline and synthetic. The trainer consumes only
user-supplied videos: no dataset is downloaded, and Something-Something V2,
Diving48 and Kinetics are never touched.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.vjepa2.dataset import (
    ManifestError,
    VideoClipDataset,
    collate_clips,
    load_video_dataset,
    parse_manifest_line,
)
from libreyolo.models.vjepa2.model import LibreVJEPA2
from libreyolo.models.vjepa2.nn import LibreVJEPA2Classifier, VJEPA2Config
from libreyolo.models.vjepa2.trainer import VJEPA2Trainer

pytestmark = [pytest.mark.unit, pytest.mark.vjepa2]


class TestManifestParsing:
    def test_plain_row(self):
        assert parse_manifest_line("clips/a.mp4 3", 1) == ("clips/a.mp4", 3)

    def test_path_with_spaces_is_parsed_from_the_last_field(self):
        """Unquoted paths containing spaces must still resolve."""
        assert parse_manifest_line("my clips/a b.mp4 2", 1) == ("my clips/a b.mp4", 2)

    def test_tab_delimiter_wins_over_spaces(self):
        assert parse_manifest_line("my clips/a b.mp4\t7", 1) == ("my clips/a b.mp4", 7)

    def test_quoted_path(self):
        assert parse_manifest_line('"my clips/a b.mp4" 1', 1) == ("my clips/a b.mp4", 1)

    def test_non_integer_label_is_an_error(self):
        with pytest.raises(ManifestError, match="not an integer"):
            parse_manifest_line("clips/a.mp4 cat", 4)

    def test_missing_label_is_an_error(self):
        with pytest.raises(ManifestError):
            parse_manifest_line("clips_a.mp4", 2)

    def test_blank_row_is_an_error(self):
        with pytest.raises(ManifestError, match="blank"):
            parse_manifest_line("   ", 9)


def _write_dataset(tmp_path, *, rows=None, names=None, make_videos=True, val=True):
    videos = tmp_path / "videos"
    videos.mkdir(exist_ok=True)
    if make_videos:
        for name in ("a.mp4", "b.mp4"):
            (videos / name).write_bytes(b"not-a-real-video")
    rows = rows if rows is not None else ["videos/a.mp4 0", "videos/b.mp4 1"]
    (tmp_path / "train.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    if val:
        (tmp_path / "val.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    names = names if names is not None else {0: "left", 1: "right"}
    names_block = "\n".join(f"  {k}: {v}" for k, v in names.items())
    yaml_text = f"path: {tmp_path.as_posix()}\ntrain: train.txt\n"
    if val:
        yaml_text += "val: val.txt\n"
    yaml_text += f"names:\n{names_block}\n"
    path = tmp_path / "data.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return path


class TestDatasetValidation:
    def test_valid_dataset_loads(self, tmp_path):
        data = load_video_dataset(_write_dataset(tmp_path))
        assert data["nc"] == 2
        assert len(data["train"]) == 2
        assert data["names"][0] == "left"

    def test_missing_yaml(self, tmp_path):
        with pytest.raises(ManifestError, match="not found"):
            load_video_dataset(tmp_path / "nope.yaml")

    def test_missing_video_is_caught_before_training(self, tmp_path):
        path = _write_dataset(tmp_path, rows=["videos/ghost.mp4 0", "videos/b.mp4 1"])
        with pytest.raises(ManifestError, match="video not found"):
            load_video_dataset(path)

    def test_label_out_of_range(self, tmp_path):
        path = _write_dataset(tmp_path, rows=["videos/a.mp4 0", "videos/b.mp4 9"])
        with pytest.raises(ManifestError, match="out of range"):
            load_video_dataset(path)

    def test_names_must_be_contiguous_from_zero(self, tmp_path):
        path = _write_dataset(tmp_path, names={0: "left", 2: "right"})
        with pytest.raises(ManifestError, match="contiguous"):
            load_video_dataset(path)

    def test_names_mismatch_is_rejected(self, tmp_path):
        """A class named but never used is a dataset bug, not a warning."""
        path = _write_dataset(
            tmp_path,
            rows=["videos/a.mp4 0", "videos/b.mp4 0"],
            names={0: "left", 1: "right"},
        )
        with pytest.raises(ManifestError, match="in no\n?\\s*manifest row|in no manifest row"):
            load_video_dataset(path)

    def test_missing_train_manifest(self, tmp_path):
        videos = tmp_path / "videos"
        videos.mkdir()
        (tmp_path / "data.yaml").write_text(
            f"path: {tmp_path.as_posix()}\nnames:\n  0: a\n", encoding="utf-8"
        )
        with pytest.raises(ManifestError, match="missing required 'train'"):
            load_video_dataset(tmp_path / "data.yaml")

    def test_empty_names_rejected(self, tmp_path):
        (tmp_path / "data.yaml").write_text(
            f"path: {tmp_path.as_posix()}\ntrain: train.txt\nnames: {{}}\n",
            encoding="utf-8",
        )
        with pytest.raises(ManifestError, match="non-empty mapping"):
            load_video_dataset(tmp_path / "data.yaml")

    def test_validation_does_not_touch_the_network(self, tmp_path, monkeypatch):
        """Dataset validation must never reach for a remote corpus."""
        import urllib.request

        def _boom(*args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("dataset validation attempted a network call")

        monkeypatch.setattr(urllib.request, "urlopen", _boom)
        load_video_dataset(_write_dataset(tmp_path))


def _tiny_classifier(nc=2, frames=4, size=64):
    cfg = VJEPA2Config(
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        mlp_ratio=2.0,
        crop_size=size,
        patch_size=16,
        tubelet_size=2,
        frames_per_clip=frames,
    )
    return LibreVJEPA2Classifier(cfg, nc=nc, probe_depth=1)


def _bare_trainer(model):
    trainer = VJEPA2Trainer.__new__(VJEPA2Trainer)
    trainer.model = model
    return trainer


class TestFreezing:
    def test_encoder_is_frozen_and_in_eval_mode(self):
        model = _tiny_classifier()
        model.train()
        trainable = _bare_trainer(model).freeze_encoder()
        assert all(not p.requires_grad for p in model.encoder.parameters())
        assert not model.encoder.training
        assert trainable, "the probe must still have trainable parameters"

    def test_only_pooler_and_classifier_train(self):
        model = _tiny_classifier()
        trainable = _bare_trainer(model).freeze_encoder()
        assert all(n.startswith(("pooler.", "classifier.")) for n in trainable)
        assert any(n.startswith("pooler.") for n in trainable)
        assert any(n.startswith("classifier.") for n in trainable)

    def test_optimizer_membership_excludes_the_encoder(self):
        model = _tiny_classifier()
        trainer = _bare_trainer(model)

        class _Cfg:
            lr0 = 1e-3
            weight_decay = 0.05

        trainer.config = _Cfg()
        optimizer = trainer._setup_optimizer()
        optimized = {id(p) for g in optimizer.param_groups for p in g["params"]}
        encoder = {id(p) for p in model.encoder.parameters()}
        assert optimized.isdisjoint(encoder)
        assert optimized


class TestForwardContract:
    def test_forward_requires_a_5d_clip(self):
        trainer = _bare_trainer(_tiny_classifier())
        with pytest.raises(ValueError, match=r"\(B, F, C, H, W\)"):
            trainer.on_forward(torch.zeros(2, 3, 64, 64), torch.zeros(2, dtype=torch.long))

    def test_forward_returns_a_cross_entropy_loss(self):
        model = _tiny_classifier()
        trainer = _bare_trainer(model)
        out = trainer.on_forward(
            torch.randn(2, 4, 3, 64, 64), torch.tensor([0, 1])
        )
        assert "total_loss" in out and out["total_loss"].requires_grad
        assert trainer.get_loss_components(out)["ce"] > 0

    def test_collate_keeps_time_out_of_batch(self):
        batch = [(torch.zeros(4, 3, 64, 64), 0), (torch.zeros(4, 3, 64, 64), 1)]
        clips, labels = collate_clips(batch)
        assert clips.shape == (2, 4, 3, 64, 64)
        assert labels.tolist() == [0, 1]

    def test_best_metric_is_top1_accuracy(self):
        assert VJEPA2Trainer.best_metric_key == "metrics/accuracy_top1"


class TestTrainingGates:
    def test_embed_training_rejects_immediately(self):
        model = LibreVJEPA2(size="l256", task="embed")
        with pytest.raises(NotImplementedError, match="self-supervised"):
            model.train(data="whatever.yaml")

    def test_predictor_pretraining_rejects(self):
        model = LibreVJEPA2(size="l256", task="classify", nb_classes=2)
        with pytest.raises(NotImplementedError, match="predictor"):
            model.train(data="whatever.yaml", predictor=True)

    def test_full_finetune_requires_explicit_support(self):
        model = LibreVJEPA2(size="l256", task="classify", nb_classes=2)
        with pytest.raises(NotImplementedError, match="freeze=0"):
            model.train(data="whatever.yaml", freeze=0)


@pytest.mark.slow
class TestSyntheticMotionConvergence:
    """The label is the DIRECTION of travel.

    A model that ignores frame order cannot beat chance here, so this is what
    proves the clip path preserves time end to end through training -- a
    static-colour dataset would pass even with time discarded.
    """

    def _make_dataset(self, tmp_path, frames=8, size=64):
        cv2 = pytest.importorskip("cv2")
        import numpy as np

        videos = tmp_path / "videos"
        videos.mkdir()

        def write(path, rightwards):
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size)
            )
            for i in range(frames * 3):
                frame = np.zeros((size, size, 3), np.uint8)
                t = i if rightwards else (frames * 3 - 1 - i)
                x = 4 + (t * 2) % (size - 12)
                cv2.rectangle(frame, (x, 26), (x + 8, 34), (255, 255, 255), -1)
                writer.write(frame)
            writer.release()

        rows = []
        for i in range(8):
            for label, rightwards in ((0, True), (1, False)):
                name = f"clip_{i}_{label}.mp4"
                write(videos / name, rightwards)
                rows.append(f"videos/{name} {label}")
        (tmp_path / "train.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (tmp_path / "data.yaml").write_text(
            f"path: {tmp_path.as_posix()}\ntrain: train.txt\n"
            "names:\n  0: rightwards\n  1: leftwards\n",
            encoding="utf-8",
        )
        return tmp_path / "data.yaml"

    def test_probe_learns_direction(self, tmp_path):
        from torch.utils.data import DataLoader

        # A convergence assertion on a randomly initialized probe is otherwise
        # seed-dependent; pin it so a failure means a real regression rather
        # than an unlucky draw.
        torch.manual_seed(0)

        frames, size = 8, 64
        yaml_path = self._make_dataset(tmp_path, frames=frames, size=size)
        data = load_video_dataset(yaml_path)

        model = _tiny_classifier(nc=2, frames=frames, size=size)
        trainer = _bare_trainer(model)

        class _Cfg:
            lr0 = 3e-3
            weight_decay = 0.05

        trainer.config = _Cfg()
        optimizer = trainer._setup_optimizer()

        loader = DataLoader(
            VideoClipDataset(data["train"], frames, 1, size, train=True),
            batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_clips,
        )

        first = last = None
        accuracy = 0.0
        for _ in range(12):
            total, correct, seen = 0.0, 0, 0
            for clips, labels in loader:
                assert clips.ndim == 5
                out = trainer.on_forward(clips, labels)
                optimizer.zero_grad()
                out["total_loss"].backward()
                optimizer.step()
                total += float(out["total_loss"])
                with torch.no_grad():
                    correct += int((model(clips).argmax(-1) == labels).sum())
                seen += labels.numel()
            mean = total / max(1, len(loader))
            first = mean if first is None else first
            last = mean
            accuracy = correct / seen

        assert last < first, f"loss did not decrease: {first} -> {last}"
        assert accuracy > 0.5, f"did not beat chance: {accuracy}"


# ---------------------------------------------------------------------------
# Epoch validation on the video val manifest (#886 side quest)
# ---------------------------------------------------------------------------


def _write_clip_dataset(tmp_path, frames=4, size=64, n_val=4):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    videos = tmp_path / "videos"
    videos.mkdir()
    rows = []
    for i in range(n_val):
        path = videos / f"clip_{i}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size))
        for t in range(frames * 3):
            frame = np.zeros((size, size, 3), np.uint8)
            cv2.rectangle(frame, (4 + 2 * t, 26), (12 + 2 * t, 34), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()
        rows.append(f"videos/{path.name} {i % 2}")
    for split in ("train", "val"):
        (tmp_path / f"{split}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        f"path: {tmp_path.as_posix()}\ntrain: train.txt\nval: val.txt\n"
        "names:\n  0: a\n  1: b\n",
        encoding="utf-8",
    )
    return yaml_path


def _tiny_wrapper(frames=4, size=64):
    class _TinyVJEPA2(LibreVJEPA2):
        crop_size = size  # the per-size property, pinned to the tiny model

    wrapper = _TinyVJEPA2.__new__(_TinyVJEPA2)
    wrapper.task = "classify"
    wrapper.size = "l256"
    wrapper.variant = None
    wrapper._requested_clip_frames = frames
    wrapper.frame_stride = 1
    wrapper.model = _tiny_classifier(nc=2, frames=frames, size=size)
    wrapper.device = torch.device("cpu")
    wrapper.names = {0: "a", 1: "b"}
    wrapper.nb_classes = 2
    return wrapper


class TestEpochValidation:
    def test_the_model_validates_on_its_video_val_manifest(self, tmp_path):
        from libreyolo.models.vjepa2.validator import VJEPA2ClipValidator
        from libreyolo.validation import ValidationConfig

        assert LibreVJEPA2.validator_class is VJEPA2ClipValidator
        yaml_path = _write_clip_dataset(tmp_path)
        validator = VJEPA2ClipValidator(
            model=_tiny_wrapper(),
            config=ValidationConfig(
                data=str(yaml_path), batch_size=2, imgsz=64, device="cpu",
                num_workers=0, verbose=False, save_dir=str(tmp_path / "val"),
            ),
        )
        metrics = validator.run()
        assert metrics["speed/images_seen"] == 4
        assert 0.0 <= metrics["metrics/accuracy_top1"] <= 1.0

    def test_epoch_validation_returns_a_top1_metric(self, tmp_path):
        """It used to hand the video YAML to the ImageFolder validator and return None."""
        from types import SimpleNamespace

        yaml_path = _write_clip_dataset(tmp_path)
        wrapper = _tiny_wrapper()
        trainer = VJEPA2Trainer.__new__(VJEPA2Trainer)
        trainer.wrapper_model = wrapper
        trainer.model = wrapper.model
        trainer.ema_model = None
        trainer.device = torch.device("cpu")
        trainer.world_size = 1
        trainer.config = SimpleNamespace(
            data=str(yaml_path), batch=2, imgsz=64, amp=False, amp_dtype="float16",
            workers=0, crop_pct=None, val_loss=False,
        )
        result = trainer._run_classify_validation(0)
        assert result is not None
        assert result["best_metric_key"] == "metrics/accuracy_top1"
        assert result["metrics"]["speed/images_seen"] == 4
