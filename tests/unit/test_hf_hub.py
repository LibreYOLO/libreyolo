"""Unit tests for the Hugging Face Hub integration (issue #786)."""

from __future__ import annotations

import sys
from types import MappingProxyType

import pytest
import torch

from libreyolo.models import LibreYOLO
from libreyolo.models.yolo9.nn import LibreYOLO9Model
from libreyolo.utils import hf_hub
from libreyolo.utils.hf_hub import (
    HubRef,
    build_model_card,
    parse_hub_reference,
    push_checkpoint_to_hub,
    push_model_to_hub,
    _select_repo_weight_file,
)
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Reference parsing
# ---------------------------------------------------------------------------


def test_parse_bare_repo_id():
    ref = parse_hub_reference("someuser/my-finetune")
    assert ref == HubRef(repo_id="someuser/my-finetune")


def test_parse_hf_uri_forms():
    assert parse_hub_reference("hf://u/r") == HubRef("u/r")
    assert parse_hub_reference("hf://u/r@abc123") == HubRef("u/r", None, "abc123")
    assert parse_hub_reference("hf://u/r/sub/file.pt") == HubRef(
        "u/r", "sub/file.pt", None
    )
    assert parse_hub_reference("hf://u/r@v2/best.pt") == HubRef(
        "u/r", "best.pt", "v2"
    )


@pytest.mark.parametrize(
    "bad", ["hf://only-owner", "hf://u/r@", "hf://u/bad*name"]
)
def test_parse_invalid_hf_uri_raises(bad):
    with pytest.raises(ValueError):
        parse_hub_reference(bad)


@pytest.mark.parametrize(
    "path",
    [
        "libreyolo9s.pt",  # no slash: plain filename
        "weights/libreyolo9s.pt",  # local artifact extension keeps old flow
        "some/dir/file.pt",  # more than one slash
        "./relative/name",
        "~/home/name",
        "/abs/name",
        "C:/drive/name",
        "dir\\name",
        "owner/data.yaml",
        "http://example.com/x",
    ],
)
def test_parse_rejects_local_and_url_paths(path):
    assert parse_hub_reference(path) is None


def test_parse_rejects_existing_local_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "owner").mkdir()
    (tmp_path / "owner" / "repo").write_text("x")
    assert parse_hub_reference("owner/repo") is None


def test_parse_rejects_when_owner_is_local_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    assert parse_hub_reference("runs/exp1") is None


@pytest.mark.parametrize(
    "path",
    [
        "ckpts/model_best.pth.tar",
        "ckpts/last.ckpt",
        "cfg/yolov4.weights",
        "paddle/model.pdparams",
        "keras/model.h5",
        "models/librefacerec-l",
        "models/librefacerec-l.onnx",
    ],
)
def test_parse_keeps_missing_local_style_paths_local(tmp_path, monkeypatch, path):
    monkeypatch.chdir(tmp_path)
    assert parse_hub_reference(path) is None
    assert not hf_hub.looks_like_repo_id(path)


def test_parse_keeps_versioned_repo_ids_on_the_hub(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert parse_hub_reference("someuser/yolo9-v1.5") == HubRef("someuser/yolo9-v1.5")
    assert parse_hub_reference("hf://models/librefacerec-l") == HubRef(
        "models/librefacerec-l"
    )


def _no_hub(*args, **kwargs):
    raise AssertionError("local-style path must not reach the Hub")


def test_factory_routes_librefacerec_path_to_auto_download(tmp_path, monkeypatch):
    import libreyolo.models.facerec as facerec

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hf_hub, "resolve_hub_checkpoint", _no_hub)
    seen = []
    monkeypatch.setattr(
        facerec, "LibreFaceEmbedder", lambda path, device: seen.append(path)
    )

    LibreYOLO("models/librefacerec-l", device="cpu")

    assert seen == ["models/librefacerec-l"]


def test_factory_missing_local_checkpoint_does_not_reach_the_hub(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hf_hub, "resolve_hub_checkpoint", _no_hub)

    with pytest.raises(ValueError, match="Model weights file not found"):
        LibreYOLO("ckpts/model_best.pth.tar", device="cpu")


# ---------------------------------------------------------------------------
# Repo file selection
# ---------------------------------------------------------------------------


def test_select_single_pt():
    assert _select_repo_weight_file(["README.md", "model.pt"], "u/r") == "model.pt"


def test_select_prefers_single_pt_over_safetensors():
    files = ["model.safetensors", "model.pt", "README.md"]
    assert _select_repo_weight_file(files, "u/r") == "model.pt"


def test_select_multiple_pt_raises_with_hf_syntax():
    with pytest.raises(ValueError, match="hf://u/r/"):
        _select_repo_weight_file(["a.pt", "b.pt"], "u/r")


def test_select_no_weights_raises():
    with pytest.raises(FileNotFoundError):
        _select_repo_weight_file(["README.md"], "u/r")


# ---------------------------------------------------------------------------
# Missing optional dependency
# ---------------------------------------------------------------------------


def test_missing_hub_package_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match=r"libreyolo\[hf\]"):
        hf_hub.resolve_hub_checkpoint(HubRef("u/r"))


# ---------------------------------------------------------------------------
# Resolution against a faked huggingface_hub
# ---------------------------------------------------------------------------


def _make_yolo9_checkpoint(path, names=None):
    model = LibreYOLO9Model(config="t", nb_classes=80)
    torch.save(
        wrap_libreyolo_checkpoint(
            model.state_dict(),
            model_family="yolo9",
            size="t",
            task="detect",
            nc=80,
            names=names,
            imgsz=640,
        ),
        path,
    )
    return path


def test_resolve_hub_checkpoint_downloads_selected_file(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    ckpt = _make_yolo9_checkpoint(tmp_path / "model.pt")

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_files(self, repo_id, revision=None):
            assert repo_id == "someuser/finetune"
            return ["README.md", "model.pt"]

    calls = {}

    def fake_download(repo_id, filename, revision=None, token=None):
        calls["args"] = (repo_id, filename, revision)
        return str(ckpt)

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    monkeypatch.setattr(hub, "hf_hub_download", fake_download)

    local = hf_hub.resolve_hub_checkpoint(HubRef("someuser/finetune"))
    assert local == str(ckpt)
    assert calls["args"] == ("someuser/finetune", "model.pt", None)


def test_resolve_repo_not_found_teaches_auth(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import RepositoryNotFoundError

    class _FakeResponse:
        status_code = 404
        headers = {}
        url = "https://huggingface.co/api/models/ghost/repo"
        text = ""
        request = None

    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, revision=None):
            raise RepositoryNotFoundError("nope", response=_FakeResponse())

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    with pytest.raises(FileNotFoundError) as excinfo:
        hf_hub.resolve_hub_checkpoint(HubRef("ghost/repo"))
    message = str(excinfo.value)
    assert "hf auth login" in message
    assert "HF_TOKEN" in message


def test_resolve_reports_offline_instead_of_missing_file(monkeypatch):
    """No network is not the same failure as the file being absent."""
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import LocalEntryNotFoundError

    def offline(**kwargs):
        raise LocalEntryNotFoundError("no internet and no cache")

    monkeypatch.setattr(hub, "hf_hub_download", offline)
    with pytest.raises(ConnectionError, match="network connection"):
        hf_hub.resolve_hub_checkpoint(HubRef("u/r", "model.pt"))


def test_resolve_missing_entry_names_the_resolved_file(monkeypatch):
    """The message must name the auto-selected file, not 'None'."""
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import EntryNotFoundError

    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, revision=None):
            return ["README.md", "model.pt"]

    def missing(**kwargs):
        raise EntryNotFoundError("gone")

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    monkeypatch.setattr(hub, "hf_hub_download", missing)
    with pytest.raises(FileNotFoundError, match="'model.pt'"):
        hf_hub.resolve_hub_checkpoint(HubRef("u/r"))


def test_factory_refuses_metadata_less_hub_checkpoint(tmp_path, monkeypatch):
    """Guessing a family from raw keys misroutes arbitrary Hub files."""
    import torch as _torch

    foreign = tmp_path / "model.pt"
    _torch.save({"transformer.h.0.attn.weight": _torch.zeros(4, 4)}, foreign)
    monkeypatch.setattr(
        hf_hub, "maybe_resolve_hub_reference", lambda p, **kw: str(foreign)
    )

    with pytest.raises(ValueError, match="does not contain a LibreYOLO checkpoint"):
        LibreYOLO("someuser/not-a-libreyolo-model", device="cpu")


def test_factory_loads_bare_repo_reference(tmp_path, monkeypatch):
    ckpt = _make_yolo9_checkpoint(
        tmp_path / "model.pt", names={i: f"c{i}" for i in range(80)}
    )
    monkeypatch.setattr(
        hf_hub, "maybe_resolve_hub_reference", lambda p, **kw: str(ckpt)
    )

    loaded = LibreYOLO("someuser/finetune", device="cpu")

    assert loaded.nb_classes == 80
    assert loaded.names[0] == "c0"


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------


def _metadata(**overrides):
    metadata = {
        "schema_version": "1.0",
        "libreyolo_version": "1.5.0",
        "model_family": "yolo9",
        "size": "t",
        "task": "detect",
        "nc": 2,
        "names": {0: "red", 1: "white"},
        "imgsz": 640,
    }
    metadata.update(overrides)
    return metadata


def test_model_card_front_matter_and_body():
    card = build_model_card(
        _metadata(), "someuser/finetune", license_id="mit", metrics={"mAP50": 0.5}
    )
    front = card.split("---")[1]
    assert "library_name: libreyolo" in front
    assert "pipeline_tag: object-detection" in front
    assert "license: mit" in front
    assert "- yolo9" in front
    assert 'LibreYOLO("someuser/finetune")' in card
    assert "| mAP50 | 0.5000 |" in card
    assert "red, white" in card


def test_model_card_unknown_task_omits_pipeline_tag():
    card = build_model_card(_metadata(task="gaze"), "u/r")
    assert "pipeline_tag" not in card
    assert "license:" not in card


def test_model_card_accepts_names_as_list():
    """A list is valid per the checkpoint schema and must not crash the card."""
    card = build_model_card(_metadata(names=["red", "white"]), "u/r")
    assert "red, white" in card


def test_model_card_orders_string_keyed_names_numerically():
    names = {str(i): f"c{i}" for i in range(12)}
    card = build_model_card(_metadata(nc=12, names=names), "u/r")
    assert "c0, c1, c2, c3" in card
    assert "c1, c10, c11, c2" not in card


def test_model_card_front_matter_resists_metadata_injection():
    """Checkpoint metadata is untrusted: it must not forge front-matter keys."""
    card = build_model_card(
        _metadata(model_family="x\nlicense: proprietary"), "u/r"
    )
    front = card.split("---")[1]
    assert "\nlicense: proprietary" not in front
    assert '- "x license: proprietary"' in front


def test_model_card_truncates_many_names():
    names = {i: f"class_{i}" for i in range(120)}
    card = build_model_card(_metadata(nc=120, names=names), "u/r")
    assert "class_49" in card
    assert "class_50, " not in card
    assert "..." in card


# ---------------------------------------------------------------------------
# Pushing
# ---------------------------------------------------------------------------


class _RecordingApi:
    instances = []

    def __init__(self, token=None):
        self.token = token
        self.created = []
        self.uploads = []
        _RecordingApi.instances.append(self)

    def create_repo(self, repo_id, private=False, exist_ok=False, repo_type=None):
        self.created.append((repo_id, private))
        return f"https://huggingface.co/{repo_id}"

    def upload_file(
        self, path_or_fileobj, path_in_repo, repo_id, commit_message=None
    ):
        content = None
        if path_in_repo == "README.md":
            with open(path_or_fileobj, encoding="utf-8") as fh:
                content = fh.read()
        self.uploads.append((path_in_repo, repo_id, content))


@pytest.fixture
def recording_api(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    _RecordingApi.instances = []
    monkeypatch.setattr(hub, "HfApi", _RecordingApi)
    return _RecordingApi


def test_push_checkpoint_uploads_weights_and_card(tmp_path, recording_api):
    ckpt = _make_yolo9_checkpoint(tmp_path / "best.pt")

    url = push_checkpoint_to_hub(ckpt, "someuser/finetune", private=True)

    assert url == "https://huggingface.co/someuser/finetune"
    api = recording_api.instances[-1]
    assert api.created == [("someuser/finetune", True)]
    names = [upload[0] for upload in api.uploads]
    assert names == ["model.pt", "README.md"]
    readme = api.uploads[1][2]
    assert "library_name: libreyolo" in readme


def test_push_rejects_metadata_less_checkpoint(tmp_path, recording_api):
    bad = tmp_path / "raw.pt"
    torch.save({"conv.weight": torch.zeros(1)}, bad)
    with pytest.raises(ValueError, match="schema v1.0"):
        push_checkpoint_to_hub(bad, "someuser/finetune")
    assert recording_api.instances == []


@pytest.mark.parametrize("bad_repo", ["norepo", "a/b/c", "bad*chars/x", ""])
def test_push_rejects_invalid_repo_id(tmp_path, bad_repo):
    with pytest.raises(ValueError, match="owner/name"):
        push_checkpoint_to_hub(tmp_path / "x.pt", bad_repo)


def test_push_model_saves_then_uploads(tmp_path, recording_api):
    class DummyModel:
        def save(self, path):
            return str(_make_yolo9_checkpoint(path))

    url = push_model_to_hub(DummyModel(), "someuser/finetune")
    assert url == "https://huggingface.co/someuser/finetune"
    assert recording_api.instances[-1].uploads[0][0] == "model.pt"


# ---------------------------------------------------------------------------
# Training logger
# ---------------------------------------------------------------------------


def _train_end_event(save_dir):
    from libreyolo.training.callbacks import TrainEndEvent

    return TrainEndEvent(
        total_epochs=2,
        completed_epochs=2,
        model_family="yolo9",
        model_size="t",
        task="detect",
        save_dir=str(save_dir),
        final_loss=1.0,
        best_metric=0.5,
        best_epoch=1,
        total_seconds=12.0,
        results=MappingProxyType(
            {"best_mAP50": 0.5, "final_loss": 1.0, "note": "text", "flag": True}
        ),
    )


def _patch_ambient_token(monkeypatch, allowed=True):
    """Stub the write preflight the logger runs at construction time."""
    pytest.importorskip("huggingface_hub")
    seen: list[tuple[str, bool]] = []

    def fake_preflight(repo_id, *, private=True, token=None):
        seen.append((repo_id, private))
        if not allowed:
            raise PermissionError("cannot write; run `hf auth login`")
        return None

    monkeypatch.setattr(hf_hub, "assert_can_push", fake_preflight)
    return seen


def test_hf_logger_fails_fast_without_write_access(monkeypatch):
    """A credential problem must surface before training, not after it."""
    from libreyolo.training.loggers import HuggingFaceHubLogger

    _patch_ambient_token(monkeypatch, allowed=False)
    with pytest.raises(PermissionError, match="hf auth login"):
        HuggingFaceHubLogger("someuser/finetune")


def test_hf_logger_preflights_target_repo(monkeypatch):
    from libreyolo.training.loggers import HuggingFaceHubLogger

    seen = _patch_ambient_token(monkeypatch)
    HuggingFaceHubLogger("someuser/finetune")
    assert seen == [("someuser/finetune", True)]


def test_assert_can_push_rejects_foreign_namespace(monkeypatch):
    """A token that exists proves nothing about where it may write."""
    hub = pytest.importorskip("huggingface_hub")

    class FakeApi:
        def __init__(self, token=None):
            pass

        def whoami(self):
            return {"name": "someuser", "orgs": [{"name": "someorg"}]}

        def create_repo(self, *args, **kwargs):
            raise AssertionError("must be rejected before touching the Hub")

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    with pytest.raises(PermissionError, match="not your username"):
        hf_hub.assert_can_push("google/not-mine")

    # Own namespace and orgs are allowed through to repo creation.
    created = []
    FakeApi.create_repo = lambda self, repo_id, **kw: created.append(repo_id)
    hf_hub.assert_can_push("someorg/model")
    assert created == ["someorg/model"]


def test_hf_logger_pushes_best_checkpoint(tmp_path, monkeypatch):
    from libreyolo.training.loggers import HuggingFaceHubLogger

    _patch_ambient_token(monkeypatch)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    _make_yolo9_checkpoint(weights_dir / "best.pt")

    pushed = {}

    def fake_push(path, repo_id, **kwargs):
        pushed["path"] = str(path)
        pushed["repo_id"] = repo_id
        pushed["metrics"] = kwargs["metrics"]
        return "https://huggingface.co/someuser/finetune"

    monkeypatch.setattr(hf_hub, "push_checkpoint_to_hub", fake_push)

    hub_logger = HuggingFaceHubLogger("someuser/finetune")
    hub_logger.on_train_end(_train_end_event(tmp_path))

    assert pushed["repo_id"] == "someuser/finetune"
    assert pushed["path"].endswith("best.pt")
    # Non-numeric and boolean result values never reach the model card.
    assert pushed["metrics"] == {"best_mAP50": 0.5, "final_loss": 1.0}


def test_hf_logger_defaults_to_private(tmp_path, monkeypatch):
    """Unattended uploads must not publish by surprise (push_to_hub differs)."""
    from libreyolo.training.loggers import HuggingFaceHubLogger

    _patch_ambient_token(monkeypatch)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    _make_yolo9_checkpoint(weights_dir / "best.pt")

    seen = {}
    monkeypatch.setattr(
        hf_hub,
        "push_checkpoint_to_hub",
        lambda path, repo_id, **kwargs: seen.update(kwargs) or "url",
    )

    HuggingFaceHubLogger("someuser/finetune").on_train_end(
        _train_end_event(tmp_path)
    )
    assert seen["private"] is True


def test_hf_logger_no_checkpoint_is_a_noop(tmp_path, monkeypatch, caplog):
    from libreyolo.training.loggers import HuggingFaceHubLogger

    _patch_ambient_token(monkeypatch)
    called = []
    monkeypatch.setattr(
        hf_hub, "push_checkpoint_to_hub", lambda *a, **k: called.append(a)
    )

    hub_logger = HuggingFaceHubLogger("someuser/finetune")
    with caplog.at_level("WARNING", logger="libreyolo"):
        hub_logger.on_train_end(_train_end_event(tmp_path))

    assert called == []
    assert not hub_logger._disabled


# ---------------------------------------------------------------------------
# resolve_loggers string form
# ---------------------------------------------------------------------------


def test_resolve_loggers_hf_string(monkeypatch):
    from libreyolo.training.loggers import HuggingFaceHubLogger, resolve_loggers

    _patch_ambient_token(monkeypatch)
    resolved = resolve_loggers("hf:someuser/finetune")
    assert len(resolved) == 1
    assert isinstance(resolved[0], HuggingFaceHubLogger)
    assert resolved[0].repo_id == "someuser/finetune"


def test_resolve_loggers_bare_hf_name_errors():
    from libreyolo.training.loggers import resolve_loggers

    with pytest.raises(ValueError, match="hf:owner/repo"):
        resolve_loggers("hf")
