"""Tests for dataset config loading safety and path resolution."""

from pathlib import Path, PurePosixPath, PureWindowsPath

import os
import pytest
import yaml
from PIL import Image

from libreyolo.data.utils import (
    BUILTIN_DATASETS_DIR,
    get_img_files,
    get_coco_image_dir,
    img2label_paths,
    load_data_config,
    resolve_default_coco_image_dir,
)

pytestmark = pytest.mark.unit


def _write_dataset_yaml(tmp_path: Path) -> Path:
    yaml_path = tmp_path / "scripted.yaml"
    marker_path = tmp_path / "marker.txt"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {tmp_path / 'dataset'}",
                "train: images/train",
                "val: images/val",
                "download: |",
                f'  Path(r"{marker_path}").write_text("ran")',
            ]
        )
    )
    return yaml_path


def test_load_data_config_blocks_embedded_scripts_by_default(tmp_path):
    yaml_path = _write_dataset_yaml(tmp_path)
    marker_path = tmp_path / "marker.txt"

    load_data_config(str(yaml_path), autodownload=True)

    assert not marker_path.exists()


def test_load_data_config_can_opt_in_to_embedded_scripts(tmp_path):
    yaml_path = _write_dataset_yaml(tmp_path)
    marker_path = tmp_path / "marker.txt"

    load_data_config(str(yaml_path), autodownload=True, allow_scripts=True)

    assert marker_path.read_text() == "ran"


def test_embedded_scripts_map_common_yolo_helper_imports(tmp_path, monkeypatch):
    import libreyolo.data.utils as data_utils

    marker_path = tmp_path / "marker.txt"
    yaml_path = tmp_path / "scripted.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {tmp_path / 'dataset'}",
                "train: images/train",
                "val: images/val",
                "download: |",
                "  from ultralytics.utils.downloads import download",
                "  from ultralytics.utils import ASSETS_URL",
                f'  Path(r"{marker_path}").write_text(ASSETS_URL)',
                "  download([ASSETS_URL + '/labels.zip'], dir=path)",
            ]
        )
    )

    monkeypatch.setattr(data_utils, "ASSETS_URL", "libreyolo-assets")
    called = {}

    def fake_download(urls, dir=".", unzip=True, delete=True, threads=1):
        called["urls"] = urls
        called["dir"] = dir

    monkeypatch.setattr(data_utils, "download", fake_download)

    load_data_config(str(yaml_path), autodownload=True, allow_scripts=True)

    assert marker_path.read_text() == "libreyolo-assets"
    assert called == {
        "urls": ["libreyolo-assets/labels.zip"],
        "dir": tmp_path / "dataset",
    }


def test_builtin_dataset_yamls_do_not_use_ultralytics_assets():
    forbidden = ("github.com/ultralytics/assets", "ASSETS_URL", "coco2017labels.zip")
    offenders = {}

    for yaml_path in BUILTIN_DATASETS_DIR.glob("*.yaml"):
        text = yaml_path.read_text(encoding="utf-8")
        matches = [term for term in forbidden if term in text]
        if matches:
            offenders[yaml_path.name] = matches

    assert offenders == {}


def test_known_classify_datasets_are_libre_hosted():
    # Named classification datasets must resolve to LibreYOLO-controlled hosting,
    # rebuilt from clean upstream sources — never a competitor's release assets.
    # (The download-script import shim in data.utils is a separate, sanctioned
    # compatibility layer and is intentionally not covered here.)
    from libreyolo.data.classify_dataset import _KNOWN_DATASETS

    offenders = {
        name: url
        for name, url in _KNOWN_DATASETS.items()
        if "huggingface.co/datasets/LibreYOLO" not in url
    }
    assert offenders == {}, f"non-LibreYOLO-hosted known datasets: {offenders}"


def test_load_data_config_resolves_directory_test_split(tmp_path):
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "test" / "images"
    labels_dir = dataset_root / "test" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    image_path = images_dir / "sample.jpg"
    Image.new("RGB", (32, 32), color="white").save(image_path)
    label_path = labels_dir / "sample.txt"
    label_path.write_text("0 0.5 0.5 0.25 0.25\n")

    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "names": ["marble"],
                "nc": 1,
            }
        )
    )

    config = load_data_config(str(yaml_path), autodownload=False)

    assert config["test"] == str(images_dir)
    assert config["test_img_files"] == [image_path]
    assert config["test_label_files"] == [label_path]


def test_load_data_config_resolves_coco_annotation_paths(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "annotations").mkdir(parents=True)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "train": "images/train",
                "val": "images/val",
                "annotations": {
                    "train": "annotations/train.json",
                    "val": str(dataset_root / "annotations" / "valid.json"),
                },
                "names": ["marble"],
                "nc": 1,
            }
        )
    )

    config = load_data_config(str(yaml_path), autodownload=False)

    assert config["train_annotation_file"] == str(
        dataset_root / "annotations" / "train.json"
    )
    assert config["val_annotation_file"] == str(
        dataset_root / "annotations" / "valid.json"
    )


@pytest.mark.parametrize("split_value", [["images/a", "images/b"], "train.txt", "TRAIN.TXT"])
def test_coco_image_dir_requires_single_directory(split_value):
    with pytest.raises(ValueError, match="Native COCO JSON loading expects"):
        get_coco_image_dir({"train": split_value}, "train", "train2017")


def test_resolve_default_coco_image_dir_prefers_standard_images_layout(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train2017").mkdir(parents=True)

    assert (
        resolve_default_coco_image_dir(
            dataset_root,
            "train",
            "instances_train2017.json",
        )
        == "images/train2017"
    )
    assert (
        resolve_default_coco_image_dir(
            dataset_root,
            "val",
            "instances_val2017.json",
        )
        == "val2017"
    )


@pytest.mark.parametrize(
    ("image_path", "expected_label"),
    [
        (
            PurePosixPath("/home/user/dataset/images/train/sample.jpg"),
            "/home/user/dataset/labels/train/sample.txt",
        ),
        (
            PurePosixPath("/Users/user/dataset/images/val/sample.jpg"),
            "/Users/user/dataset/labels/val/sample.txt",
        ),
        (
            PureWindowsPath(r"C:\Users\user\dataset\images\test\sample.jpg"),
            r"C:\Users\user\dataset\labels\test\sample.txt",
        ),
    ],
)
def test_img2label_paths_handles_platform_path_styles(image_path, expected_label):
    labels = img2label_paths([image_path])

    assert str(labels[0]).replace("\\", "/") == expected_label.replace("\\", "/")


def test_get_img_files_txt_keeps_existing_and_unmaterialized_entries(tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (16, 16), color="white").save(image_path)
    missing_path = tmp_path / "later.jpg"
    txt_path = tmp_path / "images.txt"
    txt_path.write_text(f"{image_path.name}\n{missing_path.name}\nnotes.txt\n")

    assert get_img_files(txt_path) == [missing_path, image_path]


def test_get_img_files_names_a_missing_directory(tmp_path):
    from libreyolo.data.utils import get_img_files

    with pytest.raises(FileNotFoundError, match="Image directory not found"):
        get_img_files(tmp_path / "datasets" / "coco5000" / "images" / "train")
    with pytest.raises(ValueError, match="Unsupported path format"):
        (tmp_path / "notes.md").write_text("x")
        get_img_files(tmp_path / "notes.md")



def _make_split_dataset(root: Path) -> None:
    for split in ("train", "valid"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(img_dir / "a.jpg")
        (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")


@pytest.fixture
def fake_datasets_dir(tmp_path, monkeypatch):
    """A populated ~/datasets and a CWD elsewhere, like a real user machine."""
    import libreyolo.data.utils as data_utils

    home = tmp_path / "home"
    datasets_dir = home / "datasets"
    (datasets_dir / "other" / "train" / "images").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(data_utils, "DATASETS_DIR", datasets_dir)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return datasets_dir


@pytest.mark.parametrize("rel", [".", "./"])
def test_path_dot_resolves_to_yaml_folder(tmp_path, fake_datasets_dir, rel):
    ds = tmp_path / "hf_cache" / "cells-uyemf"
    _make_split_dataset(ds)
    yaml_path = ds / "data.yaml"
    yaml_path.write_text(
        f"path: {rel}\ntrain: train/images\nval: valid/images\nnc: 1\nnames: [cell]\n"
    )

    cfg = load_data_config(str(yaml_path), autodownload=False)

    assert Path(cfg["path"]) == ds.resolve()
    assert len(cfg["train_img_files"]) == 1
    assert len(cfg["val_img_files"]) == 1


def test_path_subfolder_resolves_to_yaml_folder(tmp_path, fake_datasets_dir):
    ds = tmp_path / "proj"
    _make_split_dataset(ds / "sub")
    yaml_path = ds / "data.yaml"
    yaml_path.write_text("path: ./sub\ntrain: train/images\nval: valid/images\nnames: [cell]\n")

    cfg = load_data_config(str(yaml_path), autodownload=False)

    assert Path(cfg["path"]) == (ds / "sub").resolve()
    assert len(cfg["train_img_files"]) == 1


def test_named_path_still_falls_back_to_datasets_dir(tmp_path, fake_datasets_dir):
    _make_split_dataset(fake_datasets_dir / "mine")
    yaml_path = tmp_path / "mine.yaml"
    yaml_path.write_text("path: mine\ntrain: train/images\nval: valid/images\nnames: [x]\n")

    cfg = load_data_config(str(yaml_path), autodownload=False)

    assert Path(cfg["path"]) == fake_datasets_dir / "mine"
    assert len(cfg["train_img_files"]) == 1


def test_builtin_yaml_path_points_into_datasets_dir(fake_datasets_dir):
    cfg = load_data_config("coco8.yaml", autodownload=False)
    assert Path(cfg["path"]) == fake_datasets_dir / "coco8"


def test_zero_images_warning_names_resolved_path(tmp_path, fake_datasets_dir, caplog):
    ds = tmp_path / "empty"
    (ds / "train" / "images").mkdir(parents=True)
    yaml_path = ds / "data.yaml"
    yaml_path.write_text("path: .\ntrain: train/images\nnames: [x]\n")

    with caplog.at_level("WARNING"):
        load_data_config(str(yaml_path), autodownload=False)

    expected = str(ds.resolve() / "train" / "images")
    assert any("No images found" in r.getMessage() and expected in r.getMessage()
               for r in caplog.records)


def test_zip_download_never_deletes_existing_dest(tmp_path, monkeypatch):
    """A ZIP whose single root folder has another name must not wipe dest."""
    import zipfile

    import libreyolo.data.utils as data_utils

    src_zip = tmp_path / "ds.zip"
    with zipfile.ZipFile(src_zip, "w") as zf:
        zf.writestr("other-name/train/images/a.jpg", b"x")
        zf.writestr("other-name/data.yaml", b"from zip")

    def fake_download(url, dest, chunk_size=8192):
        dest.write_bytes(src_zip.read_bytes())

    monkeypatch.setattr(data_utils, "_download_file", fake_download)

    project = tmp_path / "parent" / "project"
    project.mkdir(parents=True)
    (project / "data.yaml").write_text("mine")
    (project / "notes.txt").write_text("keep me")
    sibling = tmp_path / "parent" / "other-name"
    sibling.mkdir()
    (sibling / "untouched.txt").write_text("sibling")

    data_utils._download_and_extract_zip("https://example.invalid/ds.zip", project)

    assert (project / "notes.txt").read_text() == "keep me"
    assert (project / "data.yaml").read_text() == "mine"  # not overwritten
    assert (project / "train" / "images" / "a.jpg").exists()
    assert (sibling / "untouched.txt").read_text() == "sibling"
    assert not list((tmp_path / "parent").glob(".libreyolo-extract-*"))


def test_zero_images_warning_includes_search_error(tmp_path, fake_datasets_dir, caplog):
    ds = tmp_path / "partial"
    _make_split_dataset(ds)
    yaml_path = ds / "data.yaml"
    yaml_path.write_text(
        "path: .\ntrain: [train/images, train/missing]\nval: valid/images\nnames: [x]\n"
    )

    with caplog.at_level("WARNING"):
        load_data_config(str(yaml_path), autodownload=False)

    msgs = [r.getMessage() for r in caplog.records if "No images found" in r.getMessage()]
    assert msgs and "missing" in msgs[0]


def _fake_zip_download(monkeypatch, tmp_path, entries):
    import zipfile

    import libreyolo.data.utils as data_utils

    src_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(src_zip, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)

    def fake_download(url, dest, chunk_size=8192):
        dest.write_bytes(src_zip.read_bytes())

    monkeypatch.setattr(data_utils, "_download_file", fake_download)
    return data_utils


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_zip_download_does_not_follow_symlinks_out_of_dest(tmp_path, monkeypatch):
    data_utils = _fake_zip_download(
        monkeypatch, tmp_path, {"bundle/assets/payload.txt": b"x", "bundle/ok.txt": b"y"}
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    project = tmp_path / "parent" / "project"
    project.mkdir(parents=True)
    (project / "assets").symlink_to(outside, target_is_directory=True)

    data_utils._download_and_extract_zip("https://example.invalid/evil.zip", project)

    assert not (outside / "payload.txt").exists()
    assert (project / "ok.txt").exists()


def test_zip_download_rejects_parent_dir_root(tmp_path, monkeypatch):
    data_utils = _fake_zip_download(monkeypatch, tmp_path, {"../payload.txt": b"x"})
    parent = tmp_path / "parent"
    project = parent / "project"
    project.mkdir(parents=True)
    neighbour = parent / "neighbour.txt"
    neighbour.write_text("stay here")

    data_utils._download_and_extract_zip("https://example.invalid/evil.zip", project)

    assert neighbour.read_text() == "stay here"
    assert not (project / "neighbour.txt").exists()
    assert not list(parent.glob(".libreyolo-extract-*"))
