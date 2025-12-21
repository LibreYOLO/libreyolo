import pytest
from pathlib import Path
import shutil
import os

@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def test_image(project_root):
    image_path = project_root / "media" / "test_image_1_creative_commons.jpg"
    if not image_path.exists():
        pytest.skip("Test image not found")
    return str(image_path)

@pytest.fixture(scope="session")
def weights_dir(project_root):
    # Directory where converted weights are stored
    path = project_root / "weights"
    path.mkdir(exist_ok=True)
    return path

@pytest.fixture(scope="session")
def temp_weights_dir(project_root):
    # Directory for original ultralytics weights
    path = project_root / "temporary"
    path.mkdir(exist_ok=True)
    return path

@pytest.fixture(scope="session")
def output_dir(project_root):
    # For test artifacts
    path = project_root / "tests" / "output"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

