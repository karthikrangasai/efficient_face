from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def pytest_addoption(parser):
    parser.addoption(
        "--train_folder",
        action="store",
        type=str,
        default=None,
        help="Path to folder of the training data",
        required=False,
    )
    parser.addoption(
        "--val_folder",
        action="store",
        type=str,
        default=None,
        help="Path to folder of the validation/testing data",
        required=False,
    )


# Taken from https://github.com/PyTorchLightning/lightning-flash/blob/0.7.4/tests/image/classification/test_data.py
def _rand_image():
    _size = np.random.choice([196, 244])
    size = (_size, _size)
    return Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype="uint8"))


@pytest.fixture(scope="session")
def random_dataset_path(tmp_path_factory: Path):
    temp_path: Path = tmp_path_factory.mktemp("data")
    for i in range(5):
        d = temp_path / f"class{i}"
        d.mkdir()
        for j in range(20):
            _rand_image().save(d / "image_{j}.jpg")
    return temp_path


@pytest.fixture(scope="session")
def train_folder(request, random_dataset_path):
    _train_folder = request.config.getoption("--train_folder")
    if _train_folder is None:
        return random_dataset_path
    return _train_folder


@pytest.fixture(scope="session")
def val_folder(request, random_dataset_path):
    _val_folder = request.config.getoption("--val_folder")
    if _val_folder is None:
        return random_dataset_path
    return _val_folder
