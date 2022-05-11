from pathlib import Path

import numpy as np
import pytest
from datasets import load_dataset
from flash import DataKeys
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
    for i in range(10):
        if (i % 2) == 0:
            d = temp_path / f"class{i/2}"
            d.mkdir()
            for j in range(20):
                _rand_image().save(d / "image_{j}.jpg")
        else:
            d = temp_path / f"text_file_{i//2}.txt"
            d.write_text("Hello, World.")
    return temp_path


@pytest.fixture(scope="session")
def random_hf_dataset_path(random_dataset_path: Path):
    parent_path: Path = random_dataset_path.parent
    temp_path: Path = parent_path / "hf_data"
    temp_path.mkdir()

    dataset = load_dataset("imagefolder", data_dir=str(random_dataset_path), split="train")

    def transforms(examples):
        examples[DataKeys.INPUT] = [image.convert("RGB") for image in examples["image"]]
        examples[DataKeys.TARGET] = examples["label"]
        return examples

    dataset = dataset.map(
        function=transforms, batched=True, remove_columns=["image", "label"], batch_size=5, writer_batch_size=5
    )
    dataset.save_to_disk(str(temp_path))
    return str(temp_path)
