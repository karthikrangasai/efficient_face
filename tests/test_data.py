from typing import Dict, List, Tuple

from flash.core.data.io.input import DataKeys
from pytorch_lightning import seed_everything
from torch import Tensor

from efficient_face.datasets import EfficientFaceDataModule


def assert_batch_values(batch: Dict):
    assert DataKeys.INPUT in batch.keys()
    assert DataKeys.TARGET in batch.keys()

    assert isinstance(batch[DataKeys.INPUT], Tensor)
    assert isinstance(batch[DataKeys.TARGET], Tensor)


def test_train_data_loading(train_folder):
    seed_everything(1234)

    datamodule = EfficientFaceDataModule.from_label_class_subfolders(
        train_folder=train_folder,
        batch_size=4,
    )

    train_dataloader = datamodule.train_dataloader()
    batch: Dict = next(iter(train_dataloader))
    assert_batch_values(batch)


def test_val_data_loading(val_folder):
    seed_everything(1234)

    datamodule = EfficientFaceDataModule.from_label_class_subfolders(
        val_folder=val_folder,
        batch_size=4,
    )

    val_dataloader = datamodule.val_dataloader()
    batch: Dict = next(iter(val_dataloader))
    assert_batch_values(batch)
