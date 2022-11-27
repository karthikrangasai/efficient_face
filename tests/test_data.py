from pathlib import Path
from typing import List, Tuple

from pytorch_lightning import seed_everything
from torch import Tensor

from efficient_face.data import ciFAIRDataModule


def assert_batch_values(batch: List[Tensor]) -> None:
    assert len(batch) == 2

    assert isinstance(batch[0], Tensor)
    assert isinstance(batch[1], Tensor)


def test_data_loading() -> None:
    seed_everything(1234)

    datamodule = ciFAIRDataModule(batch_size=4)
    datamodule.prepare_data()
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()  # type: ignore
    batch: List[Tensor] = next(iter(train_dataloader))
    assert_batch_values(batch)

    val_dataloader = datamodule.val_dataloader()  # type: ignore
    batch = next(iter(val_dataloader))
    assert_batch_values(batch)
