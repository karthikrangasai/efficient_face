from pathlib import Path
from typing import Tuple, Type

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything

from efficient_face.data import ciFAIRDataModule
from efficient_face.models import SoftmaxBasedModel, TripletLossBasedModel


def num_gpus() -> int:
    if torch.cuda.is_available():
        return 1
    return 0


@pytest.mark.parametrize(("model_type", "batch_size"), [(SoftmaxBasedModel, 4), (TripletLossBasedModel, 4)])
def test_model(
    random_dataset_and_logs_path: Tuple[Path, Path], model_type: Type[LightningModule], batch_size: int
) -> None:
    data_dir, logs_dir = random_dataset_and_logs_path
    seed_everything(1234)
    datamodule = ciFAIRDataModule(batch_size=batch_size)
    model = model_type()
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=logs_dir.as_posix(),
        max_epochs=2,
        limit_train_batches=5,
        limit_val_batches=5,
        accelerator="cpu",
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)
    # trainer.validate(model, datamodule=datamodule)
