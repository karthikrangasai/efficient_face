from pathlib import Path
from typing import Type

import pytest
import torch
from flash import Task, Trainer
from pytorch_lightning import seed_everything

from efficient_face.data import ciFAIRDataModule
from efficient_face.models import SoftmaxBasedTask, TripletLossBasedTask


def num_gpus() -> int:
    if torch.cuda.is_available():
        return 1
    return 0


@pytest.mark.parametrize(("model_type", "batch_size"), [(SoftmaxBasedTask, 4), (TripletLossBasedTask, 4)])
def test_model(
    random_dataset_path: Path, tmpdir: pytest.TempdirFactory, model_type: Type[Task], batch_size: int
) -> None:
    seed_everything(1234)
    datamodule = ciFAIRDataModule.load_ciFAIR10(root=random_dataset_path, batch_size=batch_size)
    model = model_type()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=5,
        limit_val_batches=5,
        gpus=num_gpus(),
        fast_dev_run=True,
    )
    trainer.fit(model, datamodule=datamodule)
    # trainer.validate(model, datamodule=datamodule)


# @pytest.mark.parametrize(
#     ("model_type", "batch_size"), [(EfficientFaceModel, 4), (SAMEfficientFaceModel, 4), (ESAMEfficientFaceModel, 8)]
# )
# def test_model_with_hf(random_hf_dataset_path, tmpdir, model_type, batch_size):
#     seed_everything(1234)
#     datamodule = EfficientFaceDataModule.from_hf_datasets(train_folder=random_hf_dataset_path, batch_size=batch_size)
#     model = model_type()
#     trainer = Trainer(
#         default_root_dir=tmpdir,
#         max_epochs=2,
#         limit_train_batches=5,
#         limit_val_batches=5,
#         gpus=num_gpus(),
#     )
#     trainer.fit(model, datamodule=datamodule)


# @pytest.mark.parametrize(
#     ("model_type", "batch_size"), [(EfficientFaceModel, 4), (SAMEfficientFaceModel, 4), (ESAMEfficientFaceModel, 8)]
# )
# def test_model_on_cifar10(random_cifar10_path, tmpdir, model_type, batch_size):
#     seed_everything(1234)
#     datamodule = EfficientFaceDataModule.from_cifar10(train_folder=random_cifar10_path, batch_size=batch_size)
#     model = model_type()
#     trainer = Trainer(
#         default_root_dir=tmpdir,
#         max_epochs=2,
#         limit_train_batches=5,
#         limit_val_batches=5,
#         gpus=num_gpus(),
#     )
#     trainer.fit(model, datamodule=datamodule)
