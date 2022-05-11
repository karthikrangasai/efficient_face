import pytest
import torch
from flash import Trainer
from pytorch_lightning import seed_everything

from efficient_face.data import EfficientFaceDataModule
from efficient_face.model import EfficientFaceModel, SAMEfficientFaceModel


def num_gpus():
    if torch.cuda.is_available():
        return 1
    return 0


@pytest.mark.parametrize("model_type", [EfficientFaceModel, SAMEfficientFaceModel])
def test_model(random_dataset_path, tmpdir, model_type):
    seed_everything(1234)
    datamodule = EfficientFaceDataModule.from_label_class_subfolders(
        train_folder=random_dataset_path, val_folder=random_dataset_path, batch_size=4
    )
    model = model_type()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        limit_train_batches=10,
        gpus=num_gpus(),
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)


@pytest.mark.parametrize("model_type", [EfficientFaceModel, SAMEfficientFaceModel])
def test_model_with_hf(random_hf_dataset_path, tmpdir, model_type):
    seed_everything(1234)
    datamodule = EfficientFaceDataModule.from_hf_datasets(
        train_folder=random_hf_dataset_path, val_folder=random_hf_dataset_path, batch_size=4
    )
    model = model_type()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        limit_train_batches=10,
        gpus=num_gpus(),
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)
