import pytest
import torch
from flash import Trainer
from pytorch_lightning import seed_everything

from efficient_face.data import EfficientFaceDataModule
from efficient_face.models import EfficientFaceModel, ESAMEfficientFaceModel, SAMEfficientFaceModel


def num_gpus():
    if torch.cuda.is_available():
        return 1
    return 0


@pytest.mark.parametrize(
    ("model_type", "batch_size"), [(EfficientFaceModel, 4), (SAMEfficientFaceModel, 4), (ESAMEfficientFaceModel, 8)]
)
def test_model(random_dataset_path, tmpdir, model_type, batch_size):
    seed_everything(1234)
    datamodule = EfficientFaceDataModule.from_label_class_subfolders(
        train_folder=random_dataset_path, batch_size=batch_size
    )
    model = model_type()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=5,
        limit_val_batches=5,
        gpus=num_gpus(),
    )
    trainer.fit(model, datamodule=datamodule)


@pytest.mark.parametrize(
    ("model_type", "batch_size"), [(EfficientFaceModel, 4), (SAMEfficientFaceModel, 4), (ESAMEfficientFaceModel, 8)]
)
def test_model_with_hf(random_hf_dataset_path, tmpdir, model_type, batch_size):
    seed_everything(1234)
    datamodule = EfficientFaceDataModule.from_hf_datasets(train_folder=random_hf_dataset_path, batch_size=batch_size)
    model = model_type()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=5,
        limit_val_batches=5,
        gpus=num_gpus(),
    )
    trainer.fit(model, datamodule=datamodule)


@pytest.mark.parametrize(
    ("model_type", "batch_size"), [(EfficientFaceModel, 4), (SAMEfficientFaceModel, 4), (ESAMEfficientFaceModel, 8)]
)
def test_model_on_cifar10(random_cifar10_path, tmpdir, model_type, batch_size):
    seed_everything(1234)
    datamodule = EfficientFaceDataModule.from_cifar10(train_folder=random_cifar10_path, batch_size=batch_size)
    model = model_type()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=5,
        limit_val_batches=5,
        gpus=num_gpus(),
    )
    trainer.fit(model, datamodule=datamodule)
