import pytest
import torch
from flash import Trainer
from pytorch_lightning import seed_everything

from efficient_face.datasets import EfficientFaceDataModule
from efficient_face.model import EfficientFaceModel, SAMEfficientFaceModel


def num_gpus():
    if torch.cuda.is_available():
        return 1
    return 0


@pytest.mark.parametrize("model_type", [EfficientFaceModel, SAMEfficientFaceModel])
def test_model(train_folder, val_folder, tmpdir, model_type):
    seed_everything(1234)
    datamodule = EfficientFaceDataModule.from_label_class_subfolders(
        train_folder=train_folder, val_folder=val_folder, batch_size=4
    )
    model = model_type()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        max_epochs=3,
        limit_train_batches=10,
        gpus=num_gpus(),
        val_check_interval=0,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)
