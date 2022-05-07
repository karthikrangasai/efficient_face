import torch
from flash import Trainer
from pytorch_lightning import seed_everything

from efficient_face.datasets import EfficientFaceDataModule
from efficient_face.model import EfficientFaceModel, SAMEfficientFaceModel


def num_gpus():
    if torch.cuda.is_available():
        return 1
    return 0


def model_pipeline(_train_folder, _val_folder, temp_dir, model):
    seed_everything(1234)
    datamodule = EfficientFaceDataModule.from_label_class_subfolders(
        train_folder=_train_folder, val_folder=_val_folder, batch_size=4
    )
    trainer = Trainer(
        default_root_dir=temp_dir,
        fast_dev_run=True,
        max_epochs=3,
        limit_train_batches=10,
        gpus=num_gpus(),
        val_check_interval=0,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)


def test_model(train_folder, val_folder, tmpdir):
    model_pipeline(train_folder, val_folder, tmpdir, EfficientFaceModel())


def test_sam_model(train_folder, val_folder, tmpdir):
    model_pipeline(train_folder, val_folder, tmpdir, SAMEfficientFaceModel())
