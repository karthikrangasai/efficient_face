from argparse import ArgumentParser
from typing import Dict

import yaml
from flash import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments.slurm_environment import SLURMEnvironment

from efficient_face.datasets import EfficientFaceDataModule
from efficient_face.model import EfficientFaceModel, SAMEfficientFaceModel

MODEL_TYPE = {"Normal": EfficientFaceModel, "SAM": SAMEfficientFaceModel, "ESAM": None}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train_args_file",
        required=True,
        type=str,
        help="Path to the training arguments yaml file.",
    )
    args = vars(parser.parse_args())

    with open(args["train_args_file"]) as f:
        train_args = yaml.safe_load(f)

    seed_everything(train_args["random_seed"])
    triplet_strategy = train_args["model"]["triplet_strategy"]
    message = f"Loss configuration for {triplet_strategy} is not defined."
    assert triplet_strategy not in ["ADAPTIVE", "ASSORTED", "CONSTELLATION"], message

    datamodule = EfficientFaceDataModule.from_label_class_subfolders(**train_args["data_module"])

    model_type: str = train_args["model"].pop("model_type")
    _optimizer: Dict = train_args["model"].pop("optimizer")
    _lr_scheduler: Dict = train_args["model"].pop("lr_scheduler")

    assert model_type in ["Normal", "SAM"], f"Implementatation for {model_type} is not yet completed"
    model = MODEL_TYPE[model_type](
        **train_args["model"],
        optimizer=(_optimizer["name"], _optimizer["kwargs"]),
        lr_scheduler=(_lr_scheduler["name"], _lr_scheduler["kwargs"]),
    )

    # init callbacks
    callbacks = [
        LearningRateMonitor(**train_args["callbacks"]["lr_monitor"]),
        ModelCheckpoint(**train_args["callbacks"]["model_checkpoint"]),
        RichModelSummary(),
        RichProgressBar(),
    ]

    # init logger
    logger = WandbLogger(**train_args["logger"])

    trainer = Trainer(
        callbacks=callbacks, logger=logger, plugins=[SLURMEnvironment(auto_requeue=True)], **train_args["trainer"]
    )
    trainer.fit(model, datamodule=datamodule)
