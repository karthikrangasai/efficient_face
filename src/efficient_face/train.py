from argparse import ArgumentParser
from typing import Any, Dict

import pytorch_lightning.callbacks as plcb
import yaml
from flash import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger

from efficient_face.data import ciFAIRDataModule
from efficient_face.models import MODEL_TYPE


def train(train_args: Dict[str, Any]) -> None:
    seed_everything(train_args["random_seed"])
    triplet_strategy = train_args["model"]["triplet_strategy"]
    message = f"Loss configuration for {triplet_strategy} is not defined."
    assert triplet_strategy not in ["ADAPTIVE", "ASSORTED", "CONSTELLATION"], message

    datamodule = ciFAIRDataModule(**train_args["data_module"])

    model_type: str = train_args["model"].pop("model_type")
    assert model_type in ["Normal", "SAM"], f"[ERROR]: {model_type} is not yet implementated."
    model_cls = MODEL_TYPE[model_type]

    _optimizer: Dict = train_args["model"].pop("optimizer")
    _lr_scheduler: Dict = train_args["model"].pop("lr_scheduler")

    model = model_cls(
        **train_args["model"],
        optimizer=(_optimizer["name"], _optimizer["kwargs"]),
        lr_scheduler=(_lr_scheduler["name"], _lr_scheduler["kwargs"]),
    )

    # init callbacks
    callbacks = [
        plcb.LearningRateMonitor(**train_args["callbacks"]["lr_monitor"]),
        plcb.ModelCheckpoint(**train_args["callbacks"]["model_checkpoint"]),
        plcb.RichModelSummary(),
        plcb.RichProgressBar(),
    ]

    # init logger
    logger = WandbLogger(**train_args["logger"])

    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        **train_args["trainer"],
    )
    trainer.fit(model, datamodule=datamodule)


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

    train(train_args=train_args)
