from argparse import ArgumentParser
from typing import Any, Callable, Dict

import pytorch_lightning.callbacks as plcb
import pytorch_lightning.loggers as pll
import pytorch_lightning.plugins.environments.slurm_environment as slurm
import yaml
from flash import Trainer
from pytorch_lightning import seed_everything

from efficient_face.data import EfficientFaceDataModule
from efficient_face.model import EfficientFaceModel, SAMEfficientFaceModel

MODEL_TYPE = {"Normal": EfficientFaceModel, "SAM": SAMEfficientFaceModel}
DATAMODULE_TYPE = {
    "Normal": EfficientFaceDataModule.from_label_class_subfolders,
    "HF": EfficientFaceDataModule.from_hf_datasets,
}


def train(train_args: Dict[str, Any]) -> None:
    seed_everything(train_args["random_seed"])
    triplet_strategy = train_args["model"]["triplet_strategy"]
    message = f"Loss configuration for {triplet_strategy} is not defined."
    assert triplet_strategy not in ["ADAPTIVE", "ASSORTED", "CONSTELLATION"], message

    datamodule_type: str = train_args["data_module"].pop("data_module_type")
    assert datamodule_type in DATAMODULE_TYPE.keys(), f"[ERROR]: {datamodule_type} is not yet implementated."
    datamodule_method = DATAMODULE_TYPE[datamodule_type]
    datamodule = datamodule_method(**train_args["data_module"])

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
    logger = pll.WandbLogger(**train_args["logger"])

    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        plugins=[slurm.SLURMEnvironment(auto_requeue=True)],
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
