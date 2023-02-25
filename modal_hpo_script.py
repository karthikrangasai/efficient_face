# # `efficient-face` model training script using `modal`
#

import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, Tuple

import modal

# MODAL Setup
# COMMIT_HASH = "..."
MODAL_STUB = modal.Stub(
    name="efficient-face-wandb-HPO",
    image=modal.Image.debian_slim(python_version="3.8.16")
    .apt_install("git")
    .pip_install(
        # f"git+https://github.com/karthikrangasai/efficient_face.git@{COMMIT_HASH}#egg=efficient_face",
        "git+https://github.com/karthikrangasai/efficient_face.git#egg=efficient_face",
        "torchtext==0.14",
    ),
)

# A persistent shared volume will store trained model artefacts across Modal app runs.
# This is crucial as training runs are separate from the Gradio.app we run as a webhook.
# PERSISTENT_VOLUME = modal.SharedVolume().persist("training-checkpoints")
CHECKPOINTS_HOME = Path("/training_checkpoints")

CACHE_DIR = "/cache"
SHARED_VOLUME = modal.SharedVolume().persist("ciFAIR-dataset-cache")

NUM_HOURS = 1
NUM_MINUTES = NUM_HOURS * 60
TIMEOUT_SECONDS = int(NUM_MINUTES * 60)

# secret=
@MODAL_STUB.function(
    gpu="T4",
    cpu=0.5,
    secrets=[
        modal.Secret.from_name("wandb-api-key"),
        modal.Secret.from_name("efficient-face-wandb-sweep-id"),
    ],
    timeout=TIMEOUT_SECONDS,
    shared_volumes={CACHE_DIR: SHARED_VOLUME},
)
def hyperparameter_optimization() -> None:
    import dataclasses
    import wandb
    from pytorch_lightning import Trainer, seed_everything
    import pytorch_lightning.callbacks as plcb
    from pytorch_lightning.loggers.wandb import WandbLogger
    from efficient_face.data import ciFAIRDataModule
    from efficient_face.models import SoftmaxBasedModel, TripletLossBasedModel
    from torch.optim import Adam, Adadelta, Adagrad, RMSprop, Optimizer
    from torch_optimizer import Ranger, Lookahead, SGDW
    from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingWarmRestarts, StepLR, _LRScheduler, OneCycleLR

    def train_with_args() -> None:
        ############################## WANBD STUFF ####################################
        # Taken from https://github.com/borisdayma/lightning-kitti/blob/master/train.py
        # Sweep parameters
        hyperparameter_defaults = dict(
            learning_rate=0.00001,
            model_name="resnet18",
            triplet_strategy="VANILLA",
            optimizer="adam",
            lr_scheduler="constant",
        )

        wandb.init(config=hyperparameter_defaults, project="efficient_face")
        # Config parameters are automatically set by W&B sweep agent
        config = wandb.config
        ###############################################################################

        OPTIMS: Tuple[Type[Optimizer], ...] = Adam, Adadelta, Adagrad, RMSprop, Ranger, Lookahead, SGDW
        LRS: Tuple[Type[_LRScheduler], ...] = ConstantLR, CosineAnnealingWarmRestarts, StepLR, OneCycleLR

        OPTIM_MAPPING: Dict[str, Type[Optimizer]] = {cls.__name__: cls for cls in OPTIMS}
        OPTIM_TO_INIT_ARGS_MAPPING: Dict[Type[Optimizer], Dict[str, Any]] = {
            Adam: dict(weight_decay=1e-5),
            Adadelta: dict(weight_decay=1e-5),
            Adagrad: dict(weight_decay=1e-5),
            RMSprop: dict(weight_decay=1e-5),
            Ranger: dict(),
            Lookahead: dict(k=5, alpha=0.5),
            SGDW: dict(weight_decay=1e-5, nesterov=True),
        }
        LR_MAPPING: Dict[str, Type[_LRScheduler]] = {
            cls.__name__.lower()[:-2] if cls.__name__.lower().endswith("lr") else cls.__name__.lower(): cls
            for cls in LRS
        }
        LR_TO_INIT_MAPPING: Dict[Type[_LRScheduler], Dict[str, Any]] = {
            ConstantLR: dict(factor=2, total_iters=2),
            CosineAnnealingWarmRestarts: dict(T_0=125, eta_min=0.00001),
            StepLR: dict(step_size=250, gamma=0.1),
            OneCycleLR: dict(num_steps_arg="total_steps", num_steps_factor=1.0, max_lr=0.4, three_phase=True),
        }

        RANDOM_SEED = 3407
        seed_everything(RANDOM_SEED, workers=True)

        @dataclasses.dataclass
        class DataModuleConfig:
            batch_size: int = 128
            num_workers: int = 2

        @dataclasses.dataclass
        class TrainerConfig:
            num_sanity_val_steps: int = 0
            check_val_every_n_epoch: int = 2
            detect_anomaly: bool = True
            max_epochs: int = 5
            accelerator: str = "gpu"
            devices: int = 1

        @dataclasses.dataclass
        class ModelConfig:
            # Model Params
            model_name: str = "resnet18"  # "mobilenetv3_small_100"  # efficientnet_b0
            embedding_size: int = 128

            # Loss Function Params
            distance_metric: str = "L2"
            triplet_strategy: str = "VANILLA"
            miner_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
            loss_func_kwargs: Dict[str, Any] = dataclasses.field(default_factory=lambda: dict(margin=0.2))

            # Optimizer Params
            learning_rate: float = 0.2
            optimizer: Type[Optimizer] = Adam

            # Don't add `params` and `lr` arguments here
            optimizer_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
            lr_scheduler: Optional[Type[_LRScheduler]] = None

            # Change `num_steps_arg` to the argument name when changing LR Scheduler
            lr_scheduler_kwargs: Dict[str, Any] = dataclasses.field(
                default_factory=lambda: dict(num_steps_arg=None, num_steps_factor=1.0)
            )

            def __post_init__(self) -> None:
                self.optimizer_kwargs["lr"] = self.learning_rate

        datamodule_config = DataModuleConfig()
        model_config = ModelConfig(
            model_name=config.model_name,
            learning_rate=config.learning_rate,
            triplet_strategy=config.triplet_strategy,
            optimizer=OPTIM_MAPPING[config.optimizer],
            optimizer_kwargs=OPTIM_TO_INIT_ARGS_MAPPING[OPTIM_MAPPING[config.optimizer]],
            lr_scheduler=LR_MAPPING[config.lr_scheduler],
            lr_scheduler_kwargs=LR_TO_INIT_MAPPING[LR_MAPPING[config.lr_scheduler]],
        )
        trainer_config = TrainerConfig()

        # Setup Callbacks
        ENABLE_CHECKPOINTING = False
        CALLBACKS = [
            plcb.RichModelSummary(),
            plcb.RichProgressBar(),
            plcb.LearningRateMonitor(logging_interval="step"),
        ]

        if ENABLE_CHECKPOINTING:
            checkpoint = plcb.ModelCheckpoint(
                dirpath=CHECKPOINTS_HOME,
                filename="{epoch}--{val_loss:.3f}",
                monitor="val_loss",
                save_last=True,
                save_top_k=2,
                mode="min",
                auto_insert_metric_name=True,
                every_n_epochs=2,
            )
            CALLBACKS.append(checkpoint)

        # Setup logger
        wandb_enabled = bool(os.environ.get("WANDB_API_KEY"))
        LOGGER = WandbLogger(
            project="efficient_face",
            log_model=ENABLE_CHECKPOINTING,  # If Checkpointing enabled, then log the model.
            group="HPO",  # During trainig change to `model_config.model_name`
            id=None,  # Change when a run has failed to auto-resume it.
        )

        # Setup DataModule
        datamodule = ciFAIRDataModule(
            model_name=model_config.model_name,
            batch_size=datamodule_config.batch_size,
            num_workers=datamodule_config.num_workers,
        )

        # Setup Model
        model = TripletLossBasedModel(
            model_name=model_config.model_name,
            embedding_size=model_config.embedding_size,
            distance_metric=model_config.distance_metric,
            triplet_strategy=model_config.triplet_strategy,
            miner_kwargs=model_config.miner_kwargs,
            loss_func_kwargs=model_config.loss_func_kwargs,
            learning_rate=model_config.learning_rate,
            optimizer=model_config.optimizer,
            optimizer_kwargs=model_config.optimizer_kwargs,
            lr_scheduler=model_config.lr_scheduler,
            lr_scheduler_kwargs=model_config.lr_scheduler_kwargs,
        )

        trainer = Trainer(
            num_sanity_val_steps=0,
            check_val_every_n_epoch=2,
            detect_anomaly=True,
            deterministic=True,
            max_epochs=trainer_config.max_epochs,
            accelerator=trainer_config.accelerator,
            devices=trainer_config.devices,
            logger=LOGGER,
            callbacks=CALLBACKS,
        )

        trainer.fit(model, datamodule=datamodule)
        if ENABLE_CHECKPOINTING:
            print(checkpoint.best_model_path)

    sweep_id = os.environ["EFFICIENT_FACE_WANDB_SWEEP_ID"]
    wandb.agent(sweep_id=sweep_id, function=train_with_args, project="efficient_face", count=1)


if __name__ == "__main__":
    # ### Detaching our training run
    #
    # `MODAL_GPU=T4 modal run --detach modal_hpo_script.py::MODAL_STUB`
    #
    with MODAL_STUB.run():
        hyperparameter_optimization.call()
