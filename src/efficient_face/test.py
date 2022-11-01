from flash import Trainer
from pytorch_lightning import seed_everything

from efficient_face.data import EfficientFaceDataModule
from efficient_face.models.esam_efficient_face_model import ESAMEfficientFaceModel

seed_everything(69)
# datamodule = EfficientFaceDataModule.from_cifar10(train_folder="data", batch_size=8)
datamodule = EfficientFaceDataModule.from_label_class_subfolders(
    train_folder="data/data_example", val_folder="data/data_example", batch_size=8
)
model = ESAMEfficientFaceModel(
    triplet_strategy="FOCAL",
    optimizer=("sgdw", {"momentum": 0.9, "weight_decay": 0.00001, "dampening": 0.0, "nesterov": False}),
    lr_scheduler=(
        "cycliclr",
        {
            "base_lr": 0.001,
            "max_lr": 0.1,
            "step_size_up": 5,
            "step_size_down": 5,
            "mode": "triangular",
            "gamma": 1.0,
            "scale_fn": None,
            "scale_mode": "iterations",
            "cycle_momentum": True,
            "base_momentum": 0.8,
            "max_momentum": 0.9,
        },
    ),
)
trainer = Trainer(max_epochs=1, log_every_n_steps=1)
trainer.fit(model, datamodule=datamodule)
