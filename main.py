import torch
from flash import Trainer
from pytorch_lightning import seed_everything

from efficient_face.data import ciFAIRDataModule
from efficient_face.models import SoftmaxBasedTask, TripletLossBasedTask


def num_gpus() -> int:
    if torch.cuda.is_available():
        return 1
    return 0


seed_everything(1234)
datamodule = ciFAIRDataModule.load_ciFAIR10(batch_size=4)
model = SoftmaxBasedTask()
trainer = Trainer(max_epochs=2, limit_train_batches=5, limit_val_batches=5, accelerator="cpu")
trainer.fit(model, datamodule=datamodule)
# trainer.validate(model, datamodule=datamodule)
