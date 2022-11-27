from typing import Any, Dict, List, Optional, Type, Union

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from efficient_face.losses import DISTANCES, LOSS_CONFIGURATION
from efficient_face.models.utils import LR_SCHEDULERS, OPTIMIZERS, TripletLossBackboneModel


class TripletLossBasedModel(LightningModule):
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        embedding_size: int = 128,
        distance_metric: str = "L2",
        triplet_strategy: str = "VANILLA",
        miner_kwargs: Optional[Dict[str, Any]] = None,
        loss_func_kwargs: Optional[Dict[str, Any]] = None,
        learning_rate: float = 1e-3,
        optimizer: Type[Optimizer] = Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[Type[_LRScheduler]] = None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        loss_func_kwargs = dict() if loss_func_kwargs is None else loss_func_kwargs
        miner_kwargs = dict() if miner_kwargs is None else miner_kwargs

        self.model = TripletLossBackboneModel(model_name=model_name, embedding_size=embedding_size)

        self.learning_rate = learning_rate
        self.optimizer_cls = optimizer
        self.lr_scheduler_cls = lr_scheduler
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else dict(lr=self.learning_rate)
        self.lr_scheduler_kwargs = lr_scheduler_kwargs if lr_scheduler_kwargs is not None else dict()

        loss_configuration = LOSS_CONFIGURATION[triplet_strategy]
        distance_func = DISTANCES[distance_metric]

        self.miner = loss_configuration.miner(**miner_kwargs)
        self.loss_fn = loss_configuration.loss_func(distance=distance_func, **loss_func_kwargs)

        self.save_hyperparameters(
            "learning_rate",
            "model_name",
            "embedding_size",
            "distance_metric",
            "triplet_strategy",
            "loss_func_kwargs",
            "optimizer",
            "lr_scheduler",
            ignore=["model"],
        )

    def step(self, batch: List[Tensor], batch_idx: int, stage: str) -> Tensor:
        inputs, labels = batch
        embeddings = self.model(inputs.float())
        loss = self.loss_fn(embeddings, labels, self.miner(embeddings, labels))

        self.log(f"{stage}_loss", loss, logger=True, on_step=True, on_epoch=True, reduce_fx="mean")
        self.log("batch_size", labels.shape[0], logger=True, on_step=True, on_epoch=True, reduce_fx="mean")

        return loss

    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        return self.step(batch, batch_idx, stage="train")

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        return self.step(batch, batch_idx, stage="val")

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, _LRScheduler]]:
        optimizer_dict: Dict[str, Union[Optimizer, _LRScheduler]] = dict()

        optimizer_dict["optimizer"] = self.optimizer_cls(params=self.model.parameters(), **self.optimizer_kwargs)  # type: ignore

        if self.lr_scheduler_cls is not None:
            optimizer_dict["lr_scheduler"] = self.lr_scheduler_cls(**self.lr_scheduler_kwargs)
        return optimizer_dict
