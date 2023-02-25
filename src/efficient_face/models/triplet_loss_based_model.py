from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch_optimizer import SGDW, Lookahead

from efficient_face.losses import DISTANCES, LOSS_CONFIGURATION
from efficient_face.metrics.metrics import compute_metrics_for_triplets
from efficient_face.models.utils import TripletLossBackboneModel


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
        loss_func_kwargs = dict(margin=0.2) if loss_func_kwargs is None else loss_func_kwargs
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
        self.margin: float = loss_func_kwargs["margin"]

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
        anc_pos_neg = self.miner(embeddings, labels)
        loss = self.loss_fn(embeddings, labels, anc_pos_neg)

        anchors, positives, negatives = anc_pos_neg
        accuracy, precision, recall, f1_score = compute_metrics_for_triplets(
            embeddings[anchors], embeddings[positives], embeddings[negatives], self.margin, use_cosine_similarity=False
        )

        self.log(f"{stage}_loss", loss, logger=True, on_step=True, on_epoch=True, reduce_fx="mean")
        self.log(f"{stage}_accuracy", accuracy, logger=True, on_step=True, on_epoch=True, reduce_fx="mean")
        self.log(f"{stage}_precision", precision, logger=True, on_step=True, on_epoch=True, reduce_fx="mean")
        self.log(f"{stage}_recall", recall, logger=True, on_step=True, on_epoch=True, reduce_fx="mean")
        self.log(f"{stage}_f1_score", f1_score, logger=True, on_step=True, on_epoch=True, reduce_fx="mean")
        self.log("batch_size", labels.shape[0], logger=True, on_step=True, on_epoch=True, reduce_fx="mean")

        return loss

    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        return self.step(batch, batch_idx, stage="train")

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        return self.step(batch, batch_idx, stage="val")

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, _LRScheduler]]:
        optimizer_dict: Dict[str, Union[Optimizer, _LRScheduler]] = dict()

        if self.optimizer_cls.__name__.lower() == "lookahead":
            sgdw_optimizer = SGDW(
                params=self.model.parameters(),
                lr=self.optimizer_kwargs.pop("lr"),
                momentum=0.8,
                weight_decay=1e-5,
                nesterov=True,
            )
            optimizer = Lookahead(sgdw_optimizer, **self.optimizer_kwargs)
        else:
            optimizer = self.optimizer_cls(params=self.model.parameters(), **self.optimizer_kwargs)  # type: ignore
        optimizer_dict["optimizer"] = optimizer

        if self.lr_scheduler_cls is not None:
            arg_name = self.lr_scheduler_kwargs.pop("num_steps_arg", None)
            num_steps_factor = self.lr_scheduler_kwargs.pop("num_steps_factor", 1.0)
            if arg_name is not None:
                self.lr_scheduler_kwargs[arg_name] = int(self.trainer.estimated_stepping_batches / num_steps_factor)

            optimizer_dict["lr_scheduler"] = self.lr_scheduler_cls(optimizer=optimizer, **self.lr_scheduler_kwargs)
        return optimizer_dict
