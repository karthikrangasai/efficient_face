from typing import Any, Dict, List, Optional, Type, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from efficient_face.losses import DISTANCES, LOSS_CONFIGURATION
from efficient_face.metrics.metrics import compute_metrics_for_softmax, compute_metrics_for_triplets
from efficient_face.models.utils import SoftmaxBackboneModel


class SoftmaxBasedModel(LightningModule):
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
        self.model = SoftmaxBackboneModel(model_name=model_name, embedding_size=embedding_size, num_classes=10)
        self.train_loss_fn = CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.optimizer_cls = optimizer
        self.lr_scheduler_cls = lr_scheduler
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else dict(lr=self.learning_rate)
        self.lr_scheduler_kwargs = lr_scheduler_kwargs if lr_scheduler_kwargs is not None else dict()

        loss_configuration = LOSS_CONFIGURATION[triplet_strategy]
        distance_func = DISTANCES[distance_metric]

        miner_kwargs = dict() if miner_kwargs is None else miner_kwargs
        val_loss_func_kwargs = dict() if loss_func_kwargs is None else loss_func_kwargs

        self.miner = loss_configuration.miner(**miner_kwargs)
        loss_func_kwargs = dict(margin=0.2) if loss_func_kwargs is None else loss_func_kwargs
        self.val_loss_fn = loss_configuration.loss_func(distance=distance_func, **val_loss_func_kwargs)
        self.margin: float = loss_func_kwargs["margin"]

        self.save_hyperparameters(
            "model_name",
            "embedding_size",
            "learning_rate",
            "optimizer",
            "lr_scheduler",
            ignore=["model", "loss_fn"],
        )

    def step(self, batch: List[Tensor], batch_idx: int, stage: str) -> Tensor:
        inputs, labels = batch
        if stage == "val":
            embeddings = self.model(inputs.float())
            anc_pos_neg = self.miner(embeddings, labels)
            loss = self.val_loss_fn(embeddings, labels, anc_pos_neg)
            anchors, positives, negatives = anc_pos_neg
            accuracy, precision, recall, f1_score = compute_metrics_for_triplets(
                embeddings[anchors],
                embeddings[positives],
                embeddings[negatives],
                self.margin,
                use_cosine_similarity=False,
            )
        else:
            preds = self.model(inputs.float())
            loss = self.train_loss_fn(preds, labels)
            accuracy, precision, recall, f1_score = compute_metrics_for_softmax(preds, labels)

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

        optimizer = self.optimizer_cls(params=self.model.parameters(), **self.optimizer_kwargs)  # type: ignore
        optimizer_dict["optimizer"] = optimizer

        if self.lr_scheduler_cls is not None:
            arg_name = self.lr_scheduler_kwargs.pop("num_steps_arg", None)
            num_steps_factor = self.lr_scheduler_kwargs.pop("num_steps_factor", 1.0)
            if arg_name is not None:
                self.lr_scheduler_kwargs[arg_name] = self.trainer.estimated_stepping_batches / num_steps_factor

            optimizer_dict["lr_scheduler"] = self.lr_scheduler_cls(optimizer=optimizer, **self.lr_scheduler_kwargs)
        return optimizer_dict
