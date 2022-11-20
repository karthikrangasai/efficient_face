from typing import Any, Dict, Optional, Union

import torch
from flash.core.data.io.input import DataKeys
from flash.core.model import OutputKeys, Task
from flash.core.utilities.types import LR_SCHEDULER_TYPE, OPTIMIZER_TYPE
from torch.nn import ModuleDict
from torchmetrics import Metric

from efficient_face.losses import DISTANCES, LOSS_CONFIGURATION
from efficient_face.models.utils import TripletLossBackboneModel


class TripletLossBasedTask(Task):
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        embedding_size: int = 128,
        distance_metric: str = "L2",
        triplet_strategy: str = "VANILLA",
        learning_rate: Optional[float] = 1e-3,
        miner_kwargs: Optional[Dict[str, Any]] = None,
        loss_func_kwargs: Optional[Dict[str, Any]] = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        **kwargs: Any,
    ) -> None:
        if loss_func_kwargs is None:
            loss_func_kwargs = {}
        if miner_kwargs is None:
            miner_kwargs = {}

        super().__init__(
            model=TripletLossBackboneModel(model_name=model_name, embedding_size=embedding_size),
            loss_fn=None,
            learning_rate=learning_rate,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            **kwargs,
        )

        loss_configuration = LOSS_CONFIGURATION[triplet_strategy]
        distance_func = DISTANCES[distance_metric]

        self.miner = loss_configuration.miner(**miner_kwargs)
        self.loss_fn = loss_configuration.loss_func(distance=distance_func, **loss_func_kwargs)

        self.train_metrics = ModuleDict({})
        self.test_metrics = ModuleDict({})
        self.save_hyperparameters(
            "learning_rate",
            "optimizer",
            "model_name",
            "embedding_size",
            "distance_metric",
            "triplet_strategy",
            "loss_func_kwargs",
            ignore=["model", "backbone", "head", "adapter"],
        )

    def step(self, batch: Dict[DataKeys, Any], batch_idx: int, metrics: ModuleDict) -> Dict[OutputKeys, Any]:
        inputs = batch[DataKeys.INPUT]
        labels = batch[DataKeys.TARGET]
        embeddings = self.model(inputs)
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_fn(embeddings, labels, hard_pairs)

        logs = {}

        output = {OutputKeys.OUTPUT: embeddings}
        metric_embeddings = self.to_metrics_format(output[OutputKeys.OUTPUT])

        for name, metric in metrics.items():
            if isinstance(metric, Metric):
                metric(metric_embeddings, labels)
                logs[name] = metric
            else:
                logs[name] = metric(metric_embeddings, labels)

        output[OutputKeys.LOSS] = loss
        output[OutputKeys.LOGS] = self.compute_logs(logs, loss)
        output[OutputKeys.TARGET] = labels
        output[OutputKeys.BATCH_SIZE] = labels.shape[0] if isinstance(labels, torch.Tensor) else None
        return output

    def compute_logs(self, logs: Dict[str, Any], loss: torch.Tensor) -> Dict[str, Any]:
        logs.update({"loss": loss})
        return logs
