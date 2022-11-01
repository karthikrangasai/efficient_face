from typing import Any, Dict, Optional, Union

from flash.core.data.io.input import DataKeys
from flash.core.model import OutputKeys, Task
from flash.core.utilities.types import LR_SCHEDULER_TYPE, OPTIMIZER_TYPE
from torch import Tensor
from torch.nn import CrossEntropyLoss, ModuleDict
from torchmetrics import Metric
from torchmetrics.functional.classification.accuracy import accuracy
from torchmetrics.functional.classification.precision_recall import precision, recall

from efficient_face.model_conf import DISTANCES, LOSS_CONFIGURATION
from efficient_face.models.utils import BackboneModel, SoftmaxBackboneModel


class SoftmaxBasedTask(Task):
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        embedding_size: int = 128,
        num_classes: int = 10,
        learning_rate: Optional[float] = 1e-3,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        **kwargs,
    ) -> None:

        super().__init__(
            model=SoftmaxBackboneModel(model_name=model_name, embedding_size=embedding_size, num_classes=num_classes),
            loss_fn=None,
            learning_rate=learning_rate,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            **kwargs,
        )
        self.loss_fn = CrossEntropyLoss()

        self.save_hyperparameters(
            "learning_rate",
            "optimizer",
            "model_name",
            "embedding_size",
            "num_classes",
            "lr_scheduler",
            ignore=["model", "backbone", "head", "adapter"],
        )

    def step(self, batch: Dict[DataKeys, Any], batch_idx: int) -> Dict[OutputKeys, Any]:
        inputs = batch[DataKeys.INPUT]
        labels = batch[DataKeys.TARGET]
        y_pred = self.model(inputs)
        loss: Tensor = self.loss_fn(y_pred, labels)

        logs = {}

        output = {OutputKeys.OUTPUT: y_pred}

        logs["accuracy"] = accuracy(y_pred, labels)
        logs["preicsion"] = precision(y_pred, labels)
        logs["recall"] = recall(y_pred, labels)

        output[OutputKeys.LOSS] = loss
        output[OutputKeys.LOGS] = self.compute_logs(logs, loss)
        output[OutputKeys.TARGET] = labels
        output[OutputKeys.BATCH_SIZE] = labels.shape[0] if isinstance(labels, Tensor) else None
        return output

    def compute_logs(self, logs: Dict[str, Any], loss: Tensor) -> Dict[str, Any]:
        logs.update({"loss": loss})
        return logs
