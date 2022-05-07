from typing import Any, Dict, List, Optional, Tuple, Union

import timm
import torch
from flash.core.data.io.input import DataKeys
from flash.core.model import OutputKeys, Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.types import LR_SCHEDULER_TYPE, OPTIMIZER_TYPE
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.nn import Dropout, Linear, Module, ModuleDict, Sequential
from torch.nn.functional import normalize
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric

from efficient_face.model_conf import DISTANCES, LOSS_CONFIGURATION
from efficient_face.sam_optimizer import SAM


class MetricEmbedding(Module):
    def __init__(self, in_features, unit, use_dropout=False):
        """L2 Normalized `Dense` layer usually used as output layer.
        Args:
            unit: Dimension of the output embbeding. Commonly between
            32 and 512. Larger embeddings usually result in higher accuracy up
            to a point at the expense of making search slower.
        """
        super().__init__()
        self.unit = unit
        if use_dropout:
            self.dense = Sequential(
                Dropout(p=0.3),
                Linear(in_features=in_features, out_features=unit, dtype=torch.float32),
            )
        else:
            self.dense = Linear(in_features=in_features, out_features=unit, dtype=torch.float32)

    def forward(self, x):
        x = self.dense(x)
        x = normalize(x, p=2.0, dim=1)
        return x


class BackboneModel(torch.nn.Module):
    def __init__(self, model_name="efficientnet_b0", embedding_size=128):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.backbone.classifier = MetricEmbedding(
            in_features=self.backbone.num_features,
            unit=embedding_size,
            use_dropout="efficientnet" in model_name,
        )

    def forward(self, x):
        x = self.backbone(x)
        return x


class FaceRecognitionModel(Task):
    def __init__(
        self,
        model_name: Union[str, torch.nn.Module] = "efficientnet_b0",
        embedding_size: int = 128,
        distance_metric: str = "L2",
        loss_conf: str = "VANILLA",
        learning_rate: Optional[float] = 1e-3,
        miner_kwargs: Optional[Dict[str, Any]] = None,
        loss_func_kwargs: Optional[Dict[str, Any]] = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        **kwargs,
    ) -> None:
        if loss_func_kwargs is None:
            loss_func_kwargs = {}
        if miner_kwargs is None:
            miner_kwargs = {}

        super().__init__(
            model=BackboneModel(model_name=model_name, embedding_size=embedding_size),
            loss_fn=LOSS_CONFIGURATION[loss_conf]["loss_func"](distance=DISTANCES[distance_metric], **loss_func_kwargs),
            learning_rate=learning_rate,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            **kwargs,
        )

        self.miner = LOSS_CONFIGURATION[loss_conf]["miner"](**miner_kwargs)
        self.loss_fn = LOSS_CONFIGURATION[loss_conf]["loss_func"](
            distance=DISTANCES[distance_metric], **loss_func_kwargs
        )
        self.train_metrics = ModuleDict({})
        self.test_metrics = ModuleDict({})

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


class SAMFaceRecognitionModel(Task):
    def __init__(
        self,
        model_name: Union[str, torch.nn.Module] = "efficientnet_b0",
        embedding_size: int = 128,
        distance_metric: str = "L2",
        loss_conf: str = "VANILLA",
        learning_rate: Optional[float] = 1e-3,
        miner_kwargs: Optional[Dict[str, Any]] = None,
        loss_func_kwargs: Optional[Dict[str, Any]] = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        **kwargs,
    ) -> None:
        if loss_func_kwargs is None:
            loss_func_kwargs = {}
        if miner_kwargs is None:
            miner_kwargs = {}

        super().__init__(
            model=BackboneModel(model_name=model_name, embedding_size=embedding_size),
            loss_fn=LOSS_CONFIGURATION[loss_conf]["loss_func"](distance=DISTANCES[distance_metric], **loss_func_kwargs),
            learning_rate=learning_rate,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            **kwargs,
        )

        self.miner = LOSS_CONFIGURATION[loss_conf]["miner"](**miner_kwargs)
        self.loss_fn = LOSS_CONFIGURATION[loss_conf]["loss_func"](
            distance=DISTANCES[distance_metric], **loss_func_kwargs
        )
        self.train_metrics = ModuleDict({})
        self.test_metrics = ModuleDict({})

        # Manual Optimization
        self.automatic_optimization = False

    def forward_step(self, batch: Any, batch_idx: int) -> Dict[OutputKeys, Any]:
        inputs = batch[DataKeys.INPUT]
        labels = batch[DataKeys.TARGET]
        embeddings = self.model(inputs)
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_fn(embeddings, labels, hard_pairs)
        output = {
            OutputKeys.OUTPUT: embeddings,
            OutputKeys.LOSS: loss,
            OutputKeys.BATCH_SIZE: labels.shape[0] if isinstance(labels, torch.Tensor) else None,
            OutputKeys.TARGET: labels,
            OutputKeys.LOGS: {"loss": loss},
        }
        return output

    def metrics_setp(
        self, batch: Dict[DataKeys, Any], batch_idx: int, output: Dict[OutputKeys, Any], metrics: ModuleDict
    ) -> Dict[OutputKeys, Any]:
        labels = batch[DataKeys.TARGET]
        metric_embeddings = self.to_metrics_format(output[OutputKeys.OUTPUT])
        for name, metric in metrics.items():
            if isinstance(metric, Metric):
                metric(metric_embeddings, labels)
                output[OutputKeys.LOGS][name] = metric
            else:
                output[OutputKeys.LOGS][name] = metric(metric_embeddings, labels)
        return output

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        sam_optimizer = self.get_optimizers()
        sam_optimizer.zero_grad()

        # Compute and add e_w
        output1 = self.forward_step(batch, batch_idx)
        self.manual_backward(output1[OutputKeys.LOSS])
        sam_optimizer.compute_and_add_e_w()
        sam_optimizer.zero_grad()

        # Subtract e_w and step actual optimizer
        output2 = self.forward_step(batch, batch_idx)
        self.manual_backward(output2[OutputKeys.LOSS])
        sam_optimizer.subtract_e_w_and_step_optimizer()
        sam_optimizer.zero_grad()

        # Log the loss
        log_kwargs = {"batch_size": output1.get(OutputKeys.BATCH_SIZE, None)}
        self.log_dict(
            {f"train_{k}": v for k, v in output1[OutputKeys.LOGS].items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            **log_kwargs,
        )
        return output1[OutputKeys.LOSS]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        output = self.forward_step(batch, batch_idx)
        output = self.metrics_setp(batch, batch_idx, output, self.val_metrics)
        log_kwargs = {"batch_size": output.get(OutputKeys.BATCH_SIZE, None)}
        self.log_dict(
            {f"val_{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            **log_kwargs,
        )

    def get_optimizers(
        self, use_pl_optimizer: bool = True
    ) -> Union[Optimizer, LightningOptimizer, List[Optimizer], List[LightningOptimizer]]:
        if use_pl_optimizer:
            opts = list(self.trainer.strategy._lightning_optimizers.values())
        else:
            opts = self.trainer.optimizers

        # single optimizer
        if isinstance(opts, list) and len(opts) == 1 and isinstance(opts[0], (Optimizer, LightningOptimizer)):
            return opts[0]
        # multiple opts
        return opts

    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:
        optimizer = super().configure_optimizers()

        if isinstance(optimizer, Tuple):
            return SAM(optimizer=optimizer[0]), optimizer[1]

        return SAM(optimizer=optimizer)
