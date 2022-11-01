from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from flash.core.data.io.input import DataKeys
from flash.core.model import OutputKeys, Task
from flash.core.utilities.types import LR_SCHEDULER_TYPE, OPTIMIZER_TYPE
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from pytorch_metric_learning.reducers import DoNothingReducer
from torch.nn import ModuleDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric

from efficient_face.model_conf import DISTANCES, LOSS_CONFIGURATION
from efficient_face.models.utils import BackboneModel
from efficient_face.sam_optimizers import ESAM


class ESAMEfficientFaceModel(Task):
    def __init__(
        self,
        model_name: Union[str, torch.nn.Module] = "efficientnet_b0",
        embedding_size: int = 128,
        distance_metric: str = "L2",
        triplet_strategy: str = "VANILLA",
        learning_rate: Optional[float] = 1e-3,
        miner_kwargs: Optional[Dict[str, Any]] = None,
        loss_func_kwargs: Optional[Dict[str, Any]] = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        sam_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if loss_func_kwargs is None:
            loss_func_kwargs = {}
        loss_func_kwargs["reducer"] = DoNothingReducer()
        if miner_kwargs is None:
            miner_kwargs = {}

        super().__init__(
            model=BackboneModel(model_name=model_name, embedding_size=embedding_size),
            loss_fn=LOSS_CONFIGURATION[triplet_strategy]["loss_func"](
                distance=DISTANCES[distance_metric], **loss_func_kwargs
            ),
            learning_rate=learning_rate,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            **kwargs,
        )

        self.miner = LOSS_CONFIGURATION[triplet_strategy]["miner"](**miner_kwargs)
        self.loss_fn = LOSS_CONFIGURATION[triplet_strategy]["loss_func"](
            distance=DISTANCES[distance_metric], **loss_func_kwargs
        )
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
        self.sam_kwargs = sam_kwargs
        if self.sam_kwargs is None:
            self.sam_kwargs = {}
        # Manual Optimization
        self.automatic_optimization = False

    def forward_step(
        self, batch: Any, batch_idx: int, hard_pairs: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[OutputKeys, Any]:
        inputs = batch[DataKeys.INPUT]
        labels = batch[DataKeys.TARGET]
        embeddings = self.model(inputs)
        if hard_pairs is None:
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
        self,
        batch: Dict[DataKeys, Any],
        batch_idx: int,
        output: Dict[OutputKeys, Any],
        metrics: ModuleDict,
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

    def get_opt_loss_and_reduced_loss(self, loss_dict: Dict[str, Any]):
        # {'loss': {'losses': tensor([0., 0., 0.],...ackward0>), 'indices': (...), 'reduction_type': 'triplet'}}
        loss: torch.Tensor = loss_dict["loss"]["losses"]
        opt_step_loss = loss.detach().clone()
        opt_step_indices: torch.Tensor = loss_dict["loss"]["indices"]

        reduced_loss = loss.mean()
        return reduced_loss, opt_step_loss, opt_step_indices

    def get_optimizer_closure(
        self,
        opt_step_loss: torch.Tensor,
        opt_step_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch: Any,
        batch_idx: int,
    ):
        def closure_wrapper() -> Tuple[
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            int,
            Callable[[Optional[List[int]], bool], Tuple[torch.Tensor, torch.Tensor]],
        ]:
            def closure(
                indices: Optional[List[int]] = None,
                backward: bool = False,
                hard_pair_indices: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
            ):
                if indices is not None and len(indices) > 0:
                    _batch = {key: value[indices] for key, value in batch.items()}
                    closure_output = self.forward_step(_batch, batch_idx, hard_pair_indices)
                else:
                    closure_output = self.forward_step(batch, batch_idx, hard_pair_indices)

                reduced_loss, opt_step_loss, opt_step_indices = self.get_opt_loss_and_reduced_loss(
                    closure_output[OutputKeys.LOSS]
                )
                closure_output[OutputKeys.LOSS] = reduced_loss

                if backward:
                    self.manual_backward(closure_output[OutputKeys.LOSS])
                return opt_step_loss

            return opt_step_loss, opt_step_indices, len(opt_step_loss), closure

        return closure_wrapper

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        esam_optimizer: ESAM = self.get_optimizers()
        esam_optimizer.zero_grad()

        # Fist forward pass
        output = self.forward_step(batch, batch_idx)
        reduced_loss, opt_step_loss, opt_step_indices = self.get_opt_loss_and_reduced_loss(output[OutputKeys.LOSS])
        self.manual_backward(reduced_loss)

        esam_optimizer.step(
            self.get_optimizer_closure(
                opt_step_loss=opt_step_loss,
                opt_step_indices=opt_step_indices,
                batch=batch,
                batch_idx=batch_idx,
            )
        )

        scheduler = self.get_lr_schedulers()
        if scheduler is not None:
            scheduler.step()

        # Log the loss
        output[OutputKeys.LOSS] = reduced_loss
        output[OutputKeys.LOGS]["loss"] = reduced_loss
        log_kwargs = {"batch_size": output.get(OutputKeys.BATCH_SIZE, None)}
        self.log_dict(
            {f"train_{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            **log_kwargs,
        )
        return output[OutputKeys.LOSS]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        output = self.forward_step(batch, batch_idx)
        reduced_loss, _, _ = self.get_opt_loss_and_reduced_loss(output[OutputKeys.LOSS])
        output = self.metrics_setp(batch, batch_idx, output, self.val_metrics)

        output[OutputKeys.LOSS] = reduced_loss
        output[OutputKeys.LOGS]["loss"] = reduced_loss
        log_kwargs = {"batch_size": output.get(OutputKeys.BATCH_SIZE, None)}
        self.log_dict(
            {f"val_{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            **log_kwargs,
        )

    # Taken from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/core/lightning.py
    # Because in Flash `self.optimizers` points to Optimizers Registry and not the PL LightningModule method.
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

    # Taken from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/core/lightning.py
    # Because in Flash `self.lr_schedulers` points to LR_Schedulers Registry and not the PL LightningModule method.
    def get_lr_schedulers(self) -> Optional[Union[LRSchedulerTypeUnion, List[LRSchedulerTypeUnion]]]:
        if not self.trainer.lr_scheduler_configs:
            return None

        # ignore other keys "interval", "frequency", etc.
        lr_schedulers = [config.scheduler for config in self.trainer.lr_scheduler_configs]

        # single scheduler
        if len(lr_schedulers) == 1:
            return lr_schedulers[0]

        # multiple schedulers
        return lr_schedulers

    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:
        base_optimizer = super().configure_optimizers()

        if isinstance(base_optimizer, Tuple):
            optimizers, _lr_schedulers = base_optimizer
            sam_optimizers = [ESAM(optimizer=_optimizer, **self.sam_kwargs) for _optimizer in optimizers]
            lr_schedulers = [self._instantiate_lr_scheduler(optimizer)["scheduler"] for optimizer in sam_optimizers]
            del _lr_schedulers
            return sam_optimizers, lr_schedulers

        return ESAM(optimizer=base_optimizer)
