from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import timm
import torch
from flash.core.data.io.input import DataKeys
from flash.core.model import OutputKeys, Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.types import LR_SCHEDULER_TYPE, OPTIMIZER_TYPE
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from pytorch_metric_learning.reducers import DoNothingReducer
from torch.nn import Dropout, Linear, Module, ModuleDict, Sequential
from torch.nn.functional import normalize
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric

from efficient_face.model_conf import DISTANCES, LOSS_CONFIGURATION
from efficient_face.sam_optimizers import ESAM, SAM


class MetricEmbedding(Module):
    def __init__(self, in_features: int, embedding_size: int, use_dropout: bool = False) -> None:
        """L2 Normalized `Linear` layer usually used as output layer.
        Args:
            embedding_size: Dimension of the output embbeding. Commonly between
            32 and 512. Larger embeddings usually result in higher accuracy up
            to a point at the expense of making search slower.
        """
        super().__init__()
        self.embedding_size = embedding_size
        if use_dropout:
            self.dense = Sequential(
                Dropout(p=0.3),
                Linear(in_features=in_features, out_features=self.embedding_size, dtype=torch.float32),
            )
        else:
            self.dense = Linear(in_features=in_features, out_features=self.embedding_size, dtype=torch.float32)

    def forward(self, x):
        x = self.dense(x)
        return normalize(x, p=2.0, dim=1)


class BackboneModel(torch.nn.Module):
    def __init__(self, model_name: str = "efficientnet_b0", embedding_size=128):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.backbone.classifier = MetricEmbedding(
            in_features=self.backbone.num_features,
            embedding_size=embedding_size,
            use_dropout="efficientnet" in model_name,
        )

    def forward(self, x):
        x = self.backbone(x)
        return x


class SoftmaxBackboneModel(torch.nn.Module):
    def __init__(self, model_name: str = "efficientnet_b0", embedding_size=128, num_classes: int = 10):
        if num_classes not in [10, 100]:
            raise ValueError("CIFAR has either 10 or 100 classes.")
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.backbone.classifier = MetricEmbedding(
            in_features=self.backbone.num_features,
            embedding_size=embedding_size,
            use_dropout="efficientnet" in model_name,
        )
        self.linear = Linear(in_features=embedding_size, out_features=num_classes, dtype=torch.float32)

    def forward(self, x):
        x = self.backbone(x)
        x = self.linear(x)
        return x
