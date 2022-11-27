from typing import Dict, Type

import timm
import torch
from torch.nn import Dropout, Identity, Linear, Module, Sequential
from torch.nn.functional import normalize
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

OPTIMIZERS: Dict[str, Type[Optimizer]] = dict(adam=Adam)
LR_SCHEDULERS: Dict[str, Type[_LRScheduler]] = dict()


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
        self.dense = Sequential(
            Dropout(p=0.3) if use_dropout else Identity(),
            Linear(in_features=in_features, out_features=self.embedding_size, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        return normalize(x, p=2.0, dim=1)


class TripletLossBackboneModel(torch.nn.Module):
    def __init__(self, model_name: str = "efficientnet_b0", embedding_size: int = 128) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.backbone.classifier = MetricEmbedding(
            in_features=self.backbone.num_features,
            embedding_size=embedding_size,
            use_dropout="efficientnet" in model_name,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x


class SoftmaxBackboneModel(torch.nn.Module):
    def __init__(self, model_name: str = "efficientnet_b0", embedding_size: int = 128, num_classes: int = 10) -> None:
        if num_classes not in [10, 100]:
            raise ValueError("ciFAIR has either 10 or 100 classes.")
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.backbone.classifier = MetricEmbedding(
            in_features=self.backbone.num_features,
            embedding_size=embedding_size,
            use_dropout="efficientnet" in model_name,
        )
        self.linear = Linear(in_features=embedding_size, out_features=num_classes, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if torch.is_grad_enabled():
            return self.linear(x)
        return x
