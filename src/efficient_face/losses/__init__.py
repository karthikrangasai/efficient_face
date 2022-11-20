from dataclasses import dataclass
from typing import Dict, Type

from pytorch_metric_learning.distances import BaseDistance, CosineSimilarity, LpDistance
from pytorch_metric_learning.losses import MultiSimilarityLoss, TripletMarginLoss
from pytorch_metric_learning.losses.base_metric_loss_function import BaseMetricLossFunction
from pytorch_metric_learning.miners import BaseTupleMiner, BatchHardMiner, MultiSimilarityMiner, TripletMarginMiner

from efficient_face.losses.hap2s_e__loss import HAP2S_E_Loss
from efficient_face.losses.hap2s_p__loss import HAP2S_P_Loss
from efficient_face.losses.triplet_batch_hard_v2_loss import ImprovedTripletMarginLoss
from efficient_face.losses.triplet_focal_loss import TripletFocalMarginLoss

__all__ = [
    "MultiSimilarityLoss",
    "TripletFocalMarginLoss",
    "HAP2S_E_Loss",
    "HAP2S_P_Loss",
    "ImprovedTripletMarginLoss",
    "TripletMarginLoss",
]


DISTANCES: Dict[str, BaseDistance] = {
    "L2": LpDistance(p=2),
    "squared-L2": LpDistance(p=2, power=2),
    "angular": CosineSimilarity(),
}


@dataclass
class LossConfiguration:
    loss_func: Type[BaseMetricLossFunction]
    miner: Type[BaseTupleMiner]


LOSS_CONFIGURATION = {
    "VANILLA": LossConfiguration(loss_func=TripletMarginLoss, miner=TripletMarginMiner),
    "BATCH_HARD": LossConfiguration(loss_func=TripletMarginLoss, miner=BatchHardMiner),
    "BATCH_HARD_V2": LossConfiguration(loss_func=ImprovedTripletMarginLoss, miner=BatchHardMiner),
    "FOCAL": LossConfiguration(loss_func=TripletFocalMarginLoss, miner=BatchHardMiner),
    "MULTISIMILARITY": LossConfiguration(loss_func=MultiSimilarityLoss, miner=MultiSimilarityMiner),
    "HAP2S_E": LossConfiguration(loss_func=HAP2S_E_Loss, miner=BatchHardMiner),
    "HAP2S_P": LossConfiguration(loss_func=HAP2S_P_Loss, miner=BatchHardMiner),
    # "ADAPTIVE": {"miner": BatchHardMiner},
    # "ASSORTED": {"miner": BatchHardMiner},
    # "CONSTELLATION": {"miner": BatchHardMiner},
}
