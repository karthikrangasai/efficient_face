from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.miners import BatchHardMiner, MultiSimilarityMiner, TripletMarginMiner

from efficient_face.losses import (
    HAP2S_E_Loss,
    HAP2S_P_Loss,
    ImprovedTripletMarginLoss,
    MultiSimilarityLoss,
    TripletFocalMarginLoss,
    TripletMarginLoss,
)

DISTANCES = {
    "L2": LpDistance(p=2),
    "squared-L2": LpDistance(p=2, power=2),
    "angular": CosineSimilarity(),
}


LOSS_CONFIGURATION = {
    "VANILLA": {"loss_func": TripletMarginLoss, "miner": TripletMarginMiner},
    "BATCH_HARD": {"loss_func": TripletMarginLoss, "miner": BatchHardMiner},
    "BATCH_HARD_V2": {"loss_func": ImprovedTripletMarginLoss, "miner": BatchHardMiner},
    "FOCAL": {"loss_func": TripletFocalMarginLoss, "miner": BatchHardMiner},
    "ADAPTIVE": {"miner": BatchHardMiner},
    "ASSORTED": {"miner": BatchHardMiner},
    "CONSTELLATION": {"miner": BatchHardMiner},
    "MULTISIMILARITY": {"loss_func": MultiSimilarityLoss, "miner": MultiSimilarityMiner},
    "HAP2S_E": {"loss_func": HAP2S_E_Loss, "miner": BatchHardMiner},
    "HAP2S_P": {"loss_func": HAP2S_P_Loss, "miner": BatchHardMiner},
}
