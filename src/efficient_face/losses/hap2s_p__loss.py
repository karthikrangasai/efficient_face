from typing import Any, Dict

import torch
import torch.nn.functional as F
from pytorch_metric_learning.distances import BaseDistance, LpDistance
from pytorch_metric_learning.losses.generic_pair_loss import GenericPairLoss
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class HAP2S_P_Loss(GenericPairLoss):
    """
    Based on https://arxiv.org/pdf/1807.11206.pdf
    Args:
        alpha: The exponential weight for positive pairs
        beta: The exponential weight for negative pairs
        base: The shift in the exponent applied to both positive and negative pairs
    """

    def __init__(self, margin: float = 0.2, alpha: float = 2, smooth_loss: bool = False, **kwargs: Any) -> None:
        super().__init__(mat_based_loss=True, **kwargs)
        self.margin = margin
        self.alpha = alpha
        self.smooth_loss = smooth_loss
        self.add_to_recordable_attributes(list_of_names=["margin", "alpha"], is_stat=False)

    def _compute_loss(
        self, mat: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor
    ) -> Dict[str, Dict[str, Any]]:
        positive_weights = torch.pow(mat + 1, self.alpha) * pos_mask
        weighted_pdist = positive_weights * mat
        normed_weighted_pdist = torch.sum(weighted_pdist, 1) / (torch.sum(positive_weights, 1) + 1e-6)

        negative_weights = torch.pow(mat + 1, (-2.0 * self.alpha)) * neg_mask
        weighted_ndist = negative_weights * mat
        normed_weighted_ndist = torch.sum(weighted_ndist, 1) / (torch.sum(negative_weights, 1) + 1e-6)

        current_margins = self.distance.margin(normed_weighted_pdist, normed_weighted_ndist)
        if self.smooth_loss:
            loss = F.softplus(current_margins)
        else:
            loss = F.relu(current_margins + self.margin)

        return {
            "loss": {
                "losses": loss,
                "indices": c_f.torch_arange_from_size(mat),
                "reduction_type": "element",
            }
        }

    def get_default_distance(self) -> BaseDistance:
        return LpDistance()
