import torch
import torch.nn.functional as F
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses.generic_pair_loss import GenericPairLoss
from pytorch_metric_learning.utils import common_functions as c_f


class HAP2S_E_Loss(GenericPairLoss):
    """
    Based on https://arxiv.org/pdf/1807.11206.pdf
    Args:
        alpha: The exponential weight for positive pairs
        beta: The exponential weight for negative pairs
        base: The shift in the exponent applied to both positive and negative pairs
    """

    def __init__(self, margin=0.2, sigma=0.5, smooth_loss=False, **kwargs):
        super().__init__(mat_based_loss=True, **kwargs)
        self.margin = margin
        self.sigma = sigma
        self.smooth_loss = smooth_loss
        self.add_to_recordable_attributes(list_of_names=["margin", "sigma"], is_stat=False)

    def _compute_loss(self, mat: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor):
        positive_weights = torch.exp(mat / self.sigma) * pos_mask
        weighted_pdist = positive_weights * mat
        normed_weighted_pdist = torch.sum(weighted_pdist, 1) / (torch.sum(positive_weights, 1) + 1e-6)

        negative_weights = torch.exp((-1.0 * mat) / self.sigma) * neg_mask
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

    def get_default_distance(self):
        return LpDistance()
