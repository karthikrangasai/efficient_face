import torch
from pytorch_metric_learning.losses.base_metric_loss_function import BaseMetricLossFunction
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class ImprovedTripletMarginLoss(BaseMetricLossFunction):
    """
    Args:
        margin1: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        margin2: The desired difference filter-er for the anchor-positive distance.
        beta: Float, multiplier for intra-class constraint.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
    """

    def __init__(
        self, margin1=-1.0, margin2=0.01, beta=0.002, swap=False, smooth_loss=False, triplets_per_anchor="all", **kwargs
    ):
        super().__init__(**kwargs)
        self.margin1 = margin1
        self.margin2 = margin2
        self.beta = beta
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, ref_labels, t_per_anchor=self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        mat = self.distance(embeddings, ref_emb)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        if self.swap:
            pn_dists = mat[positive_idx, negative_idx]
            an_dists = self.distance.smallest_dist(an_dists, pn_dists)

        current_margins = self.distance.margin(ap_dists, an_dists)

        loss = torch.maximum(current_margins, self.margin1) + torch.multiply(
            torch.maximum(an_dists, self.margin2), self.beta
        )

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }

    def get_default_reducer(self):
        return AvgNonZeroReducer()
