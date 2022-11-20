from typing import Any, Dict

import torch
from pytorch_metric_learning.losses.base_metric_loss_function import BaseMetricLossFunction
from pytorch_metric_learning.reducers import AvgNonZeroReducer, BaseReducer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class TripletFocalMarginLoss(BaseMetricLossFunction):
    """
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
    """

    def __init__(
        self,
        margin: float = 0.05,
        swap: bool = False,
        smooth_loss: bool = False,
        triplets_per_anchor: str = "all",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.margin = margin
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels) -> Dict[str, Dict[str, Any]]:  # type: ignore
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, ref_labels, t_per_anchor=self.triplets_per_anchor  # type: ignore
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        mat = self.distance(embeddings, ref_emb)
        ap_dists = torch.exp(mat[anchor_idx, positive_idx])
        an_dists = torch.exp(mat[anchor_idx, negative_idx])
        if self.swap:
            pn_dists = mat[positive_idx, negative_idx]
            an_dists = torch.exp(self.distance.smallest_dist(an_dists, pn_dists))

        current_margins = self.distance.margin(ap_dists, an_dists)
        violation = current_margins + self.margin
        if self.smooth_loss:
            loss = torch.nn.functional.softplus(violation)
        else:
            loss = torch.nn.functional.relu(violation)

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }

    def get_default_reducer(self) -> BaseReducer:
        return AvgNonZeroReducer()
