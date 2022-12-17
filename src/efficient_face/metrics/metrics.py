from typing import Tuple

from torch import Tensor
from torch.nn.functional import cosine_similarity, pairwise_distance
from torchmetrics.functional.classification.accuracy import accuracy
from torchmetrics.functional.classification.f_beta import binary_f1_score
from torchmetrics.functional.classification.precision_recall import precision, recall


def compute_metrics_for_triplets(
    anchors: Tensor, positives: Tensor, negatives: Tensor, margin: float, use_cosine_similarity: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    distance_function = cosine_similarity if use_cosine_similarity else pairwise_distance
    anchor_positive_distances = distance_function(anchors, positives)  # type: ignore
    anchor_negative_distances = distance_function(anchors, negatives)  # type: ignore

    true_positives = (anchor_positive_distances < margin).sum()
    false_positives = (anchor_negative_distances < margin).sum()

    true_negatives = (anchor_negative_distances > margin).sum()
    false_negatives = (anchor_positive_distances > margin).sum()

    accuracy = (true_positives + true_negatives) / (true_negatives + true_negatives + false_negatives + false_positives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_negatives / (true_positives + false_negatives)
    f1_score = (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score


def compute_metrics_for_softmax(preds: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    _accuracy = accuracy(preds, labels)
    _precision = precision(preds, labels)
    _recall = recall(preds, labels)
    f1_score = (_precision * _recall) / (_precision + _recall)
    return _accuracy, _precision, _recall, f1_score
