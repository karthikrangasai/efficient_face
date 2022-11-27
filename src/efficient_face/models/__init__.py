from efficient_face.models.softmax_based_model import SoftmaxBasedModel
from efficient_face.models.triplet_loss_based_model import TripletLossBasedModel

__all__ = ["TripletLossBasedModel", "SoftmaxBasedModel"]
MODEL_TYPE = {"softmax": SoftmaxBasedModel, "triplet_loss": TripletLossBasedModel}
