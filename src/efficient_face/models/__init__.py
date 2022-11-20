from efficient_face.models.softmax_based_model import SoftmaxBasedTask
from efficient_face.models.triplet_loss_based_model import TripletLossBasedTask

__all__ = ["TripletLossBasedTask", "SoftmaxBasedTask"]
MODEL_TYPE = {"softmax": SoftmaxBasedTask, "triplet_loss": TripletLossBasedTask}
