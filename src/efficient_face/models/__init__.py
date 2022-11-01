from efficient_face.models.efficient_face_model import EfficientFaceModel
from efficient_face.models.esam_efficient_face_model import ESAMEfficientFaceModel
from efficient_face.models.sam_efficient_face_model import SAMEfficientFaceModel

MODEL_TYPE = {"Normal": EfficientFaceModel, "SAM": SAMEfficientFaceModel, "ESAM": ESAMEfficientFaceModel}
