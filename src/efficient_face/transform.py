from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Union

import torch
from flash.core.data.io.input_transform import InputTransform
from torchvision import transforms as T


@dataclass
class FaceRecognitionInputTransform(InputTransform):

    crop_size: Union[int, Tuple[int, int]] = 224
    model_name: str = "efficientnet_b0"

    def __post_init__(self):
        return super().__post_init__()

    def get_model_preprocessor(self, model_name: str) -> Callable:
        if model_name == "efficientnet_b0":
            return T.Resize(size=[224, 224])

        return T.Normalize((0.0, 0.0, 0.0), (0.225, 0.225, 0.225))

    def input_per_sample_transform(self) -> Callable:
        return T.Compose(
            [
                T.ToTensor(),
                self.get_model_preprocessor(self.model_name),
                T.RandomCrop(size=self.crop_size),
                T.RandomHorizontalFlip(p=0.5),
            ]
        )

    def target_per_sample_transform(self) -> Callable:
        return torch.as_tensor
