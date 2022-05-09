from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Union

import torch
from flash.core.data.io.input_transform import InputTransform
from torchvision import transforms as T


@dataclass
class FaceRecognitionInputTransform(InputTransform):

    image_size: Union[int, Tuple[int, int]] = 226
    crop_size: Union[int, Tuple[int, int]] = 224
    model_name: str = "efficientnet_b0"

    def __post_init__(self):
        return super().__post_init__()

    def get_model_preprocessor(self, model_name: str) -> Callable:
        return T.Lambda(lambda x: x)

    def input_per_sample_transform(self) -> Callable:
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(size=self.image_size),
                self.get_model_preprocessor(self.model_name),
                T.RandomCrop(size=self.crop_size),
                T.RandomHorizontalFlip(p=0.5),
            ]
        )

    def target_per_sample_transform(self) -> Callable:
        return torch.as_tensor
