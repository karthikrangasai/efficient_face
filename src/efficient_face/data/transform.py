from typing import Callable, Tuple, Union

from PIL.Image import Image
from torch import Tensor
from torchvision import transforms as T


def get_model_preprocessor(model_name: str) -> Callable[[Tensor], Tensor]:
    if model_name.startswith("tf_efficientnetv2"):
        return T.Compose(
            [
                T.Normalize(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return T.Lambda(lambda x: x)


def get_image_transform(
    model_name: str = "efficientnet_b0",
    image_size: Union[int, Tuple[int, int]] = 226,
    crop_size: Union[int, Tuple[int, int]] = 224,
) -> Callable[[Image], Tensor]:
    return T.Compose(
        [
            T.Resize(size=image_size),
            get_model_preprocessor(model_name),
            T.RandomCrop(size=crop_size),
            T.RandomHorizontalFlip(p=0.5),
            T.PILToTensor(),
        ]
    )
