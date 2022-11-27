from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.transforms import Compose


def get_image_transform(model_name: str) -> Compose:
    config = resolve_data_config({}, model=model_name)
    config["separate"] = True
    config["is_training"] = True
    config["hflip"] = 0.5
    config["vflip"] = 0.0
    transform = create_transform(**config)  # type: ignore
    primary_tfl, _, final_tfl = transform  # type: ignore
    return Compose([primary_tfl, final_tfl])
