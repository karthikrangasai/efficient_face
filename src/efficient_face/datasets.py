from pathlib import Path
from typing import Any, Dict, List, Optional

from flash import RunningStage
from flash.core.data.data_module import DataModule
from flash.core.data.io.input import DataKeys, Input
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS

from efficient_face.transform import FaceRecognitionInputTransform


class EfficientFaceImageInput(Input):
    def load_data(self, data_folder_path: Path) -> List[Dict[DataKeys, Any]]:
        class_mapping = {folder_name.name: index for index, folder_name in enumerate(data_folder_path.iterdir())}

        data = []
        for folder_name in data_folder_path.iterdir():
            if folder_name.is_dir():
                for image_file_name in folder_name.iterdir():
                    if image_file_name.is_file() and str(image_file_name).lower().endswith(IMG_EXTENSIONS):
                        data.append({DataKeys.INPUT: image_file_name, DataKeys.TARGET: class_mapping[folder_name.name]})

        return data

    def load_sample(self, sample: Dict[DataKeys, Any]) -> Dict[DataKeys, Any]:
        sample[DataKeys.INPUT] = Image.open(sample[DataKeys.INPUT]).convert("RGB")
        return sample


class EfficientFaceDataModule(DataModule):
    @classmethod
    def from_label_class_subfolders(
        cls,
        train_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        train_transform_kwargs: Optional[Dict] = None,
        val_transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "EfficientFaceDataModule":

        train_input = None
        if train_folder is not None:
            if train_transform_kwargs is None:
                train_transform_kwargs = {}

            train_input = EfficientFaceImageInput(
                RunningStage.TRAINING,
                Path(train_folder),
                transform=FaceRecognitionInputTransform(running_stage=RunningStage.TRAINING, **train_transform_kwargs),
            )

        val_input = None
        if val_folder is not None:
            if val_transform_kwargs is None:
                val_transform_kwargs = {}

            val_input = EfficientFaceImageInput(
                RunningStage.VALIDATING,
                Path(val_folder),
                transform=FaceRecognitionInputTransform(running_stage=RunningStage.VALIDATING, **val_transform_kwargs),
            )

        return cls(
            train_input=train_input,
            val_input=val_input,
            **data_module_kwargs,
        )
