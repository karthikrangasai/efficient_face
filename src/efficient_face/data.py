from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torchvision.datasets
from datasets import load_from_disk
from flash import RunningStage
from flash.core.data.data_module import DataModule
from flash.core.data.io.input import DataKeys, Input
from PIL import Image
from torchvision.datasets import CIFAR10, LFWPairs
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

    # def val_load_data(self, data_folder_path: Path) -> Union[Sequence, Iterable]:
    #     """Override the ``val_load_data`` hook with data loading logic that is only required during validating.

    #     Args:
    #         *args: Any arguments that the input requires.
    #         **kwargs: Any additional keyword arguments that the input requires.
    #     """
    #     dataset = LFWPairs(
    #         root=str(data_folder_path),
    #     )
    #     return self.load_data(*args, **kwargs)

    def load_sample(self, sample: Dict[DataKeys, Any]) -> Dict[DataKeys, Any]:
        sample[DataKeys.INPUT] = Image.open(sample[DataKeys.INPUT]).convert("RGB")
        return sample


class EfficientFaceHFImageInput(Input):
    def load_data(self, data_folder_path: str) -> List[Dict[DataKeys, Any]]:
        dataset = load_from_disk(data_folder_path)
        return dataset

    def load_sample(self, sample: Dict[DataKeys, Any]) -> Dict[DataKeys, Any]:
        _sample = {}
        _sample[DataKeys.INPUT] = sample["image"].convert("RGB")
        _sample[DataKeys.TARGET] = sample["label"]
        return _sample


class Cifar10TestingInput(Input):
    def load_data(self, data_folder_path: str, train: bool) -> List[Dict[DataKeys, Any]]:
        dataset = CIFAR10(root=data_folder_path, train=train, download=True)
        return dataset

    def load_sample(self, sample) -> Dict[DataKeys, Any]:
        _sample = {}
        _sample[DataKeys.INPUT] = sample[0]
        _sample[DataKeys.TARGET] = sample[1]
        return _sample


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

    @classmethod
    def from_hf_datasets(
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

            train_input = EfficientFaceHFImageInput(
                RunningStage.TRAINING,
                train_folder,
                transform=FaceRecognitionInputTransform(running_stage=RunningStage.TRAINING, **train_transform_kwargs),
            )

        val_input = None
        if val_folder is not None:
            if val_transform_kwargs is None:
                val_transform_kwargs = {}

            val_input = EfficientFaceHFImageInput(
                RunningStage.VALIDATING,
                val_folder,
                transform=FaceRecognitionInputTransform(running_stage=RunningStage.VALIDATING, **val_transform_kwargs),
            )

        return cls(
            train_input=train_input,
            val_input=val_input,
            **data_module_kwargs,
        )

    @classmethod
    def from_cifar10(
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

            train_input = Cifar10TestingInput(
                RunningStage.TRAINING,
                train_folder,
                True,
                transform=FaceRecognitionInputTransform(running_stage=RunningStage.TRAINING, **train_transform_kwargs),
            )

        val_input = None
        if val_folder is not None:
            if val_transform_kwargs is None:
                val_transform_kwargs = {}

            val_input = Cifar10TestingInput(
                RunningStage.VALIDATING,
                val_folder,
                False,
                transform=FaceRecognitionInputTransform(running_stage=RunningStage.VALIDATING, **val_transform_kwargs),
            )

        return cls(
            train_input=train_input,
            val_input=val_input,
            **data_module_kwargs,
        )
