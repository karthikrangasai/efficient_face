from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from efficient_face.data.cifair import ciFAIR10, ciFAIR100
from efficient_face.data.transform import get_image_transform
from efficient_face.paths import DATA_DIR


class ciFAIRDataModule(LightningDataModule):
    def __init__(
        self,
        root: Path = DATA_DIR,
        batch_size: int = 16,
        model_name: str = "efficientnet_b0",
        image_size: Union[int, Tuple[int, int]] = 226,
        crop_size: Union[int, Tuple[int, int]] = 224,
    ) -> None:

        super().__init__()

        self.root = root
        self.batch_size = batch_size
        self.model_name = model_name
        self.image_size = image_size
        self.crop_size = crop_size

        self.image_transform = get_image_transform(
            model_name=self.model_name,
            image_size=self.image_size,
            crop_size=self.crop_size,
        )

        self.target_transform = torch.tensor

    def prepare_data(self) -> None:
        ciFAIR10(str(self.root), train=True, download=True)
        ciFAIR10(str(self.root), train=False, download=True)
        ciFAIR100(str(self.root), train=True, download=True)
        ciFAIR100(str(self.root), train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # Entire ciFAIR10 dataset
        self.train_set = ciFAIR10(
            str(self.root), train=True, transform=self.image_transform, target_transform=self.target_transform
        ) + ciFAIR10(
            str(self.root), train=False, transform=self.image_transform, target_transform=self.target_transform
        )

        # Entire ciFAIR100 dataset
        self.val_set = ciFAIR100(
            str(self.root), train=True, transform=self.image_transform, target_transform=self.target_transform
        ) + ciFAIR100(
            str(self.root), train=False, transform=self.image_transform, target_transform=self.target_transform
        )

    def train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(self.train_set, self.batch_size, shuffle=True, pin_memory=True, num_workers=2)
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(self.val_set, self.batch_size, shuffle=False, pin_memory=True, num_workers=2)
        return val_dataloader
