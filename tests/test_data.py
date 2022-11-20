# from pathlib import Path
# from typing import Any, Dict, List, Tuple

# from flash.core.data.io.input import DataKeys
# from pytorch_lightning import seed_everything
# from torch import Tensor
# from torch.utils.data import DataLoader

# from efficient_face.data import ciFAIRDataModule


# def assert_batch_values(batch: Dict[DataKeys, Any]) -> None:
#     assert DataKeys.INPUT in batch.keys()
#     assert DataKeys.TARGET in batch.keys()

#     assert isinstance(batch[DataKeys.INPUT], Tensor)
#     assert isinstance(batch[DataKeys.TARGET], Tensor)


# def test_data_loading(random_dataset_path: Path) -> None:
#     seed_everything(1234)

#     datamodule = ciFAIRDataModule.load_ciFAIR10(root=random_dataset_path, batch_size=4)

#     train_dataloader: DataLoader = datamodule.train_dataloader()  # type: ignore
#     batch: Dict[DataKeys, Any] = next(iter(train_dataloader))
#     assert_batch_values(batch)

#     val_dataloader: DataLoader = datamodule.val_dataloader()  # type: ignore
#     batch = next(iter(val_dataloader))
#     assert_batch_values(batch)
