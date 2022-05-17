from argparse import ArgumentParser
from typing import Dict

from datasets import Dataset, load_dataset
from flash.core.data.io.input import DataKeys

from efficient_face.utils import profile


def transforms(examples: Dict) -> Dict:
    examples[DataKeys.INPUT] = [image.convert("RGB") for image in examples["image"]]
    examples[DataKeys.TARGET] = examples["label"]
    return examples


def main(
    data_folder: str,
    split: str,
    output_folder: str,
    apply_mapping: bool,
    batch_size: int,
    writer_batch_size: int,
    num_proc: int,
) -> None:

    dataset: Dataset = profile(
        function=load_dataset, fn_kwargs=dict(path="imagefolder", data_dir=data_folder, split=split)
    )

    if apply_mapping:
        dataset: Dataset = profile(
            function=dataset.map,
            fn_kwargs=dict(
                function=transforms,
                batched=True,
                remove_columns=["image", "label"],
                batch_size=batch_size,
                writer_batch_size=writer_batch_size,
                num_proc=num_proc,
            ),
        )

    profile(function=dataset.save_to_disk, fn_kwargs=dict(dataset_path=output_folder))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_folder", required=True, type=str)
    parser.add_argument("--split", required=True, type=str)
    parser.add_argument("--output_folder", required=True, type=str)

    parser.add_argument("--apply_mapping", action="store_true", required=False)
    parser.add_argument("--batch_size", required=False, type=int, default=10)
    parser.add_argument("--writer_batch_size", required=False, type=int, default=10)
    parser.add_argument("--num_proc", required=False, type=int, default=None)
    args = vars(parser.parse_args())
    main(
        data_folder=args["data_folder"],
        split=args["split"],
        output_folder=args["output_folder"],
        apply_mapping=args["apply_mapping"],
        batch_size=args["batch_size"],
        writer_batch_size=args["writer_batch_size"],
        num_proc=args["num_proc"],
    )
