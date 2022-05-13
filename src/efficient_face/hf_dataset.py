import time
from argparse import ArgumentParser
from datetime import datetime

from datasets import load_dataset
from dateutil import relativedelta
from flash.core.data.io.input import DataKeys


def transforms(examples):
    examples[DataKeys.INPUT] = [image.convert("RGB") for image in examples["image"]]
    examples[DataKeys.TARGET] = examples["label"]
    return examples


def main(data_folder: str, split: str, output_folder: str, batch_size: int, writer_batch_size: int, num_proc: int):
    print("[INFO]: Starting `load_dataset`.", flush=True)
    dataset = load_dataset("imagefolder", data_dir=data_folder, split=split)
    print("[INFO]: Finished `load_dataset`.", flush=True)

    print("[INFO]: Starting `dataset.map`.", flush=True)
    dt0 = datetime.fromtimestamp(time.time())

    dataset = dataset.map(
        function=transforms,
        batched=True,
        remove_columns=["image", "label"],
        batch_size=batch_size,
        writer_batch_size=writer_batch_size,
        num_proc=num_proc,
    )

    dt1 = datetime.fromtimestamp(time.time())
    print("[INFO]: Finished `dataset.map`.", flush=True)
    rd = relativedelta.relativedelta(dt1, dt0)
    print(
        "\n"
        "Total time: "
        f"{rd.years}Y {rd.months}M {rd.days}D {rd.hours}h {rd.minutes}m {rd.seconds}s {rd.microseconds}us"
        "\n",
        flush=True,
    )

    print("[INFO]: Starting `dataset.save_to_disk`.", flush=True)
    dataset.save_to_disk(output_folder)
    print("[INFO]: Finished `dataset.save_to_disk`.", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_folder", required=True, type=str)
    parser.add_argument("--split", required=True, type=str)
    parser.add_argument("--output_folder", required=True, type=str)

    parser.add_argument("--batch_size", required=False, type=int, default=10)
    parser.add_argument("--writer_batch_size", required=False, type=int, default=10)
    parser.add_argument("--num_proc", required=False, type=int, default=None)
    args = vars(parser.parse_args())
    main(
        data_folder=args["data_folder"],
        split=args["split"],
        output_folder=args["output_folder"],
        batch_size=args["batch_size"],
        writer_batch_size=args["writer_batch_size"],
        num_proc=args["num_proc"],
    )
