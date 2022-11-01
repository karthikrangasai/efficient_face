import time
from argparse import ArgumentParser
from datetime import datetime

from datasets import load_dataset
from dateutil import relativedelta


def main(data_folder: str, split: str, output_folder: str):
    print("[INFO]: Starting `load_dataset`.", flush=True)
    dt0 = datetime.fromtimestamp(time.time())
    dataset = load_dataset("imagefolder", data_dir=data_folder, split=split)
    dt1 = datetime.fromtimestamp(time.time())
    print("[INFO]: Finished `load_dataset`.", flush=True)
    rd = relativedelta.relativedelta(dt1, dt0)
    print(
        "\n"
        "Total time: "
        f"{rd.years}Y {rd.months}M {rd.days}D {rd.hours}h {rd.minutes}m {rd.seconds}s {rd.microseconds}us"
        "\n",
        flush=True,
    )

    print("[INFO]: Starting `dataset.save_to_disk`.", flush=True)
    dt2 = datetime.fromtimestamp(time.time())
    dataset.save_to_disk(dataset_path=output_folder)
    dt3 = datetime.fromtimestamp(time.time())
    print("[INFO]: Finished `dataset.save_to_disk`.", flush=True)
    rd = relativedelta.relativedelta(dt3, dt2)
    print(
        "\n"
        "Total time: "
        f"{rd.years}Y {rd.months}M {rd.days}D {rd.hours}h {rd.minutes}m {rd.seconds}s {rd.microseconds}us"
        "\n",
        flush=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_folder", required=True, type=str)
    parser.add_argument("--split", required=True, type=str)
    parser.add_argument("--output_folder", required=True, type=str)
    args = vars(parser.parse_args())
    main(
        data_folder=args["data_folder"],
        split=args["split"],
        output_folder=args["output_folder"],
    )
