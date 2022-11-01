import time
from argparse import ArgumentParser
from datetime import datetime

from datasets import load_from_disk
from dateutil import relativedelta


def main(data_folder: str):
    print("[INFO]: Starting `load_from_disk`.", flush=True)
    dt0 = datetime.fromtimestamp(time.time())

    dataset = load_from_disk(dataset_path=data_folder)

    dt1 = datetime.fromtimestamp(time.time())
    print("[INFO]: Finished `load_from_disk`.", flush=True)

    rd = relativedelta.relativedelta(dt1, dt0)
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
    args = vars(parser.parse_args())
    main(data_folder=args["data_folder"])
