from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image


def _rand_image() -> Image.Image:
    _size = np.random.choice([196, 244])
    size = (_size, _size)
    return Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype="uint8"))


def random_dataset_path(path: str) -> Path:
    temp_path: Path = Path(path)
    temp_path = temp_path / "data_example"
    for i in range(20):
        if (i % 2) == 0:
            d = temp_path / f"class{i/2}"
            d.mkdir()
            for j in range(40):
                _rand_image().save(d / f"image_{j}.jpg")
        else:
            d = temp_path / f"text_file_{i//2}.txt"
            d.write_text("Hello, World.")
    return temp_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_folder", required=True, type=str)
    args = vars(parser.parse_args())
    random_dataset_path(path=args["data_folder"])
