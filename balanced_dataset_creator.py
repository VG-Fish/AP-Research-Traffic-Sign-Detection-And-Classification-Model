from json import load
from typing import Dict, Any
from os import makedirs

"""
Get X random images from images with the {minority, majority, background} class presence of Y.
You can also define the train/val/test format too.
"""
SPLITS: Dict[str, Dict[int, float] | int] = {
    "minority": {
        0.5: 200,
        0.75: 460,
        1.0: 1070,
    },
    "majority": {
        4: 200,
        12: 50,
        16: 50,
        20: 30,
    },
    "background": 40,
    "train": 0.8,
    "val": 0.2,
    "test": 0.1,
}
"""
The parent directory you want to save to.
"""
DATASET_DIRECTORY = "balanced_augmented_mapillary_dataset"

with open("class_information.json", "r") as f:
    contents: Dict[str, Any] = load(f)

def create_directories(exist_ok: bool = False) -> None:
    directories = ["train", "val", "test"]
    for directory in directories:
        makedirs(f"{DATASET_DIRECTORY}/{directory}/images", exist_ok=exist_ok)
        makedirs(f"{DATASET_DIRECTORY}/{directory}/labels", exist_ok=exist_ok)

def main() -> None:
    create_directories(True)

if __name__ == "__main__":
    main()