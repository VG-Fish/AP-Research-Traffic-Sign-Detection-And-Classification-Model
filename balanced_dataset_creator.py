from json import load
from typing import Dict, Any, Set
from os import makedirs
from os.path import join, abspath
from random import choice
from shutil import copy, rmtree

"""
Get X random images from images with the {minority, majority, background} class presence of Y.
You can also define the train/val/test format too.
"""
# Adds up to 4,000 images.
SPLITS: Dict[str, Dict[int, float] | int | float] = {
    "minority": {
        0.5: 400,
        0.75: 920,
        1.0: 2140,
    },
    "majority": {
        2: 130,
        4: 200,
        12: 50,
        16: 50,
        20: 30,
    },
    "background": 80,
    "train": 0.7,
    "val": 0.2,
    "test": 0.1,
}
"""
The parent directory you want to save to.
"""
BALANCED_DATASET_DIRECTORY = "balanced_augmented_mapillary_dataset"
MAPILLARY_DATASET_DIRECTORY = "mapillary_dataset"

def create_directories(exist_ok: bool = False) -> None:
    directories = ["train", "val", "test"]
    for directory in directories:
        rmtree(f"{BALANCED_DATASET_DIRECTORY}/{directory}")
        makedirs(f"{BALANCED_DATASET_DIRECTORY}/{directory}/images", exist_ok=exist_ok)
        makedirs(f"{BALANCED_DATASET_DIRECTORY}/{directory}/labels", exist_ok=exist_ok)

def load_data() -> Dict[str, Any]:
     # Get all the data
    with open("class_information.json", "r") as f:
        return load(f)

def upload_data(data: Dict[str, Any], dataset_splits: Dict[str, int], type: str) -> None:
    used_images: Set[str] = set()
    for bound, amount in SPLITS[type].items():
        minority_data = data[f"{type}_class_bounds"][str(bound)]

        for directory, percent in dataset_splits.items():
            num_images = int(amount * percent)
            assert num_images <= len(minority_data), f"{num_images}, for bound {bound}, must be lowered."
            
            counter = 0
            while counter < num_images:
                rand_img = choice(minority_data)
                if rand_img in used_images:
                    used_images.add(rand_img)
                    continue

                # trash, hacky, monkey-patching code
                if directory != "test":
                    source_path = f"{join(f'{MAPILLARY_DATASET_DIRECTORY}/train/images', rand_img)}.jpg"
                    destination_path = f"{join(f'{BALANCED_DATASET_DIRECTORY}/{directory}/images', rand_img)}.jpg"
                    try:
                        copy(abspath(source_path), abspath(destination_path))
                    except FileNotFoundError:
                        source_path = f"{join(f'{MAPILLARY_DATASET_DIRECTORY}/val/images', rand_img)}.jpg"
                        copy(abspath(source_path), abspath(destination_path))
                counter += 1

def create_dataset() -> None:
    dataset_splits = {
        "train": SPLITS["train"],
        "val": SPLITS["val"], 
        "test": SPLITS["test"],
    }

    # To test if the given data is valid.
    assert abs(sum(dataset_splits.values()) - 1.0) <= 1e-5, "Your division for train, val, and test doesn't equal one."
    assert SPLITS.get("majority", False), "Define your majority class image data."
    assert SPLITS.get("minority", False), "Define your minority class image data."
    assert SPLITS.get("background", False), "Define your background image data."

    data = load_data()
    upload_data(data, dataset_splits, "minority")
    upload_data(data, dataset_splits, "majority")


def main() -> None:
    create_directories(True)
    create_dataset()

if __name__ == "__main__":
    main()