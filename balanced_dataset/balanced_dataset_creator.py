from json import load
from typing import Dict, Any, Set
from os import makedirs
from os.path import join, abspath
from random import choice, sample
from shutil import copy, rmtree
from balanced_dataset.balanced_dataset_verifier import verify_dataset

"""
Get X random images from images with the {minority, majority, background} class presence of Y.
You can also define the train/val/test format too.
"""
# Adds up to 4,000 images.
SPLITS: Dict[str, Dict[int, float] | int | float] = {
    "minority": {
        1.0: 5050,
    },
    "majority": {
        20: 30,
    },
    "background": 20,
    "train": 0.7,
    "val": 0.2,
    "test": 0.1,
}
"""
The parent directory you want to save to.
"""
BALANCED_DATASET_DIRECTORY = "balanced_mapillary_dataset"
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
                while True and bound / amount < 0.5:
                    rand_img = choice(minority_data)
                    if rand_img not in used_images:
                        used_images.add(rand_img)
                        break
                # trash, hacky, monkey-patching code
                source_img_path = f"{join(f'{MAPILLARY_DATASET_DIRECTORY}/train/images', rand_img)}.jpg"
                destination_img_path = f"{join(f'{BALANCED_DATASET_DIRECTORY}/{directory}/images', rand_img)}.jpg"
                try:
                    copy(abspath(source_img_path), abspath(destination_img_path))
                    source_label_path = f"{join(f'{MAPILLARY_DATASET_DIRECTORY}/train/labels', rand_img)}.txt"
                    destination_label_path = f"{join(f'{BALANCED_DATASET_DIRECTORY}/{directory}/labels', rand_img)}.txt"
                    copy(abspath(source_label_path), abspath(destination_label_path))
                except FileNotFoundError:
                    try:
                        source_img_path = f"{join(f'{MAPILLARY_DATASET_DIRECTORY}/val/images', rand_img)}.jpg"
                        copy(abspath(source_img_path), abspath(destination_img_path))
                        source_label_path = f"{join(f'{MAPILLARY_DATASET_DIRECTORY}/val/labels', rand_img)}.txt"
                        destination_label_path = f"{join(f'{BALANCED_DATASET_DIRECTORY}/{directory}/labels', rand_img)}.txt"
                        copy(abspath(source_label_path), abspath(destination_label_path))
                    except FileNotFoundError:
                        try:
                            source_img_path = f"{join(f'{MAPILLARY_DATASET_DIRECTORY}/test/images', rand_img)}.jpg"
                            copy(abspath(source_img_path), abspath(destination_img_path))
                        except FileNotFoundError:
                            continue
                counter += 1

def create_dataset() -> None:
    dataset_splits = {
        "train": SPLITS["train"],
        "val": SPLITS["val"], 
        "test": SPLITS["test"],
    }

    # To test if the given data is valid.
    assert abs(sum(dataset_splits.values()) - 1.0) <= 1e-5, "Your dataset division for train, val, and test doesn't equal one."
    assert SPLITS.get("majority", False), "Define your majority class image data."
    assert SPLITS.get("minority", False), "Define your minority class image data."
    assert SPLITS.get("background", False), "Define your background image data."

    data = load_data()
    upload_data(data, dataset_splits, "minority")
    upload_data(data, dataset_splits, "majority")
    # I manually added background images
    rmtree(f"{BALANCED_DATASET_DIRECTORY}/test/labels")


def main() -> None:
    create_directories(True)
    create_dataset()
    verify_dataset()

if __name__ == "__main__":
    main()