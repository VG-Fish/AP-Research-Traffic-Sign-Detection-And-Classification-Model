from json import load
from typing import Dict, Any
from os import makedirs
from os.path import join, abspath, exists
from random import sample
from shutil import copy, rmtree
from balanced_dataset_verifier import verify_dataset

SPLITS: Dict[str, int | float] = {
    "amount_per_class": 25,
    "train": 0.7,
    "val": 0.2,
    "test": 0.1,
}

"""
The parent directory you want to save to.
"""
BALANCED_DATASET_DIRECTORY = "balanced_augmented_mapillary_dataset"
MAPILLARY_DATASET_DIRECTORY = "mapillary_dataset"
DATASET_INFORMATION_PATH = "balanced_dataset/dataset_information.json"

def create_directories(exist_ok: bool = False) -> None:
    directories = ["train", "val", "test"]
    for directory in directories:
        if exists(f"{BALANCED_DATASET_DIRECTORY}/{directory}"):
            rmtree(f"{BALANCED_DATASET_DIRECTORY}/{directory}")
        makedirs(f"{BALANCED_DATASET_DIRECTORY}/{directory}/images", exist_ok=exist_ok)
        makedirs(f"{BALANCED_DATASET_DIRECTORY}/{directory}/labels", exist_ok=exist_ok)

def load_data() -> Dict[str, Any]:
     # Get all the data
    with open(DATASET_INFORMATION_PATH) as f:
        return load(f)

def upload_data(data: Dict[str, Any], for_train: bool) -> None:
    minority_class_bounds = data["minority_class_bounds"].keys()
    majority_class_bounds = data["majority_class_bounds"].keys()

    total_minority_bounds = len(minority_class_bounds)
    total_majority_bounds = len(majority_class_bounds)

    # Ensure that total_{minority, majority}_bounds is < SPLITS["amount_per_class"]
    amount_per_minority_bound = total_minority_bounds / SPLITS["amount_per_class"]
    assert amount_per_minority_bound <= 1, "Ensure that total_minority_bounds is < SPLITS['amount_per_class']"
    amount_per_minority_bound = SPLITS["amount_per_class"] * amount_per_minority_bound
    amount_per_minority_bound *= SPLITS["train"] if for_train else SPLITS["val"]
    amount_per_minority_bound = int(amount_per_minority_bound)

    amount_per_majority_bound = total_majority_bounds / SPLITS["amount_per_class"]
    assert amount_per_majority_bound <= 1, "Ensure that total_majority_bounds is < SPLITS['amount_per_class']"
    amount_per_majority_bound = int(SPLITS["amount_per_class"] * amount_per_majority_bound)
    amount_per_majority_bound *= SPLITS["train"] if for_train else SPLITS["val"]
    amount_per_majority_bound = int(amount_per_majority_bound)

    print(amount_per_minority_bound, amount_per_majority_bound)


def create_dataset() -> None:
    # To test if the given data is valid.
    assert \
        abs(SPLITS["train"] + SPLITS["val"] + SPLITS["test"] - 1.0) <= 1e-5, \
        "Your dataset division for train, val, and test doesn't equal one."

    data = load_data()
    upload_data(data, True)
    upload_data(data, False)

    rmtree(f"{BALANCED_DATASET_DIRECTORY}/test/labels")


def main() -> None:
    create_directories(True)
    create_dataset()
    # verify_dataset()

if __name__ == "__main__":
    main()