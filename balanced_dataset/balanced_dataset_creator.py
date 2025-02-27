from json import load
from typing import Dict, Any, List
from os import makedirs, link
from os.path import abspath, exists, samefile
from random import choices
from shutil import copy2, rmtree, copytree
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

def apply_augmentations_and_save(image_paths: List[str], directory: str) -> None:
    core_stem = abspath(f"{MAPILLARY_DATASET_DIRECTORY}/{directory}")
    new_stem = abspath(f"{BALANCED_DATASET_DIRECTORY}/{directory}")

    for image_path in image_paths:
        core_image = f"{core_stem}/images/{image_path}.jpg"
        core_label = f"{core_stem}/labels/{image_path}.txt"
        
        new_image = f"{new_stem}/images/{image_path}.jpg"
        new_label = f"{new_stem}/labels/{image_path}.txt"

        try:
            link(core_image, new_image)
        except OSError:
            if not samefile(core_image, new_image):
                copy2(core_image, new_image)

        if exists(core_label):
            try:
                link(core_label, new_label)
            except OSError:
                if not samefile(core_label, new_label):
                    copy2(core_label, new_label)

def upload_data(data: Dict[str, Any], directory: str) -> None:
    amount = SPLITS["amount_per_class"]
    amount_of_classes = len(data.keys())

    counter = 0
    for traffic_sign_class, info in data.items():
        print(round(counter / amount_of_classes, 2), traffic_sign_class)
        current_amount = int(SPLITS[directory] * amount)
        random_bounds = choices(list(info.keys()), k=current_amount)
        for random_bound in random_bounds:
            current_bound_dict = info[random_bound]
            image_directory = current_bound_dict.get(directory, False)
            if not image_directory:
                break
            images = choices(image_directory, k=current_amount)
            apply_augmentations_and_save(images, directory)
        counter += 1

    print("\n" * 5)

def upload_test():
    source_directory = abspath(f"{MAPILLARY_DATASET_DIRECTORY}/test/images")
    destination_directory = abspath(f"{BALANCED_DATASET_DIRECTORY}/test/images")
    copytree(source_directory, destination_directory, dirs_exist_ok=True)

def create_dataset() -> None:
    # To test if the given data is valid.s
    assert \
        abs(SPLITS["train"] + SPLITS["val"] + SPLITS["test"] - 1.0) <= 1e-5, \
        "Your dataset division for train, val, and test doesn't equal one."

    data = load_data()
    print("Loaded Data.")
    
    print("Uploading train data.")
    upload_data(data, "train")
    print("Finished uploading train data.")

    print("Uploading val data.")
    upload_data(data, "val")
    print("Finished uploading val data.")

    print("Uploading test data.")
    UPLOADED_TEST = True
    if not UPLOADED_TEST:
        upload_test()
    print("Finished uploading test data.")

    rmtree(f"{BALANCED_DATASET_DIRECTORY}/test/labels")

def main() -> None:
    create_directories(True)
    create_dataset()
    # verify_dataset()

if __name__ == "__main__":
    main()