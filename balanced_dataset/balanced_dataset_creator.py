from json import load
from typing import Dict, Any, List
from os import makedirs, link, listdir
from os.path import abspath, exists, samefile
from random import choices
from shutil import copy2, rmtree, copytree
import albumentations as A
from tqdm import tqdm
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2RGB
from multiprocessing import Pool, cpu_count
# from balanced_dataset_verifier import verify_dataset

SPLITS: Dict[str, int | float] = {
    "amount_per_class": 25,
    "train": 0.7,
    "val": 0.2,
    "test": 0.1,
}
IMAGE_TRANSFORM = A.Compose([
    A.MotionBlur(p=0.1), # Simulating realistic camera conditions while driving
    A.RandomToneCurve(p=0.01), # Switches night to day
    A.SomeOf([
        # Standard image augmentations
        A.RandomBrightnessContrast(),
        A.RandomGamma(),
    ], n=2, p=0.5),
    A.SomeOf([
        # More simulation of realistic camera conditions while driving
        A.OpticalDistortion(),
        A.AdditiveNoise("gaussian"),
        A.RandomShadow(),
        A.AutoContrast(method="pil"),
        A.Illumination(),
    ], n=2, p=0.5),
    A.OneOf([
        # Random weather conditions
        A.RandomFog(),
        A.RandomSunFlare(),
        A.RandomRain(),
    ], p=0.05),
    A.OneOf([
        # These transformations tries to get the model to focus less on color and more on shape
        A.HueSaturationValue(),
        A.ChannelShuffle(),
    ], p=0.1),
])
NUM_AUGMENTATIONS = 5

"""
The parent directory you want to save to.
"""
BALANCED_DATASET_DIRECTORY = "balanced_augmented_mapillary_dataset"
MAPILLARY_DATASET_DIRECTORY = "mapillary_dataset"
DATASET_INFORMATION_PATH = "balanced_dataset/dataset_information.json"
UPLOADED_TEST = True

def create_directories(exist_ok: bool = False) -> None:
    directories = ["train", "val"]
    for directory in directories:
        if exists(f"{BALANCED_DATASET_DIRECTORY}/{directory}"):
            rmtree(f"{BALANCED_DATASET_DIRECTORY}/{directory}")
        if exists(f"{BALANCED_DATASET_DIRECTORY}/{directory}-augmented"):
            rmtree(f"{BALANCED_DATASET_DIRECTORY}/{directory}-augmented")

        makedirs(f"{BALANCED_DATASET_DIRECTORY}/{directory}/images", exist_ok=exist_ok)
        makedirs(f"{BALANCED_DATASET_DIRECTORY}/{directory}/labels", exist_ok=exist_ok)
        makedirs(f"{BALANCED_DATASET_DIRECTORY}/{directory}-augmented/images", exist_ok=exist_ok)
        makedirs(f"{BALANCED_DATASET_DIRECTORY}/{directory}-augmented/labels", exist_ok=exist_ok)
    
    if not UPLOADED_TEST:
        makedirs(f"{BALANCED_DATASET_DIRECTORY}/{directory}/test/images", exist_ok=True)

def load_data() -> Dict[str, Any]:
     # Get all the data
    with open(DATASET_INFORMATION_PATH) as f:
        return load(f)

def save_images(image_paths: List[str], directory: str) -> None:
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
    amount_per_class = SPLITS["amount_per_class"]
    target_amount = int(SPLITS[directory] * amount_per_class)
    amount_of_classes = len(data.keys())

    for counter, (traffic_sign_class, info) in enumerate(data.items(), start=0):

        available_images = []
        for bound_dict in info.values():
            if directory in bound_dict:
                available_images.extend(bound_dict[directory])
        
        if len(available_images) <= target_amount:
            selected_images = available_images
        else:
            selected_images = choices(available_images, k=target_amount)
        
        save_images(selected_images, directory)
        print(f"{round(counter / amount_of_classes * 100, 2):02f}%: {traffic_sign_class = }, Number of images = {len(selected_images)}")

    print("\n" * 3)

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
    if not UPLOADED_TEST:
        upload_test()
    print("Finished uploading test data.")

def process_image(args):
    image_name, input_directory, output_directory, num_augmentations = args
    image_path = f"{input_directory}/{image_name}"
    image = imread(image_path)
    image = cvtColor(image, COLOR_BGR2RGB)

    for _ in range(num_augmentations):
        augmented = IMAGE_TRANSFORM(image=image)['image']
        output_path = f"{output_directory}/{image_name}"
        imwrite(output_path, augmented)

def apply_augmentations():    
    for directory in ["train", "val"]:
        input_directory = f"{BALANCED_DATASET_DIRECTORY}/{directory}/images"
        output_directory = f"{BALANCED_DATASET_DIRECTORY}/{directory}-augmented/images"
        args_list = [
            (filename, input_directory, output_directory, NUM_AUGMENTATIONS) 
            for filename in listdir(input_directory) 
        ]
        with Pool(processes=cpu_count()) as pool:
            list(tqdm(pool.imap(process_image, args_list), total=len(args_list)))
        
        source_directory = abspath(f"{MAPILLARY_DATASET_DIRECTORY}/{directory}/labels")
        destination_directory = abspath(f"{BALANCED_DATASET_DIRECTORY}/{directory}-augmented/labels")
        copytree(source_directory, destination_directory, dirs_exist_ok=True)

        print(f"Successfully applied transformations to {directory} directory.")


def main() -> None:
    create_directories(True)
    create_dataset()
    apply_augmentations()
    print("Successfully created dataset.")
    # verify_dataset(BALANCED_DATASET_DIRECTORY)

if __name__ == "__main__":
    main()