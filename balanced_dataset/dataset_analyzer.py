from json import load, dump
from pprint import pprint as pp
from random import choices
import numpy as np

DIRECTORY = "mtsd_v2_fully_annotated"
OUTPUT_DIR = "balanced_dataset"
IGNORE_PANORAMAS = True
MINORITY_SIGN_PERCENTS = {0.1, 0.25, 0.5, 0.75, 0.9, 1.0}
MAJORITY_SIGN_PERCENTS = {0.1, 0.25, 0.5, 0.75, 0.9, 1.0}

minority_sign_percents = {i: {"train": [], "val": [], "test": []} for i in sorted(MINORITY_SIGN_PERCENTS)}
majority_sign_percents = {i: {"train": [], "val": [], "test": []} for i in sorted(MAJORITY_SIGN_PERCENTS)}

background_images = []

def insert_files(directory):
    with open(f"{DIRECTORY}/splits/{directory}.txt", "r") as f:
        file_names = f.read().splitlines()

    for file_name in file_names:
        with open(f"{DIRECTORY}/annotations/{file_name}.json") as f:
            data = load(f)
            # Ignore panoramas
            if data["ispano"] and not IGNORE_PANORAMAS:
                continue

            # Note down background images
            objects = data["objects"]
            num_signs = len(objects)
            if num_signs == 0:
                background_images.append(file_name)
                continue
            
            # Note down minority and majority images
            counter = 0
            for sign in objects:
                name = sign["label"]
                if name != "other-sign":
                    counter += 1
        
            minority_sign_percent = counter / num_signs
            for bound, images in minority_sign_percents.items():
                if bound >= minority_sign_percent:
                    images[directory].append(file_name)
                    break

            majority_sign_percent = 1 - minority_sign_percent
            for bound, images in majority_sign_percents.items():
                if bound >= majority_sign_percent:
                    images[directory].append(file_name)
                    break

def insert_test_and_background_files():
    with open(f"{DIRECTORY}/splits/test.txt", "r") as f:
        file_names = f.read().splitlines()
    
    probability_calculations = dict()
    num_files = len(file_names) + len(background_images)
    total_num_images = num_files

    for bound, info in minority_sign_percents.items():
        num_images = sum([len(x) for x in info.values()])
        probability_calculations[bound] = num_images
        total_num_images += num_images
        
    probability_calculations = {bound: num_images / total_num_images for bound, num_images in probability_calculations.items()}
    
    random_minority_bounds = choices(list(MINORITY_SIGN_PERCENTS), k=num_files)
    random_majority_bounds = choices(list(MAJORITY_SIGN_PERCENTS), k=num_files)
    random_bounds = np.random.random(num_files)
    
    # Insert test images
    for (
        rand_num, file_name, random_minor_bound, random_major_bound
    ) in zip(random_bounds, file_names, random_minority_bounds, random_majority_bounds):
        if rand_num >= 0.5:
            minority_sign_percents[random_minor_bound]["test"].append(file_name)
        else:
            majority_sign_percents[random_major_bound]["test"].append(file_name)
    
    # Insert background images
    random_class_bounds = np.random.random(len(background_images))
    random_directories = choices(["train", "val", "test"], k=num_files)
    for (
        rand_num, file_name, random_minor_bound, random_major_bound, random_directory
    ) in zip(random_class_bounds, background_images, random_minority_bounds, random_majority_bounds, random_directories):
        if rand_num >= 0.5:
            minority_sign_percents[random_minor_bound][random_directory].append(file_name)
        else:
            majority_sign_percents[random_major_bound][random_directory].append(file_name)

insert_files("train")
insert_files("val")
insert_test_and_background_files()

output_data = {
    "minority_class_bounds": {
        str(bound): images for bound, images in minority_sign_percents.items()
    },
    "majority_class_bounds": {
        str(bound): images for bound, images in majority_sign_percents.items()
    },
}

with open(f"{OUTPUT_DIR}/class_information.json", "w") as f:
    dump(output_data, f, indent=2)

with open(f"{OUTPUT_DIR}/class_data.txt", "w") as f:
    lines = "-" * 25
    sep = ">" * 50
    f.write(f"{lines}\nMinority Sign Bounds\n")
    for bound, info in sorted(minority_sign_percents.items()):
        output_str = f"Bound: {bound}\n"
        for directory, images in info.items():
            output_str += f"Number of {directory} images: {len(images)}.\n"
        output_str += f"{lines}\n"
        f.write(output_str)

    f.write(f"{sep}\n{lines}\nMajority Sign Bounds\n")
    for bound, info in sorted(majority_sign_percents.items()):
        output_str = f"Bound: {bound}\n"
        for directory, images in info.items():
            output_str += f"Number of {directory} images: {len(images)}.\n"
        output_str += f"{lines}\n"
        f.write(output_str)