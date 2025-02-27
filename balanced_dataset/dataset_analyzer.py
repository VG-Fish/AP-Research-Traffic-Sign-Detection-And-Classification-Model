from json import load, dump
from pprint import pprint as pp
from random import choices, choice
import numpy as np

DIRECTORY = "mtsd_v2_fully_annotated"
OUTPUT_DIR = "balanced_dataset"
IGNORE_PANORAMAS = True

# Ensures that files containing 40% or more minority or majority signs are inside the dataset.
MINORITY_SIGN_PERCENTS = {0.5, 0.75, 0.9, 1.0}
MINORITY_SIGN_LIMIT = 0.4

MAJORITY_SIGN_PERCENTS = {0.5, 0.75, 0.9, 1.0}
MAJORITY_SIGN_LIMIT = 0.4

background_images = []

classes_to_images = dict()

def add_classes_to_image(directory, file_name, annotation, bound):
    path = f"{directory}/{file_name}"
    for sign in annotation["objects"]:
        classes_to_images \
            .setdefault(sign["label"], {}) \
            .setdefault(bound, []) \
            .append(path)

def insert_files(directory):
    with open(f"{DIRECTORY}/splits/{directory}.txt", "r") as f:
        file_names = f.read().splitlines()

    for file_name in file_names:
        with open(f"{DIRECTORY}/annotations/{file_name}.json") as f:
            data = load(f)

            # Ignore panoramas
            if IGNORE_PANORAMAS and data["ispano"]:
                continue

            # Note down background images
            objects = data["objects"]
            num_signs = len(objects)
            if num_signs == 0:
                background_images.append(f"{directory}/{file_name}")
                continue
            
            # Note down minority and majority images
            counter = 0
            for sign in objects:
                name = sign["label"]
                if name != "other-sign":
                    counter += 1
        
            minority_sign_percent = counter / num_signs
            if minority_sign_percent < MINORITY_SIGN_LIMIT:
                continue
            for bound in MINORITY_SIGN_PERCENTS:
                if bound >= minority_sign_percent:
                    add_classes_to_image(directory, file_name, data, f"{bound}-minority")
                    break

            majority_sign_percent = 1 - minority_sign_percent
            if majority_sign_percent < MAJORITY_SIGN_LIMIT:
                continue
            for bound in MAJORITY_SIGN_PERCENTS:
                if bound >= majority_sign_percent:
                    add_classes_to_image(directory, file_name, data, f"{bound}-majority")
                    break

def insert_background_files():
    # Insert background images
    random_classes = choices(list(classes_to_images.keys()), k=len(background_images))
    for rand_class, file_name in zip(random_classes, background_images):
        rand_bound = choice(list(classes_to_images[rand_class].keys()))
        classes_to_images[rand_class][rand_bound].append(file_name)

insert_files("train")
insert_files("val")
insert_background_files()

with open(f"{OUTPUT_DIR}/dataset_information.json", "w") as f:
    dump(classes_to_images, f, indent=2)

# with open(f"{OUTPUT_DIR}/class_data.txt", "w") as f:
#     lines = "-" * 25
#     f.write(f"{lines}\nMinority Sign Bounds\n")
#     for bound, info in sorted(minority_sign_percents.items()):
#         output_str = f"Bound: {bound}\n"
#         for directory, images in info.items():
#             output_str += f"Number of {directory} images: {len(images)}.\n"
#         output_str += f"{lines}\n"
#         f.write(output_str)

#     f.write(f"\n\n{lines}\nMajority Sign Bounds\n")
#     for bound, info in sorted(majority_sign_percents.items()):
#         output_str = f"Bound: {bound}\n"
#         for directory, images in info.items():
#             output_str += f"Number of {directory} images: {len(images)}.\n"
#         output_str += f"{lines}\n"
#         f.write(output_str)