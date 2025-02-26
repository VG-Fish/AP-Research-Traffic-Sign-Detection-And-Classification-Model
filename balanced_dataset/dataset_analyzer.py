from json import load, dump
from pprint import pprint as pp
from random import random, choices

DIRECTORY = "mtsd_v2_fully_annotated"
OUTPUT_DIR = "balanced_dataset"
IGNORE_PANORAMAS = True
MINORITY_SIGN_PERCENTS = {0.1, 0.25, 0.5, 0.75, 1.0}
MAJORITY_SIGN_PERCENTS = {2, 4, 8, 12, 16, 20, 20}

minority_sign_percents = {i: [] for i in MINORITY_SIGN_PERCENTS}
majority_sign_percents = {i: [] for i in MAJORITY_SIGN_PERCENTS}

background_images = []

def find_files(directory):
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
                if minority_sign_percent >= bound:
                    images.append(f"{directory}/{file_name}")
        
            majority_sign_percent = 1 / minority_sign_percent if counter != 0 else num_signs
            for bound, images in majority_sign_percents.items():
                if majority_sign_percent >= bound:
                    images.append(f"{directory}/{file_name}")

def find_test_files():
    with open(f"{DIRECTORY}/splits/test.txt", "r") as f:
        file_names = f.read().splitlines()
    
    random_minority_bounds = choices(list(MINORITY_SIGN_PERCENTS), k=len(file_names))
    random_majority_bounds = choices(list(MAJORITY_SIGN_PERCENTS), k=len(file_names))
    for (
        file_name, random_minor_bound, random_major_bound
    ) in zip(file_names, random_minority_bounds, random_majority_bounds):
        if random() >= 0.5:
            minority_sign_percents[random_minor_bound].append(f"test/{file_name}")
        else:
            majority_sign_percents[random_major_bound].append(f"test/{file_name}")

find_files("train")
find_files("val")
find_test_files()

output_data = {
    "minority_class_bounds": {
        str(bound): images for bound, images in minority_sign_percents.items()
    },
    "majority_class_bounds": {
        str(bound): images for bound, images in majority_sign_percents.items()
    },
    "background_images": background_images
}

with open(f"{OUTPUT_DIR}/class_information.json", "w") as f:
    dump(output_data, f, indent=2)

with open(f"{OUTPUT_DIR}/class_data.txt", "w") as f:
    f.write("Minority Sign Bounds\n")
    for bound, images in sorted(minority_sign_percents.items()):
        f.write(f"Bound: {bound}, Number of Images: {len(images)}\n")

    f.write("\nMajority Sign Bounds\n")
    for bound, images in sorted(majority_sign_percents.items()):
        f.write(f"Bound: {bound}, Number of Images: {len(images)}\n")

    f.write(f"\nNumber of background images: {len(background_images)}\n")