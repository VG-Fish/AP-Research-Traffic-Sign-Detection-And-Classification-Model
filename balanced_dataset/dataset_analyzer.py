from json import load, dump
from random import choices, choice

DIRECTORY = "mtsd_v2_fully_annotated"
OUTPUT_DIR = "balanced_dataset"
IGNORE_PANORAMAS = True

# Ensures that files containing X% or more minority or majority signs are inside the dataset.
MINORITY_SIGN_PERCENTS = {1.0}
MINORITY_SIGN_LIMIT = 0.75

MAJORITY_SIGN_PERCENTS = {}
MAJORITY_SIGN_LIMIT = 2

background_images = []

classes_to_images = dict()

def add_classes_to_image(directory, file_name, annotation, bound):
    for sign in annotation["objects"]:
        class_directory = classes_to_images \
            .setdefault(sign["label"], {}) \
            .setdefault(bound, {}) \
            .setdefault(directory, set())
        if len(classes_to_images[sign["label"]][bound][directory]) <= 100 and sign["label"] != "other-sign":
            class_directory.add(file_name)

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
    counter = 0
    while counter < len(background_images):
        rand_class = random_classes[counter]
        directory, file_name = background_images[counter].split("/")
        rand_bound = choice(list(classes_to_images[rand_class].keys()))

        # To ensure that the background image ends up at the correct directory
        if directory not in classes_to_images[rand_class][rand_bound].keys():
            counter += 1
            continue
        
        classes_to_images[rand_class][rand_bound][directory].add(file_name)
        counter += 1

insert_files("train")
insert_files("val")
insert_background_files()

# Make the dictionary serializable. 
for info in classes_to_images.values():
    for bounds in info.values():
        for directory in bounds:
            bounds[directory] = list(bounds[directory])

with open(f"{OUTPUT_DIR}/rare_dataset_information.json", "w") as f:
    dump(classes_to_images, f, indent=2)

with open(f"{OUTPUT_DIR}/rare_class_data.txt", "w") as f:
    lines = "-" * 25
    for traffic_sign_class, info in classes_to_images.items():
        f.write(f"{lines}\n{traffic_sign_class}:\n")
        for bound, directories in sorted(info.items()):
            for directory, images in directories.items():
                output_str = f"\nBound: {bound}\n"
                output_str += f"The number of images in the {directory} directory: {len(images)}.\n"
            f.write(output_str)
        f.write("\n")