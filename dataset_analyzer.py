from pathlib import Path
from json import load, dump

directory = "mtsd_v2_fully_annotated/annotations"
files = Path(directory).glob("*.json")
minority_sign_percents = {
    0.1: [],
    0.5: [],
    0.75: [],
    1.00: [],
}
majority_sign_percents = {
    2: [],
    4: [],
    8: [],
    12: [],
    16: [],
    20: [],
    30: [],
}
background_images = []

for file in files:
    with open(file, "r") as f:
        data = load(f)
        if data["ispano"]:
            continue

        objects = data["objects"]
        num_signs = len(objects)
        if num_signs == 0:
            background_images.append(file.stem)
            continue
        counter = 0

        for sign in objects:
            name = sign["label"]
            if name != "other-sign":
                counter += 1
        
        minority_sign_percent = counter / num_signs
        for bound, images in minority_sign_percents.items():
            if minority_sign_percent >= bound:
                images.append(file.stem)
        
        majority_sign_percent = num_signs / counter if counter != 0 else num_signs
        for bound, images in majority_sign_percents.items():
            if majority_sign_percent >= bound:
                images.append(file.stem)

output_data = {
    "minority_class_bounds": {
        str(bound): images for bound, images in minority_sign_percents.items()
    },
    "majority_class_bounds": {
        str(bound): images for bound, images in majority_sign_percents.items()
    },
    "background_images": background_images
}

with open("class_information.json", "w") as f:
    dump(output_data, f, indent=2)

with open("class_data.txt", "w") as f:
    f.write("Minority Sign Bounds\n")
    for bound, images in minority_sign_percents.items():
        f.write(f"Bound: {bound}, Number of Images: {len(images)}\n")

    f.write("\nMajority Sign Bounds\n")
    for bound, images in majority_sign_percents.items():
        f.write(f"Bound: {bound}, Number of Images: {len(images)}\n")

    f.write(f"\nNumber of background images: {len(background_images)}\n")