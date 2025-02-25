from pathlib import Path
from json import load, dump
import pprint as pp

directory = "mtsd_v2_fully_annotated/annotations"
files = Path(directory).glob("*.json")
minority_sign_percents = {
    0.1: [],
    1.00: [],
}
background_images = []

for file in files:
    with open(file, "r") as f:
        data = load(f)
        objects = data["objects"]
        num_signs = len(objects)
        if num_signs == 0:
            background_images.append(str(file))
            continue
        counter = 0

        for sign in objects:
            name = sign["label"]
            if name != "other-sign":
                counter += 1
        
        minority_sign_percent = counter / num_signs
        for bound, images in minority_sign_percents.items():
            if minority_sign_percent >= bound:
                images.append(str(file))

output_data = {
    "minority_class_bounds": {str(bound): images for bound, images in minority_sign_percents.items()},
    "background_images": background_images
}

# Write the entire dictionary as JSON to the file
with open("minority_class_bounds.json", "w") as f:
    dump(output_data, f, indent=2)

# Print summary information
for bound, images in minority_sign_percents.items():
    print(f"Bound: {bound}, Number of Images: {len(images)}")
print(f"Number of background images: {len(background_images)}")