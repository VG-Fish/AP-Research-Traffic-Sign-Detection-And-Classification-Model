from pathlib import Path
from json import load
from os import makedirs, remove
from os.path import exists
from bisect import bisect_left

directory: str = "mtsd_v2_fully_annotated/annotations" # the directory containing all of the image annotations in JSON format.
classes = dict()
files = Path(directory).glob("*.json")

train_data = []
with open(f"mtsd_v2_fully_annotated/splits/train.txt", "r") as f:
    train_data = sorted(f.readlines())

val_data = []
with open(f"mtsd_v2_fully_annotated/splits/val.txt", "r") as f:
    val_data = sorted(f.readlines())

# The YOLO data format directory.
makedirs("mapillary_dataset/train/labels", exist_ok=True)
makedirs("mapillary_dataset/val/labels", exist_ok=True)

for file in files:
    data = dict()

    # Get the name of the traffic sign class
    with open(file, "r") as f:
        data = load(f)
        # Check to see if the labels are present
        try: 
            data["objects"][0]["label"]
        except:
            continue
    
    # Ignore panoramas.
    if data["ispano"]:
        continue

    image_annotations = []
    # Iterate through all the objects
    for bounding_box_object in data["objects"]:
        traffic_sign_class = bounding_box_object["label"]
        if traffic_sign_class not in classes:
            classes[traffic_sign_class] = len(classes)
            with open("classes.txt", "a") as f:
                f.write(f"\t{len(classes)}: {traffic_sign_class}\n")
        
        # Get all the coordinate data and normalize
        image_width = data["width"]
        image_height = data["height"]
        bounding_box_coords = bounding_box_object["bbox"]

        x_center = (bounding_box_coords["xmax"] + bounding_box_coords["xmin"]) / 2 / image_width
        y_center = (bounding_box_coords["ymax"] + bounding_box_coords["ymin"]) / 2 / image_height
        bounding_box_width = (bounding_box_coords["xmax"] - bounding_box_coords["xmin"]) / image_width
        bounding_box_height = (bounding_box_coords["ymax"] - bounding_box_coords["ymin"]) / image_height
        
        image_annotations.append(f"{classes[traffic_sign_class]} {x_center} {y_center} {bounding_box_width} {bounding_box_height}")
        
    file_name = file.stem
    file_name_for_data = f"{file_name}\n"
    subfolder = ""
    if (index := bisect_left(train_data, file_name_for_data)) != len(train_data) and train_data[index] == file_name_for_data:
        subfolder = "train"
    elif (index := bisect_left(val_data, file_name_for_data)) != len(val_data) and val_data[index] == file_name_for_data:
        subfolder = "val"
    else:
        # Doesn't exist in [train, val]_data, so we won't use this file while training.
        continue
    
    # Write all annotations to file at once in write mode, not append
    if subfolder and image_annotations:
        with open(f"mapillary_dataset/{subfolder}/labels/{file_name}.txt", "w") as f:
            f.write("\n".join(image_annotations))