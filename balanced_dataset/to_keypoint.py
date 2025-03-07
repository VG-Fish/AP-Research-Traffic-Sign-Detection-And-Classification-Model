from pathlib import Path
from os import makedirs
from os.path import exists

DIRECTORIES = [
    "rare_balanced_augmented_mapillary_dataset/train-augmented/labels",
    "rare_balanced_augmented_mapillary_dataset/val/labels"
]

for directory in DIRECTORIES:
    files = Path(directory).glob("*.txt")

    if not exists(f"{directory}-pose"):
        makedirs(f"{directory}-pose")
    
    for file in files:
        new_annotations = []
        new_pose_annotations = []
        with open(file) as f:
            annotations = f.readlines()
            for annotation in annotations:
                # .rstrip() to remove newlines
                annotation = annotation.rstrip().split(" ")
                traffic_sign_class = annotation[0]
                bounding_box = annotation[1:5]

                visibility = 1.0
                # For the keypoint loss
                x_keypoint, y_keypoint = bounding_box[:2]
                new_annotations.append(f"{traffic_sign_class} {' '.join(bounding_box)}")
                new_pose_annotations.append(f"{traffic_sign_class} {' '.join(bounding_box)} {x_keypoint} {y_keypoint} {visibility}")
            
        with open(f"{directory}/{file.stem}.txt", "w") as f:
            f.write("\n".join(new_annotations))
        with open(f"{directory}-pose/{file.stem}.txt", "w") as f:
            f.write("\n".join(new_pose_annotations))
        