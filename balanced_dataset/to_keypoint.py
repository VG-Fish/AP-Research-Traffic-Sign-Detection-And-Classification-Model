from pathlib import Path

DIRECTORIES = [
    "rare_balanced_augmented_mapillary_dataset/train-augmented/labels",
    "rare_balanced_augmented_mapillary_dataset/val/labels"
]

for directory in DIRECTORIES:
    files = Path(directory).glob("*.txt")
    for file in files:
        new_annotations = []
        with open(file) as f:
            annotations = f.readlines()
            for annotation in annotations:
                # .rstrip() to remove newlines
                annotation = annotation.rstrip().split(" ")
                traffic_sign_class = annotation[0]
                bounding_box = annotation[1:]

                visibility = 1.0
                # For the keypoint loss
                x_keypoint, y_keypoint = bounding_box[:2]
                new_annotations \
                    .append(f"{traffic_sign_class} {' '.join(bounding_box)} {x_keypoint} {y_keypoint} {visibility}")
        with open(f"{directory}/{file.stem}-keypoint.txt", "w") as f:
            f.write("\n".join(new_annotations))
        