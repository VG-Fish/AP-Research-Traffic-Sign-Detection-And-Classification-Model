from json import load
from collections import Counter
from pprint import pprint as pp
from math import floor, ceil
import cv2

PATH_DIRECTORY = "mtsd_v2_fully_annotated"
DATASET_DIRECTORY = "mapillary_dataset"
MAX_AMOUNT = 50
MAX_RETRIES = 3
class_amount = Counter()
background_images = set()

def transform_image(path: str, annotations):
    directory, file = path.split("/")
    image = cv2.imread(f"{DATASET_DIRECTORY}/{directory}/images/{file}")

    print(file)
    pp(annotations)
    for anno in annotations:
        if not anno[2]:
            print("removed")
            bbox = anno[1]
            image[bbox["ymin"]:bbox["ymax"], bbox["xmin"]:bbox["xmax"]] = 0
    
    cv2.imwrite("debug.jpg", image)

def parse_files(file: str) -> None:
    amount = MAX_AMOUNT * 0.8 if file == "train" else MAX_AMOUNT

    with open(f"{PATH_DIRECTORY}/splits/{file}.txt") as f:
        paths = f.read().splitlines()

    counter = 0
    for path in paths:
        with open(f"{PATH_DIRECTORY}/annotations/{path}.json") as f:
            annotations = load(f)

        """
        Ignore panoramas, they cause model performance drops (I believe).
        Also ignore background images, but also add them to a set to insert them into the dataset later.
        """
        if annotations["ispano"]:
            continue
        elif len(annotations["objects"]) == 0:
            background_images.add(f"{file}/{path}.jpg")
            continue

        annotation_ids = Counter(map(lambda x: x["label"], annotations["objects"]))
        class_amount.update(annotation_ids)
        if max(class_amount.values()) > amount:
            class_amount.subtract(annotation_ids)
            cleaned_annotations = list(
                map(
                    (
                        lambda x: (
                            x["label"],
                            {k: ceil(v) if "max" in k else floor(v) for k, v in x["bbox"].items()},
                        )
                        + ((False,) if class_amount[x["label"]] >= amount else (True,))
                    ),
                    annotations["objects"],
                ),
            )

            if len(cleaned_annotations) == 0:
                continue
            
            counter += 1
            if counter == 50:
                transform_image(f"{file}/{path}.jpg", cleaned_annotations)
                break

def main() -> None:
    parse_files("train")
    # parse_files("val")

if __name__ == "__main__":
    main()
