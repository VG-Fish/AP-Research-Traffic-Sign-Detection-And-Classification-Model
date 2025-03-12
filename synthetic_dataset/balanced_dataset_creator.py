from json import load
from thread_safe_counter import ThreadSafeCounter
from collections import Counter
from pprint import pprint as pp
from math import floor, ceil
import cv2
from os import makedirs
from os.path import exists
from random import sample
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import albumentations as A

PATH_DIRECTORY = "mtsd_v2_fully_annotated"
DATASET_DIRECTORY = "mapillary_dataset"
SAVE_DIRECTORY = "balanced_mapillary_dataset"
MAX_AMOUNT = 100
TRAIN_FRAC = 0.8
CROP_FRAC = 0.1
class_amount = Counter()
AMOUNT_OF_CLASSES = 401

IMAGE_TRANSFORM = A.Compose([
    A.MotionBlur(p=0.01), # Simulating realistic camera conditions while driving
    A.RandomToneCurve(p=0.01), # Switches night to day
    A.OneOf([
        # Standard image augmentations
        A.RandomBrightnessContrast(p=0.7),
        A.RandomGamma(),
    ], p=0.5),
    A.SomeOf([
        # More simulation of realistic camera conditions while driving
        A.OpticalDistortion(),
        A.AdditiveNoise("gaussian"),
        A.RandomShadow(p=0.6),
        A.AutoContrast(method="pil"),
        A.OneOf([
            A.Illumination("corner"),
            A.Illumination("gaussian"),
            A.Illumination("linear"),
        ])
    ], n=2, p=0.1),
    A.OneOf([
        # Random weather conditions
        A.RandomFog(),
        A.RandomSunFlare(),
        A.RandomRain(),
    ], p=0.05),
    A.OneOf([
        # These transformations tries to get the model to focus less on color and more on shape
        A.HueSaturationValue(),
        A.ChannelShuffle(),
    ], p=0.075),
])
NUM_AUGMENTATIONS = 4

def resize_image(image, *, width, height, interpolation):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    resized = cv2.resize(image, dim, interpolation=interpolation)
    return resized

def parse_file(args) -> None:
    (directory, path, amount) = args

    if len(class_amount) == AMOUNT_OF_CLASSES and all([v == MAX_AMOUNT for v in class_amount.values()]):
        return

    with open(f"{PATH_DIRECTORY}/annotations/{path}.json") as f:
        annotations = load(f)

    """
    Ignore panoramas, they cause model performance drops (I believe).
    Also ignore background images, but also add them to a set to insert them into the dataset later.
    """
    if annotations["ispano"]:
        return

    objects = annotations["objects"]
    if all([class_amount[o["label"]] == MAX_AMOUNT for o in objects]):
        print(f"Skipping {path}.json\n")
        return
    
    image = cv2.imread(f"{DATASET_DIRECTORY}/{directory}/images/{path}.jpg")
    for idx, anno in enumerate(objects):
        name = anno["label"]

        bbox = anno["bbox"]
        x_min = floor(bbox["xmin"])
        x_max = ceil(bbox["xmax"])
        y_min = floor(bbox["ymin"])
        y_max = ceil(bbox["ymax"])

        if class_amount[name] < amount:
            class_amount[name] += 1
        else:
            image[y_min:y_max, x_min:x_max] = 0
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i in range(NUM_AUGMENTATIONS):
        print(f"Augmenting {path}.json")
        augmented = IMAGE_TRANSFORM(image=image)['image']
    # cv2.imwrite(f"s-{i}.jpg", resize_image(image, width=2048, height=1080, interpolation=cv2.INTER_AREA))
    print()

def parse_files(directory: str) -> None:
    amount = MAX_AMOUNT * 0.8 if directory == "train" else MAX_AMOUNT
    amount = ceil(amount)

    with open(f"{PATH_DIRECTORY}/splits/{directory}.txt") as f:
        paths = f.read().splitlines()
    paths = sample(paths, k=len(paths))

    # args_list = [
    #     (directory, path, amount)
    #     for path in paths
    # ]
    
    for path in paths:
        print(f"Parsing {path}.")
        parse_file((directory, path, amount))
    # with Pool(processes=cpu_count()) as pool:
    #     list(tqdm(pool.imap(parse_file, args_list), total=len(args_list)))
        
def make_dataset() -> None:
    for directory in ["train", "val"]:
        if not exists(f"{SAVE_DIRECTORY}/{directory}"):
            makedirs(f"{SAVE_DIRECTORY}/{directory}/images")
            makedirs(f"{SAVE_DIRECTORY}/{directory}/labels")

def main() -> None:
    make_dataset()
    print("Created the directories.")

    parse_files("train")
    print("Finished creating the train subdirectory.")

    parse_files("val")
    print("Finished creating the val subdirectory.")

    pp(class_amount)

if __name__ == "__main__":
    main()
