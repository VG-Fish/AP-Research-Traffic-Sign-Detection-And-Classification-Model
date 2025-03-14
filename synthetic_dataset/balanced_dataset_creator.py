from json import load
from thread_safe_counter import ThreadSafeCounter
from collections import Counter
from pprint import pprint as pp
from math import floor, ceil
import cv2
from os import makedirs
from os.path import exists
from random import shuffle
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import albumentations as A

PATH_DIRECTORY = "mtsd_v2_fully_annotated"
DATASET_DIRECTORY = "mapillary_dataset"
SAVE_DIRECTORY = "balanced_mapillary_dataset"
DATASET_INFO_JSON_FILE = "synthetic_dataset/rare_class_info.json"
DESIRED_TYPE = "1.0-minority"
MAX_AMOUNT = 200
TRAIN_FRAC = 0.8
class_amount = Counter()
times_till_last_update = ThreadSafeCounter()
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
    (directory, path, amount, times_till_last_update_bound, directory_class_amount) = args

    if times_till_last_update.value >= times_till_last_update_bound:
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
    if all([directory_class_amount.get(o["label"], 0) == MAX_AMOUNT for o in objects]):
        return
    
    image = cv2.imread(f"{DATASET_DIRECTORY}/{directory}/images/{path}.jpg")

    label_path = f"{DATASET_DIRECTORY}/{directory}/labels/{path}.txt"
    if not exists(label_path):
        print(f"{label_path} doesn't exist.")
        return
    
    with open(label_path) as f:
        labels = f.read().splitlines()

    new_labels = []
    for idx, anno in enumerate(objects, start=0):
        name = anno["label"]

        bbox = anno["bbox"]
        x_min = floor(bbox["xmin"])
        x_max = ceil(bbox["xmax"])
        y_min = floor(bbox["ymin"])
        y_max = ceil(bbox["ymax"])

        if directory_class_amount.setdefault(name, 0) < amount:
            directory_class_amount[name] += 1
            new_labels.append(labels[idx])
        else:
            image[y_min:y_max, x_min:x_max] = 0
    
    if len(new_labels) == 0:
        times_till_last_update.add(1)
        return
    
    image = resize_image(image, width=2048, height=1080, interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{SAVE_DIRECTORY}/{directory}/images/{path}.jpg", image)
    
    updated_labels = "\n".join(new_labels)
    with open(f"{SAVE_DIRECTORY}/{directory}/labels/{path}.txt", "w") as f:
        f.write(updated_labels)

    if directory == "val":
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = image.copy()
    for i in range(NUM_AUGMENTATIONS):
        augmented = IMAGE_TRANSFORM(image=augmented)['image']
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f"{SAVE_DIRECTORY}/{directory}/images/augmented_{i+1}-{path}.jpg", augmented)
        with open(f"{SAVE_DIRECTORY}/{directory}/labels/augmented_{i+1}-{path}.txt", "w") as f:
            f.write(updated_labels)

def parse_files(directory: str, times_till_last_update_bound: int) -> None:
    amount = MAX_AMOUNT / NUM_AUGMENTATIONS
    amount = amount * 0.8 if directory == "train" else amount * 0.2
    amount = ceil(amount)

    with open(f"{PATH_DIRECTORY}/splits/{directory}.txt") as f:
        paths = f.read().splitlines()
    
    shuffle(paths)

    with Manager() as manager:
        directory_class_amount = manager.dict()

        args_list = [(directory, path, amount, times_till_last_update_bound, directory_class_amount) for path in paths]
        with Pool(processes=cpu_count()) as pool:
            list(tqdm(pool.imap(parse_file, args_list, chunksize=4), total=len(args_list)))
        
        class_amount.update(dict(directory_class_amount))
        
def make_dataset() -> None:
    for directory in ["train", "val"]:
        makedirs(f"{SAVE_DIRECTORY}/{directory}/images", exist_ok=True)
        makedirs(f"{SAVE_DIRECTORY}/{directory}/labels", exist_ok=True)

def make_classes_equal(directory: str) -> None:
    with open(f"{DATASET_INFO_JSON_FILE}") as f:
        data = load(f)
    
    for traffic_sign_class, info in data.items():
        if class_amount.get(traffic_sign_class, 0) >= MAX_AMOUNT:
            continue

        paths = info[DESIRED_TYPE].get(directory, None)

        # Only skips classes when creating the val directory
        if paths is None:
            continue
        
        # Randomize the list
        shuffle(paths)

        amount = (MAX_AMOUNT - class_amount[traffic_sign_class]) // NUM_AUGMENTATIONS
        amount = amount * 0.8 if directory == "train" else amount * 0.2
        amount = ceil(amount)
        
        print(f"Equalizing {traffic_sign_class} by adding {amount} images.")
        with Manager() as manager:
            directory_class_amount = manager.dict()

            args_list = [(directory, path, amount, 1, directory_class_amount) for path in paths]
            with Pool(processes=cpu_count()) as pool:
                list(tqdm(pool.imap(parse_file, args_list, chunksize=4), total=len(args_list)))
            
            class_amount.update(dict(directory_class_amount))

def main() -> None:
    make_dataset()
    print("Created the directories.")

    parse_files("train", 1)
    print("Finished creating the train subdirectory.")

    times_till_last_update.set_val(0)

    parse_files("val", 1)
    print("Finished creating the val subdirectory.")

    print("Making all classes have the same amount of images...\n")
    times_till_last_update.set_val(0)

    make_classes_equal("train")
    print("Finished equalizing the train directory.\n")

    times_till_last_update.set_val(0)

    make_classes_equal("val")
    print("Finished equalizing the val directory.\n")

    pp(class_amount)

    print("Finished creating the dataset.")

if __name__ == "__main__":
    main()
