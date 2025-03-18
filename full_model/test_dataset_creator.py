import cv2
from json import load
from random import sample
from os.path import exists
from os import mkdir
from math import floor, ceil
from datetime import datetime

IMAGE_DIRECTORY = "balanced_mapillary_dataset/val/images"
CROP_JSON_FILE = "train/base_model/predictions.json"
OUTPUT_DIR = "cropped_dataset"
AMOUNT = 100

if not exists(OUTPUT_DIR):
    mkdir(OUTPUT_DIR)
    mkdir(f"{OUTPUT_DIR}/images/")

with open(CROP_JSON_FILE) as f:
    annotations = load(f)

confidence = 0
for annotation in annotations:
    confidence += float(annotation["score"])
confidence /= len(annotations)

with open(f"{OUTPUT_DIR}/avg_base_conf.txt", "w") as f:
    f.write(f"{confidence}\n")

for annotation in sample(annotations, k=AMOUNT):
    image_id = annotation["image_id"]
    image = cv2.imread(f"{IMAGE_DIRECTORY}/{image_id}.jpg")

    bbox = list(map(float, annotation["bbox"]))
    (x, y) = bbox[:2]
    (w, h) = bbox[2:4]
    x, y = floor(x), floor(y)
    w, h = ceil(w), ceil(h)
    
    cropped_image = image[y:y+h, x:x+w]
    cv2.imwrite(f"{OUTPUT_DIR}/images/{image_id}_{str(datetime.now())}.jpg", cropped_image)