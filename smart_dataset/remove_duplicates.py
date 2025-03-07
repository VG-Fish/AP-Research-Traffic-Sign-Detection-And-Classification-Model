from os import remove
from os.path import exists
from json import load
from typing import Dict

with open("smart_dataset/duplicates.json") as f:
    duplicates: Dict = load(f)

images_removed = 0
for duplicate_list in duplicates.values():
    if len(duplicate_list) <= 1:
        continue

    for image_path in duplicate_list[1:]:
        if exists(image_path):
            remove(image_path)
        
        label_path = f"{image_path.split('.')[0]}.txt"
        if exists(label_path):
            remove(label_path)
        images_removed += 1

print(images_removed)
