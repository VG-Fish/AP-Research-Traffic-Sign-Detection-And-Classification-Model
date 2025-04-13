from json import load
from os import listdir

ORIGIN_DIRECTORY = "mtsd_v2_fully_annotated/annotations"

for file in listdir(ORIGIN_DIRECTORY):
    with open(f"{ORIGIN_DIRECTORY}/{file}") as f:
        data = load(f)

    if data["ispano"]:
        continue

    objects = data["objects"]
    total = len(objects)

    positions = set()
    for o in objects:
        bbox = o["bbox"]
        positions.add(tuple(bbox.values()))
    
    if len(positions) != total:
        print(file)