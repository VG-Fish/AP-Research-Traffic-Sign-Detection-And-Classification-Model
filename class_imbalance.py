from pathlib import Path
from json import load
from pprint import pprint

directory = "mtsd_v2_fully_annotated/annotations"
files = Path(directory).glob("*.json")
classes = {}

for file in files:
    with open(file, "r") as f:
        data = load(f)
        for sign in data["objects"]:
            name = sign["label"]
            num = classes.get(name, 0)
            classes[name] = num + 1

nc = sorted([(val, key) for key, val in classes.items()], reverse=True)
with open("num_classes.txt", "w") as f:
    for c in nc:
        f.write(f"Class: {c[1]}, Amount: {c[0]}\n")