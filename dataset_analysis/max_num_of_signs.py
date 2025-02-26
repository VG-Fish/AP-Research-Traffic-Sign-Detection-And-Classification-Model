from pathlib import Path
from json import load

directory = "mtsd_v2_fully_annotated/annotations"
files = Path(directory).glob("*.json")
max_num = 0

for file in files:
    with open(file, "r") as f:
        data = load(f)
        max_num = max(max_num, len(data["objects"]))

print(max_num) # 73