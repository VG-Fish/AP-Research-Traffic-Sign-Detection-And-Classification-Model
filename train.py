from ultralytics import YOLO
import torch

"""
To disable sleep:
sudo pmset -b sleep 0
sudo pmset -b disablesleep 1

To enable sleep:
sudo pmset -b sleep 5
sudo pmset -b disablesleep 0

To purge RAM memory:
sudo purge

To disable CPU throttling:
sudo pmset -a lidwake 0
sudo pmset -a disablesleep 1

To keep the macbook awake:
caffeinate -i -d -m -u -t VALUE

To get GPU usage:
sudo powermetrics --samplers gpu_power

To get memory usage:
vm_stat

To increase open file limit:
ulimit -n 100000
"""

model = YOLO(f"yolo11n.pt")

def clear_cache(trainer):
    trainer._clear_memory()
model.add_callback("on_train_batch_start", clear_cache)

# Second training
results = model.train(
    # resume=True,
    data="mapillary.yaml",
    name="train3",
    epochs=2, 
    patience=3,
    batch=-1,
    save_period=1,
    imgsz=896,
    project="train",
    exist_ok=True,
    optimizer="AdamW",
    device="mps",
    amp=True, # mixed precision training
    single_cls=True,
    freeze=10, # freeze the backbone
    plots=True,
    rect=True,
    conf=0.25, # the confidence threshold
    max_det=73, # The maximum number of annotations for an image was 73
    show_boxes=True,
    multi_scale=True,
    fraction=0.1,
    dropout=0.001, # due to training on a smaller dataset
    deterministic=False,
    cos_lr=True,
    workers=16,
    save_json=True,

    # Augmentation variables
    # copy_paste=0.3,
    # box=10,
    # mixup=0.3,
    # mosaic=1,
)

val_data = model.val()
with open("train/train3/val_results.txt", "w") as f:
    f.write(val_data)