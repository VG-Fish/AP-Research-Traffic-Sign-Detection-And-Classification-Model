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

def clear_cache(_):
    torch.mps.empty_cache()

model.add_callback("on_train_batch_start", clear_cache)
model.add_callback("on_val_batch_start", clear_cache)

# Second training
results = model.train(
    # resume=True,
    data="mapillary.yaml",
    name="benchmark",
    epochs=3,
    patience=3,
    batch=32,
    save_period=1,
    imgsz=640,
    project="train",
    exist_ok=True,
    optimizer="AdamW",
    device="mps",
    amp=True, # mixed precision training
    single_cls=True,
    freeze=10, # freeze the backbone
    plots=True,
    max_det=73, # The maximum number of annotations for an image was 73
    show_boxes=True,
    # multi_scale=True,
    fraction=0.1,
    # dropout=0.01, # due to training on a smaller dataset
    # cos_lr=True,
    save_json=True,
    agnostic_nms=True, # due to having one class
    augment=True,
    conf=0.3,

    # Augmentation variables
    copy_paste=0.3,
    box=10,
    mixup=0.3,
    mosaic=1,
)
