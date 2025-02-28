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

model = YOLO("models/yolo11n.pt")

def clear_cache(_):
    torch.mps.empty_cache()

model.add_callback("on_train_batch_start", clear_cache)
model.add_callback("on_val_batch_start", clear_cache)

results = model.train(
    # Train Variables
    data="balanced_dataset/balanced_augmented_mapillary.yaml",
    project="train",
    name="balanced_pre_augmented",
    epochs=100  ,
    device="mps",
    patience=15,
    batch=32,
    save_period=1,
    imgsz=640,
    exist_ok=True,
    optimizer="AdamW",
    amp=True, # mixed precision training
    plots=True,
    max_det=73, # The max number of annotations for an image is 73
    show_boxes=True,
    cos_lr=True, # Learning rate oscillates for better convergence
    save_json=True,
    augment=True,
    seed=16,
    conf=0.01,
    iou=0.6,
    rect=True,
    multi_scale=True,

    # Augmentation Variables
    # I'm disabling these following parameters as we already did image augmentation
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    
    # I'm enabling these following parameters
    degrees=22.5,
    translate=0.1,
    scale=0.5,
    shear=5,
    perspective=0.0005,
    mosaic=1.0,
    mixup=0.5,
    copy_paste=0.5,
    copy_paste_mode="mixup",
    erasing=0.4,
    crop_fraction=1.0,
)