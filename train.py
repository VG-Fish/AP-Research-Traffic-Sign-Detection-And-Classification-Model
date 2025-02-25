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

model = YOLO("yolo11n.pt")

def clear_cache(_):
    torch.mps.empty_cache()

model.add_callback("on_train_batch_start", clear_cache)
model.add_callback("on_val_batch_start", clear_cache)


results = model.train(
    data="mapillary.yaml",
    project="train",
    name="small_objects",
    epochs=1,
    val=False,
    device="mps",
    patience=3,
    batch=48,
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
)