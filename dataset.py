from ultralytics import YOLO
import ultralytics.data.build as build
from weighted_dataset import YOLOWeightedDataset
import torch

build.YOLODataset = YOLOWeightedDataset

model = YOLO("yolo11n.pt")

def clear_cache(_):
    torch.mps.empty_cache()

model.add_callback("on_train_batch_start", clear_cache)
model.add_callback("on_val_batch_start", clear_cache)

results = model.train(
    data="mapillary.yaml",
    project="train",
    name="dataset_test",
    fraction=0.001,
    epochs=1,
    device="mps",
    patience=3,
    batch=64,
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
    val=False,
)

print(results)