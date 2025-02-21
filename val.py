from ultralytics import YOLO
import torch

model = YOLO("train/benchmark_0-9/weights/best.pt")

def clear_cache(_):
    torch.mps.empty_cache()

model.add_callback("on_val_batch_start", clear_cache)

metrics = model.val()

print(metrics)
