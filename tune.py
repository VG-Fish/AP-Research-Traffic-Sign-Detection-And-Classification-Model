from ultralytics import YOLO
import torch

model = YOLO("yolo11m.pt")

def clear_cache(tuner):
    print(tuner)
    torch.mps.empty_cache()

model.add_callback("on_pretrain_routine_start", clear_cache)

results = model.tune(
    data="mapillary.yaml",
    epochs=1,
    iterations=1,
    optimizer="AdamW",
    save=True,
    plots=True,
    project="train_tuning",
    name="tuning",
    val=False,
    device="mps",
    exist_ok=True,
    save_period=1,
    fraction=0.5,
    batch=24,
    patience=3,
    imgsz=640,
    amp=True,
    show_boxes=True,
    seed=16,
)

with open("train/tuning/results.txt", "w") as f:
    f.write(results)