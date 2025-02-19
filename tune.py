from ultralytics import YOLO

model = YOLO("yolo11m.pt")

results = model.tune(
    data="mapillary.yaml",
    epochs=10,
    iterations=100,
    optimizer="AdamW",
    save=True,
    plots=True,
    project="train_tuning",
    name="tuning",
    val=False,
    device="mps",
    exist_ok=True,
    save_period=1,
    fraction=0.1,
    batch=16,
    patience=3,
    imgsz=640,
    amp=True,
    show_boxes=True,
    seed=16,
    cos_lr=True,
    conf=0.25,
    workers=16,
)

with open("train/tuning/results.txt", "w") as f:
    f.write(results)
