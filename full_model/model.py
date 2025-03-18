from ultralytics import YOLO

model = YOLO("train/rare_balanced_augmented_640-4/weights/best.pt")

metrics = model.val(
    batch=48,
    max_det=73,
    plots=True,
    save_json=True,
    name="base_model",
    project="train",
    save_crop=True,
    exist_ok=True,
    device="mps",
)