from ultralytics import YOLO

model = YOLO("train/rare_balanced_augmented_640-4/weights/best.pt")

metrics = model.val()
print(metrics)