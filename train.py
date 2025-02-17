from ultralytics import YOLO

model = YOLO(f"yolo11m.pt")

"""
To disable sleep for Macbooks:
sudo pmset -b sleep 0
sudo pmset -b disablesleep 1

To enable sleep:
sudo pmset -b sleep 5
sudo pmset -b disablesleep 0
"""


# Second training
results = model.train(
    # resume=True,
    data="mapillary.yaml",
    name="train3",
    epochs=5, 
    patience=3,
    batch=32,
    save_period=1,
    imgsz=1024,
    cache="ram",
    project="train",
    exist_ok=True,
    optimizer="AdamW",
    device="mps",
    amp=True,
    single_cls=True,
    freeze=10, # freeze the backbone
    plots=True,
    rect=True,
    conf=0.25, # the confidence threshold
    max_det=73, # The maximum number of annotations (.txt file) for an image was 73
    save_txt=True,
    show_boxes=True,
    optimize=True,
    simplify=True,
    half=True,
    dynamic=True,
    multi_scale=True,
    fraction=0.1,
    dropout=0.001, # due to training on a smaller dataset
    deterministic=False,
    # cos_lr=True,
    lr0=0.001,
    cls=1,

    # Augmentation variables
    copy_paste=0.3,
    box=10,
    mixup=0.3,
    mosaic=1,
)

val_data = model.val()
with open("train/train3/val_results.txt", "w") as f:
    f.write(val_data)