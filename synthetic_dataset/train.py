from ultralytics import YOLO
import torch

pose_model = YOLO("models/yolo11n-pose.pt")

# Converting pose model to detect
model = YOLO("train/detect_rare_model_2/weights/best.pt").load(pose_model.model)
model.ckpt = {"model": model.model}

# Clearing memory 
def clear_cache(_):
    torch.mps.empty_cache()

model.add_callback("on_train_batch_start", clear_cache)
model.add_callback("on_val_batch_start", clear_cache)

results = model.train(
    # Train Variables
    data="synthetic_dataset/synthetic.yaml",
    project="train",
    name=f"synthetic",
    epochs=1,
    device="mps",
    patience=15,
    batch=48,
    workers=12,
    save_period=1,
    imgsz=640,
    exist_ok=True,
    optimizer="AdamW",
    amp=True, # mixed precision training
    plots=True,
    max_det=73, # The max number of annotations for an image is 73
    cos_lr=True, # Learning rate oscillates for better convergence
    save_json=True,
    seed=16,
    conf=0.01,
    iou=0.7,
    lr0=0.001,
    save_conf=True,
    save_crop=True,
    show_boxes=True,
    warmup_epochs=1, 
    close_mosaic=1,
    # To give more weight to the keypoint loss
    box=0.01,
    dfl=0.01,

    # Augmentation Variables
    # I'm disabling these following parameters as we already did image augmentation
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    flipud=0.0,
    fliplr=0.0,
    
    # I'm enabling these following parameters
    degrees=15,
    translate=0.1,
    scale=0.5,
    shear=2,
    perspective=0.0002,
    mosaic=1.0,
    mixup=1.0,
    copy_paste=0.5,
    copy_paste_mode="mixup",
    erasing=0.4,
    crop_fraction=1.0,
)