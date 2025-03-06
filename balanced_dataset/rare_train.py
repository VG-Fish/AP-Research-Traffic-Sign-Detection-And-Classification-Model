from ultralytics import YOLO
import torch

pose_model = YOLO("yolo11n-pose.pt")

# Converting pose model to detect
detect_rare_model = YOLO("train/rare_balanced_augmented_640-4/weights/best.pt").load(pose_model.model)
detect_rare_model.ckpt = {"model": detect_rare_model.model}
detect_rare_model.save("models/detect_rare_model.pt")

# Clearing memory 
def clear_cache(_):
    torch.mps.empty_cache()

detect_rare_model.add_callback("on_train_batch_start", clear_cache)
detect_rare_model.add_callback("on_val_batch_start", clear_cache)

detect_rare_model.train(
    # Train Variables
    data="balanced_dataset/small_object.yaml",
    project="train",
    name=f"detect_rare_model",
    epochs=20,
    device="mps",
    patience=15,
    batch=40,
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
    rect=True,
    lr0=0.001,
    save_conf=True,
    save_crop=True,
    show_boxes=True,
    warmup_epochs=2, 
    close_mosaic=2,

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