from ultralytics import YOLO
import torch

EPOCHS = 5

for i in range(3):
    # I moved the model instantiation in here to ensure it gets the latest model weights every time.
    core_model = YOLO(
        f"train/balanced_augmented_640/weights/best.pt" 
        if i == 0 else 
        f"train/balanced_augmented_640-{i}/weights/best.pt"
    )
    rare_model = YOLO(f"train/rare_balanced_augmented_640-{i}/weights/best.pt")

    def clear_cache(_):
        torch.mps.empty_cache()

    core_model.add_callback("on_train_batch_start", clear_cache)
    core_model.add_callback("on_val_batch_start", clear_cache)

    rare_model.add_callback("on_val_batch_start", clear_cache)
    rare_model.add_callback("on_val_batch_start", clear_cache)

    core_model.train(
        # Train Variables
        data="balanced_dataset/rare_balanced_augmented_mapillary.yaml",
        project="train",
        name=f"rare_balanced_augmented_640-{i}",
        epochs=EPOCHS,
        device="mps",
        patience=15,
        batch=32,
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
        conf=0.01,
        iou=0.5,
        rect=True,
        lr0=0.001,

        # Augmentation Variables
        # I'm disabling these following parameters as we already did image augmentation
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        
        # I'm enabling these following parameters
        degrees=22.5,
        translate=0.1,
        scale=0.5,
        shear=2.5,
        perspective=0.0002,
        mosaic=1.0,
        mixup=0.5,
        copy_paste=0.5,
        copy_paste_mode="mixup",
        erasing=0.4,
        crop_fraction=1.0,
    )

    rare_model.train(
        # Train Variables
        data="balanced_dataset/balanced_augmented_mapillary.yaml",
        project="train",
        name=f"balanced_augmented_640-{i}",
        epochs=EPOCHS,
        device="mps",
        patience=15,
        batch=32,
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
        conf=0.01,
        iou=0.5,
        rect=True,
        lr0=0.001,

        # Augmentation Variables
        # I'm disabling these following parameters as we already did image augmentation
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        
        # I'm enabling these following parameters
        degrees=22.5,
        translate=0.1,
        scale=0.5,
        shear=2.5,
        perspective=0.0002,
        mosaic=1.0,
        mixup=0.5,
        copy_paste=0.5,
        copy_paste_mode="mixup",
        erasing=0.4,
        crop_fraction=1.0,
    )