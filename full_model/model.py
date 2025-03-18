from ultralytics import YOLO
from os import listdir
from random import sample
from SRR import create_image
import cv2

base_model = YOLO("train/rare_balanced_augmented_640-4/weights/best.pt")

IMAGE_DIRECTORY = "balanced_mapillary_dataset/val/images"
CROPPED_DIRECTORY = "cropped_dataset/images"
AMOUNT = 0

images = sample(listdir(IMAGE_DIRECTORY), k=AMOUNT)

for image in images:
    img = cv2.imread(f"{IMAGE_DIRECTORY}/{image}")

    results = base_model.predict(
        source=img,
        batch=48,
        max_det=73,
        plots=True,
        save_json=True,
        name="base_model",
        project="train",
        exist_ok=True,
        device="mps",
    )

    for result in results:
        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            cropped_img = img[y1:y2, x1:x2]
            cv2.imwrite(f"{CROPPED_DIRECTORY}/{i}-{image}", cropped_img)

            create_image(
                f"{CROPPED_DIRECTORY}/{i}-{image}", 
                output_path=f"{CROPPED_DIRECTORY}/enhanced_{i}-{image}"
            )


for enhanced_image in list(filter(lambda x: x.startswith("enhanced_"), listdir(CROPPED_DIRECTORY))):
    results = base_model.predict(
        source=f"{CROPPED_DIRECTORY}/{enhanced_image}",
        imgsz=640,
        batch=48,
        max_det=73,
        plots=True,
        save=True,
        save_json=True,
        project="train",
        name="base_enhanced_model",
        exist_ok=True,
        device="mps",
        save_conf=True
    )
