# The plan to create an accurate, precise, and robust model.

## 1. Iterate through the dataset and choose/crop X random images for each class.
- Choose full resolution images until crops become necessary.
    - Keep track of the amount of instances per class. If adding a new image overflows the amount of instances per class, try a new image.
    - If the amount of retries exceeds an amount, crop the sign.
- The crops can be of varying size.
- If less than X images exist for the class, use the existing images and data augmentation to create new images.

## 2. Setup the Google VM environment and start training and fine-tuning the StarSRGAN model.
- The high-resolution images are the cropped/ traffic sign images.
- Download the model weights when completed.
- The fine-tuned model will be called as TS-StarSRGAN.

## 3. Create synthetic images using the cropped images and background images.
- Synthetic images will be chosen based on similarity using fastdup's image distance calculation.
- The images will be blended together, too.
- Multiple traffic signs will be added to the image, but the amount per image is based on the probability from the Mapillary dataset.
- The images _may_ be processed through TS-StarSRGAN.

## 4. Train the YOLO11(n/m) and YOLOv12(n/m) on the synthetic dataset.

## 5. Profit.
