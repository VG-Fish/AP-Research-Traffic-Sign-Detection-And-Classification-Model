from diffusers import LDMSuperResolutionPipeline
from cv2 import resize, cvtColor, COLOR_BGR2RGB, INTER_CUBIC, imread
from PIL import Image

device = "mps"
model_id = "CompVis/ldm-super-resolution-4x-openimages"

pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
pipeline = pipeline.to(device)

def resize_image(image, *, width, height, interpolation):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    resized = resize(image, dim, interpolation=interpolation)
    return resized

def create_image(image: str, output_path: str) -> None:
    image = cvtColor(imread(image), COLOR_BGR2RGB)
    low_res_img = resize_image(image, width=128, height=128, interpolation=INTER_CUBIC)
    low_res_img = Image.fromarray(low_res_img)

    upscaled_image = pipeline(low_res_img, num_inference_steps=10, eta=1).images[0]
    upscaled_image.save(output_path)
