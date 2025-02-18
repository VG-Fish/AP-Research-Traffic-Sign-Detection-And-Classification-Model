import requests
from PIL import Image
from io import BytesIO
from diffusers import LDMSuperResolutionPipeline

device: str = "mps"
model_id: str = "CompVis/ldm-super-resolution-4x-openimages"

# load model and scheduler
pipeline: LDMSuperResolutionPipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
pipeline: LDMSuperResolutionPipeline = pipeline.to(device)

# let's download an  image
url: str = "https://user-images.githubusercontent.com/38061659/199705896-b48e17b8-b231-47cd-a270-4ffa5a93fa3e.png"
response: requests.Response = requests.get(url)
low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
low_res_img = low_res_img.resize((128, 128))

# run pipeline in inference (sample random noise and denoise)
upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
# save image
upscaled_image.save("ldm_generated_image.png")