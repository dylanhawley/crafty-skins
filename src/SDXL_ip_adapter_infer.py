from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained("monadical-labs/minecraft-skin-generator-sdxl", torch_dtype=torch.float16).to("mps")
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
pipeline.set_ip_adapter_scale(0)

image = load_image("bald.jpg")
generator = torch.Generator(device="mps").manual_seed(26)

image = pipeline(
    prompt="wearing a superman outfit",
    ip_adapter_image=image,
    num_inference_steps=30,
    generator=generator,
).images[0]

# Save the generated image
image.save("out.png")