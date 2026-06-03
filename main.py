from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

pipe=StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

prompt=input("Enter the prompt for Generating the image : ")
with torch.autocast("cuda"):
  image=pipe(prompt).images[0]

plt.imshow(image)
plt.axis("off")
plt.show()

image.save("results/generated_image.png")