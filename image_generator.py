from diffusers import StableDiffusionPipeline
import torch
import os

print("Loading AI Model... Please wait.")

pipe = StableDiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5",
torch_dtype=torch.float32
)

pipe = pipe.to("cpu")

prompt = input("Enter prompt: ")

print("Generating image...")

image = pipe(
prompt,
num_inference_steps=20
).images[0]

os.makedirs("outputs", exist_ok=True)

image.save("outputs/result.png")

print("Image saved to outputs/result.png")

output_path = os.path.abspath("outputs/result.png")
print(output_path)

if os.path.exists(output_path):
    os.startfile(output_path)
else:
    print("File not found:", output_path)
