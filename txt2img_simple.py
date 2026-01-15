import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU before importing torch

import torch
import gc
from PIL import Image
from diffusers import StableDiffusion3Pipeline

# Force CPU execution
device = "cpu"
torch_dtype = torch.float32

print(f"Using device: {device}")
print(f"Using dtype: {torch_dtype}")

# Model and output settings
model_id = "stabilityai/stable-diffusion-3.5-medium"
output_dir = "output/txt2img"
prompt = "A futuristic cyberpunk city with neon lights"
seed = 123
output_file = os.path.join(output_dir, "sd35_fp16_seed123.png")

# Check if file already exists
if os.path.exists(output_file):
    print(f"Output file {output_file} already exists. Skipping generation.")
else:
    print("Loading SD3.5 pipeline...")
    
    # Load pipeline with minimal memory footprint
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )
    
    print("Moving pipeline to CPU...")
    pipeline = pipeline.to(device)
    
    print("Generating image...")
    generator = torch.Generator("cpu").manual_seed(seed)
    
    image = pipeline(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=3.5,
        generator=generator
    ).images[0]
    
    print(f"Saving image to {output_file}...")
    os.makedirs(output_dir, exist_ok=True)
    image.save(output_file)
    print(f"Image saved successfully!")
    
    # Cleanup
    del pipeline
    gc.collect()
