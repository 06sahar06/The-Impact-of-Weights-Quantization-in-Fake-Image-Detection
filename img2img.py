import argparse
import os
import glob
import random
import torch
import gc
from PIL import Image

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    StableDiffusion3Pipeline
)

from diffusers.quantizers import PipelineQuantizationConfig

# --- Configuration & Constants ---
MODELS = {
    'sd15': 'runwayml/stable-diffusion-v1-5',
    'sd3': 'stabilityai/stable-diffusion-3-medium-diffusers',
    'sd35': 'stabilityai/stable-diffusion-3.5-medium',
}

QUANTIZATION_LEVELS = {
    'fp16': None,
    'fp8': PipelineQuantizationConfig(
        quant_backend="bitsandbytes_8bit",
        quant_kwargs={"load_in_8bit": True},
        components_to_quantize=["text_encoder_3", "transformer"]
    ),
    'fp4': PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16
        }
    )
}

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

class ImageGenerator:
    def __init__(self, model_key, quantization, device=None):
        self.model_key = model_key
        self.quantization = quantization
        
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")

        self.pipeline = None
        self.load_pipeline()

    def load_pipeline(self):
        print(f"Loading {self.model_key} with {self.quantization} quantization...")
        
        model_id = MODELS[self.model_key]
        q_config = QUANTIZATION_LEVELS[self.quantization]
        
        # Determine Dtype
        torch_dtype = None
        if self.quantization == 'fp16':
            if self.device == 'cpu':
                torch_dtype = torch.float32 
            else:
                torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        try:
            pipeline_t2i = None
                
            # --- LOGIC: SD 3/3.5 FAMILY ---
            if 'sd3' in self.model_key:
                 if self.quantization == 'fp16':
                    pipeline_t2i = StableDiffusion3Pipeline.from_pretrained(
                        model_id, torch_dtype=torch.float16
                    )
                 else:
                    pipeline_t2i = AutoPipelineForText2Image.from_pretrained(
                        model_id, quantization_config=q_config, torch_dtype=torch_dtype
                    )
                 # SD3 usually requires CPU offload to fit in consumer VRAM
                 pipeline_t2i.enable_model_cpu_offload()

            # --- LOGIC: SD1.5 ---
            else: 
                if self.quantization == 'fp16':
                    pipeline_t2i = AutoPipelineForText2Image.from_pretrained(
                        model_id, 
                        torch_dtype=torch_dtype, 
                        use_safetensors=True
                    ).to(self.device)
                else:
                    pipeline_t2i = AutoPipelineForText2Image.from_pretrained(
                        model_id, 
                        quantization_config=q_config, 
                        torch_dtype=torch_dtype,
                        use_safetensors=True
                    ).to(self.device)
                    pipeline_t2i.enable_model_cpu_offload()

            # Convert to Image2Image pipeline
            if self.pipeline is None:
                if pipeline_t2i is not None:
                    self.pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_t2i)
                else:
                    raise ValueError("Failed to initialize a pipeline to convert to Img2Img.")
            
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise e

    def generate(self, prompt, image, strength=0.3, guidance_scale=3.5, steps=30, seed=None):
        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(seed)
        else:
             rand_seed = random.randint(0, 2**32 - 1)
             generator = torch.Generator("cpu").manual_seed(rand_seed)

        kwargs = {
            "prompt": prompt,
            "image": image,
            "strength": strength,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator 
        }
        
        return self.pipeline(**kwargs).images[0]

def main():
    parser = argparse.ArgumentParser(description="Generate images using various Stable Diffusion models (Image-to-Image only).")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation.")
    parser.add_argument("--models", nargs='+', default=['sd15'], choices=MODELS.keys(), help="Models to use.")
    parser.add_argument("--quantization", nargs='+', default=['fp16'], choices=QUANTIZATION_LEVELS.keys(), help="Quantization levels.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, default="output_img2img", help="Directory for output images.")
    parser.add_argument("--strength", type=float, default=0.3, help="Denoising strength (0.0 to 1.0).")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps.")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale.")
    parser.add_argument("--device", type=str, default=None, help="Device to use.")
    parser.add_argument("--seed", type=int, default=123, help="Fixed seed for reproducibility.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Input directory {args.input_dir} does not exist.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    input_images = sorted(glob.glob(os.path.join(args.input_dir, '*.jpg')) + 
                          glob.glob(os.path.join(args.input_dir, '*.png')) +
                          glob.glob(os.path.join(args.input_dir, '*.PNG')) +
                          glob.glob(os.path.join(args.input_dir, '*.JPG')) +
                          glob.glob(os.path.join(args.input_dir, '*.jpeg')))

    if not input_images:
        print(f"No images found in {args.input_dir}")
        return

    for model_key in args.models:
        for quant in args.quantization:
            try:
                generator = ImageGenerator(model_key, quant, device=args.device)
                
                print(f"Processing images with {model_key} ({quant})...")
                
                for i, img_path in enumerate(input_images):
                    try:
                        print(f"  Processing {os.path.basename(img_path)}...")
                        
                        input_img = Image.open(img_path).convert("RGB")
                        width, height = input_img.size

                        # Center crop 1024x1024
                        if width > 1024 and height > 1024:
                            left = (width - 1024)/2
                            top = (height - 1024)/2
                            right = (width + 1024)/2
                            bottom = (height + 1024)/2
                            input_img = input_img.crop((left, top, right, bottom))
                        
                        current_seed = args.seed + i if args.seed is not None else None
                        
                        output_img = generator.generate(
                            args.prompt, 
                            image=input_img,
                            strength=args.strength,
                            steps=args.steps, 
                            guidance_scale=args.guidance, 
                            seed=current_seed
                        )
                        
                        base_name = os.path.splitext(os.path.basename(img_path))[0]
                        seed_str = f"_seed{current_seed}" if current_seed is not None else ""
                        out_name = f"{args.output_dir}/{base_name}_{model_key}_{quant}{seed_str}.png"
                        output_img.save(out_name)
                        print(f"    Saved {out_name}")
                        
                    except Exception as e_img:
                        print(f"    Failed to process {os.path.basename(img_path)}: {e_img}")

                flush()
                del generator
                torch.cuda.empty_cache()
                flush()
                
            except Exception as e:
                print(f"Failed to load/run {model_key} with {quant}: {e}")
                flush()

if __name__ == "__main__":
    main()