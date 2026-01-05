import argparse
import os
import glob
import random
import torch
import gc
import numpy as np
from PIL import Image

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    StableDiffusion3Pipeline,
    FluxPipeline
)

from diffusers.quantizers import PipelineQuantizationConfig

# --- Configuration & Constants ---
MODELS = {
    'sdxl': 'stabilityai/stable-diffusion-xl-base-1.0',
    # UPDATED: Replaced sd2 with sd3
    'sd3': 'stabilityai/stable-diffusion-3-medium-diffusers', 
    'sd35': 'stabilityai/stable-diffusion-3.5-medium',
    'flux': 'black-forest-labs/FLUX.1-dev',
    'sd15': 'runwayml/stable-diffusion-v1-5'
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
        self.pipeline_t2i = None
        
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
        
        # Determine Variant (mostly for SDXL/SD1.5 legacies)
        variant = "fp16" if self.quantization == 'fp16' and self.model_key not in ['flux', 'sd3', 'sd35'] and self.device != 'cpu' else None
        
        try:
            if self.model_key == 'flux':
                if self.quantization == 'fp16':
                     # Flux FP16 requires offload to fit in <32GB VRAM
                     self.pipeline = FluxPipeline.from_pretrained(
                         model_id, torch_dtype=torch.bfloat16
                     )
                     self.pipeline.enable_model_cpu_offload()
                else:
                     self.pipeline = FluxPipeline.from_pretrained(
                         model_id, quantization_config=q_config, torch_dtype=torch_dtype
                     )
                     self.pipeline.enable_model_cpu_offload()
                
            # UPDATED: Group SD3 and SD3.5 logic
            elif self.model_key in ['sd35', 'sd3']:
                 if self.quantization == 'fp16':
                    self.pipeline_t2i = StableDiffusion3Pipeline.from_pretrained(
                        model_id, torch_dtype=torch.float16
                    )
                    self.pipeline_t2i.enable_model_cpu_offload()
                 else:
                    # Specific logic: If user selected sd35 quantized, upgrade to Large model (optional logic from before)
                    # For normal sd3, we keep the model_id as is.
                    model_id_load = "stabilityai/stable-diffusion-3.5-large" if (self.model_key == 'sd35' and self.quantization != 'fp16') else model_id
                    
                    self.pipeline_t2i = AutoPipelineForText2Image.from_pretrained(
                        model_id_load, quantization_config=q_config, torch_dtype=torch.float16
                    )
                    self.pipeline_t2i.enable_model_cpu_offload()
            


            else: # SDXL
                if self.quantization == 'fp16':
                    self.pipeline_t2i = AutoPipelineForText2Image.from_pretrained(
                        model_id, torch_dtype=torch_dtype, variant=variant, use_safetensors=True
                    ).to(self.device)
                else:
                    self.pipeline_t2i = AutoPipelineForText2Image.from_pretrained(
                        model_id, quantization_config=q_config, torch_dtype=torch_dtype
                    ).to(self.device)
                self.pipeline_t2i.enable_model_cpu_offload()

            # Convert to Image2Image pipeline
            if self.pipeline is None:
                if self.model_key == 'flux':
                     self.pipeline = AutoPipelineForImage2Image.from_pipe(self.pipeline)
                else:
                    self.pipeline = AutoPipelineForImage2Image.from_pipe(self.pipeline_t2i)
            

        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise e

    def generate(self, prompt, image=None, strength=0.3, guidance_scale=3.5, steps=30, seed=None):
        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(seed)
        else:
             rand_seed = random.randint(0, 2**32 - 1)
             generator = torch.Generator("cpu").manual_seed(rand_seed)

        kwargs = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator 
        }
        
        if image is not None:
            kwargs["image"] = image
            kwargs["strength"] = strength
        else:
            if self.pipeline_t2i:
                return self.pipeline_t2i(**kwargs).images[0]
            else:
                if self.model_key == 'flux':
                     return self.pipeline(**kwargs).images[0]
                raise ValueError("Text-to-Image not supported for this model configuration")

        return self.pipeline(**kwargs).images[0]

def main():
    parser = argparse.ArgumentParser(description="Generate images using various Stable Diffusion models.")
    parser.add_argument("--mode", type=str, choices=['img2img', 'txt2img'], required=True, help="Generation mode.")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation.")
    parser.add_argument("--models", nargs='+', default=['sdxl'], choices=MODELS.keys(), help="Models to use.")
    parser.add_argument("--quantization", nargs='+', default=['fp16'], choices=QUANTIZATION_LEVELS.keys(), help="Quantization levels.")
    parser.add_argument("--input_dir", type=str, default="Face2Fake_pt2/to_crop_and_to_gen", help="Directory for input images (img2img).")
    parser.add_argument("--output_dir", type=str, default="Face2Fake_pt2/output", help="Directory for output images.")
    parser.add_argument("--strength", type=float, default=0.3, help="Strength for img2img.")
    parser.add_argument("--steps", type=int, default=60, help="Inference steps.")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale.")
    parser.add_argument("--device", type=str, default=None, help="Device to use.")
    parser.add_argument("--seed", type=int, default=123, help="Fixed seed for reproducibility.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    input_images = []
    if args.mode == 'img2img':
        input_images = sorted(glob.glob(os.path.join(args.input_dir, '*.jpg')) + glob.glob(os.path.join(args.input_dir, '*.png')))
        if not input_images:
            print(f"No images found in {args.input_dir}")
            return

    for model_key in args.models:
        for quant in args.quantization:
            try:
                generator = ImageGenerator(model_key, quant, device=args.device)
                

                print(f"Generating txt2img with {model_key} ({quant})...")
                
                i = 0 
                current_seed = args.seed + i if args.seed is not None else None
                
                img = generator.generate(
                    args.prompt, 
                    steps=args.steps, 
                    guidance_scale=args.guidance, 
                    seed=current_seed
                )
                
                seed_str = f"_seed{current_seed}" if current_seed is not None else ""
                out_name = f"{args.output_dir}/{model_key}_{quant}_{i}{seed_str}.png"
                img.save(out_name)
                print(f"Saved {out_name}")
                
                flush()
                            
                del generator
                torch.cuda.empty_cache()
                flush()
                
            except Exception as e:
                print(f"Failed to run {model_key} with {quant}: {e}")
                flush()

if __name__ == "__main__":
    main()