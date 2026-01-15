# The Impact of Weight Quantization in Fake Image Detection

## Overview

This project investigates the impact of **weight quantization** in diffusion-based image generators on the performance of **state-of-the-art fake image detectors**.

Specifically, we study how different quantization levels — **FP16, FP8, and FP4** — applied to modern diffusion models affect the ability of detectors to correctly identify AI-generated images.

The goal is to understand the **trade-off between computational efficiency and detectability**, and to analyze whether aggressive quantization alters the visual or statistical artifacts exploited by fake image detectors.

---

## Scope of the Study

- **Generators**
  - Stable Diffusion XL (SDXL)
  - Stable Diffusion 3
  - Stable Diffusion 3.5
  - Flux.1

- **Quantization Levels**
  - FP16 (baseline)
  - FP8 (8-bit quantization via bitsandbytes)
  - FP4 (4-bit NF4 quantization)

- **Generation Modes**
  - Text-to-Image (txt2img)
  - Image-to-Image (img2img)

- **Evaluation**
  - Performance of state-of-the-art fake image detectors
  - Accuracy, Precision, Recall, F1-score, ROC-AUC
  - Efficiency considerations (memory usage, inference time)

---

## Tools & Libraries

This project is implemented in **PyTorch** and relies on the following main libraries:

- `torch`, `torchvision`
- `diffusers`
- `transformers`
- `accelerate`
- `bitsandbytes`
- `numpy`, `scipy`
- `Pillow`, `opencv-python`
- `sentencepiece`, `protobuf`

The project is designed to be run on **GPU-enabled environments**, and **Google Colab** is strongly recommended.

---

## Installation

### Create a Python environment (optional but recommended)

```bash
python -m venv quantization-env
source quantization-env/bin/activate  # Linux / macOS
quantization-env\Scripts\activate     # Windows
```
### Install dependencies
```bash
pip install torch torchvision diffusers transformers accelerate bitsandbytes scipy numpy Pillow opencv-python sentencepiece protobuf
```


## Dataset Generation

Fake images are generated using fixed prompts and fixed random seeds to ensure that any observed differences are caused only by quantization effects.

### Prompt Strategy

The same prompt is used across all models and quantization levels to avoid content bias.

Example prompt:

"A high-resolution portrait photograph, realistic lighting, DSLR, shallow depth of field"

### Text-to-Image Generation (txt2img)

This mode generates images directly from text prompts.

#### Command
```bash
python txt2img.py \
  --mode txt2img \
  --prompt "A futuristic cyberpunk city with neon lights" \
  --models sdxl sd35 flux sd3 sd15 \
  --quantization fp16 fp8 fp4 \
  --steps 60 \
  --seed 123 \
  --output_dir dataset/fake/txt2img
```

### Image-to-Image Generation (img2img)

This mode generates fake images starting from real seed images, producing more realistic and harder-to-detect fakes.

#### Input

Place real images in a directory, for example:

dataset/real/
├── demo1.png
├── demo2.png
└── demo3.png

#### Command
```bash
python img2img.py \
  --input_dir dataset/real \
  --prompt "oil painting, van gogh style, thick brushstrokes" \
  --models sd15 sd3 sd35 \
  --quantization fp16 fp8 fp4 \
  --strength 0.6 \
  --steps 30 \
  --seed 123 \
  --output_dir dataset/fake/img2img
```
## Fake Image Detection

State-of-the-art fake image detectors are evaluated on the generated datasets.

Detectors are sourced from:

🔗 Image Deepfake Detectors Public Library
https://github.com/truebees-ai/Image-Deepfake-Detectors-Public-Library

#### Labels
0 → Real image
1 → Fake image

### References

Hugging Face Diffusers Quantization
https://huggingface.co/docs/diffusers/en/quantization/bitsandbytes

Quanto & Diffusers Blog
https://huggingface.co/blog/quanto-diffusers

Diffusers Quantization API
https://huggingface.co/docs/diffusers/en/api/quantization

Image Deepfake Detectors Library
https://github.com/truebees-ai/Image-Deepfake-Detectors-Public-Library
