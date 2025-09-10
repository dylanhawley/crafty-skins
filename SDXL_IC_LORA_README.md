# SDXL In-Context LoRA for Image Conditional Generation

This repository contains scripts for training and using a LoRA (Low-Rank Adaptation) with Stable Diffusion XL (SDXL) for image conditional generation on composite images. The approach is inspired by the "In-Context LoRA for Diffusion Transformers" paper, where the model learns to generate masked panels based on the visible panels in a 4-panel composite image.

## Overview

The training process uses composite images (2x2 grid layouts) where during training, some panels are randomly masked, and the model learns to inpaint the masked regions based on the visible panels and text description. During inference, you can specify which panels to inpaint, enabling conditional generation based on the surrounding context.

## Files

- `src/training/train_sdxl_ic_lora.py` - Training script for SDXL LoRA
- `src/inference/sdxl_ic_lora_inference.py` - Inference script for conditional generation
- `requirements_sdxl_lora.txt` - Python dependencies

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements_sdxl_lora.txt
```

2. Make sure you have access to a GPU with sufficient VRAM (recommended: 16GB+ for training, 8GB+ for inference).

## Dataset Preparation

Your training dataset should be organized as follows:

```
data/
├── image1.png
├── image1.txt
├── image2.jpg
├── image2.txt
└── ...
```

Each image should be a composite 4-panel image (2x2 grid), and each `.txt` file should contain the text description for the corresponding image. The panels are indexed as:

```
┌─────────┬─────────┐
│    0    │    1    │
│ top-left│top-right│
├─────────┼─────────┤
│    2    │    3    │
│bottom-  │bottom-  │
│left     │right    │
└─────────┴─────────┘
```

## Training

### Basic Training Command

```bash
python src/training/train_sdxl_ic_lora.py \
    --data_dir /path/to/your/dataset \
    --output_dir ./output/lora_model \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 64
```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_dir` | Path to training dataset directory | Required |
| `--output_dir` | Output directory for model and checkpoints | `output` |
| `--model_id` | Base SDXL model ID | `stabilityai/stable-diffusion-xl-base-1.0` |
| `--num_epochs` | Number of training epochs | `100` |
| `--batch_size` | Training batch size | `4` |
| `--learning_rate` | Learning rate | `1e-4` |
| `--lora_rank` | LoRA rank (higher = more parameters) | `64` |
| `--lora_alpha` | LoRA alpha parameter | `64` |
| `--lora_dropout` | LoRA dropout rate | `0.1` |
| `--image_size` | Image resolution for training | `1024` |
| `--mask_probability` | Probability of masking panels during training | `0.7` |
| `--max_masked_panels` | Maximum number of panels to mask | `2` |
| `--save_every` | Save checkpoint every N steps | `1000` |
| `--resume_from_checkpoint` | Path to checkpoint to resume from | `None` |
| `--seed` | Random seed | `42` |

### Advanced Training Example

```bash
python src/training/train_sdxl_ic_lora.py \
    --data_dir /path/to/your/dataset \
    --output_dir ./output/minecraft_lora \
    --num_epochs 100 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --lora_rank 128 \
    --lora_alpha 128 \
    --mask_probability 0.8 \
    --max_masked_panels 3 \
    --save_every 500 \
    --seed 42
```

### Memory Optimization Tips

1. **Reduce batch size**: Start with `--batch_size 1` if you encounter OOM errors
2. **Lower LoRA rank**: Use `--lora_rank 32` for less memory usage
3. **Use gradient checkpointing**: The script automatically uses mixed precision (fp16)
4. **Enable CPU offloading**: The script uses Accelerate for efficient memory management

## Inference

### Basic Inference Command

```bash
python src/inference/sdxl_ic_lora_inference.py \
    --lora_path ./output/lora_model/final_model \
    --input_image /path/to/composite_image.png \
    --prompt "Your descriptive prompt here" \
    --panels_to_inpaint 2 3 \
    --output_dir ./inference_output
```

### Inference Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--lora_path` | Path to trained LoRA weights | Required |
| `--input_image` | Path to input composite image | Required |
| `--prompt` | Text prompt for generation | Required |
| `--panels_to_inpaint` | Panel indices to inpaint (space-separated) | `[2, 3]` |
| `--output_dir` | Output directory for generated images | `inference_output` |
| `--negative_prompt` | Negative prompt | `None` |
| `--num_inference_steps` | Number of denoising steps | `50` |
| `--guidance_scale` | Classifier-free guidance scale | `7.5` |
| `--num_images` | Number of images to generate | `1` |
| `--seed` | Random seed for reproducibility | `None` |
| `--use_gradient_mask` | Use soft gradient mask for smooth transitions | `False` |
| `--gradient_width` | Width of gradient transition | `20` |
| `--strength` | Denoising strength (0-1) | `1.0` |
| `--base_model` | Base SDXL model ID | `stabilityai/stable-diffusion-xl-base-1.0` |
| `--save_comparison` | Save side-by-side comparison image | `False` |

### Inference Examples

#### Generate bottom panels based on top panels:
```bash
python src/inference/sdxl_ic_lora_inference.py \
    --lora_path ./output/lora_model/final_model \
    --input_image examples/input.png \
    --prompt "A minecraft character skin design with armor and weapons" \
    --panels_to_inpaint 2 3 \
    --num_images 3 \
    --save_comparison \
    --seed 42
```

#### Generate with soft gradient mask:
```bash
python src/inference/sdxl_ic_lora_inference.py \
    --lora_path ./output/lora_model/final_model \
    --input_image examples/input.png \
    --prompt "A medieval knight character design" \
    --panels_to_inpaint 1 3 \
    --use_gradient_mask \
    --gradient_width 30 \
    --strength 0.8
```

#### Generate single panel:
```bash
python src/inference/sdxl_ic_lora_inference.py \
    --lora_path ./output/lora_model/final_model \
    --input_image examples/input.png \
    --prompt "Complete the character design" \
    --panels_to_inpaint 3 \
    --num_inference_steps 75 \
    --guidance_scale 9.0
```

## Key Features

### Training Features

1. **Random Panel Masking**: During training, panels are randomly masked to teach the model in-context learning
2. **Mixed Precision**: Automatic fp16 training for memory efficiency
3. **Gradient Accumulation**: Handles large effective batch sizes
4. **Checkpoint Resume**: Can resume training from any checkpoint
5. **LoRA Integration**: Uses PEFT library for efficient fine-tuning
6. **Multi-GPU Support**: Built on Accelerate for distributed training

### Inference Features

1. **Flexible Panel Selection**: Choose any combination of panels to inpaint
2. **Gradient Masking**: Soft transitions between masked and visible regions
3. **Batch Processing**: Process multiple images at once
4. **Comparison Visualization**: Side-by-side visualization of input, mask, and output
5. **Memory Optimization**: CPU offloading and efficient attention mechanisms

## Tips for Best Results

### Training Tips

1. **Dataset Quality**: Ensure your composite images have consistent layout and high quality
2. **Prompt Quality**: Use detailed, consistent text descriptions
3. **Masking Strategy**: Higher mask probability (0.7-0.9) generally works better
4. **Training Duration**: Start with 50-100 epochs, monitor loss curves
5. **LoRA Rank**: Higher rank (64-128) for complex tasks, lower (16-32) for simple ones

### Inference Tips

1. **Prompt Engineering**: Use detailed prompts that describe the desired output
2. **Guidance Scale**: Higher values (7.5-12) for more prompt adherence
3. **Steps**: More steps (50-100) for higher quality
4. **Strength**: Lower values (0.6-0.8) for more conservative edits
5. **Gradient Masking**: Use for smoother transitions between panels

## Troubleshooting

### Common Training Issues

1. **OOM Errors**: Reduce batch size, lower LoRA rank, or use smaller images
2. **Loss Not Decreasing**: Check learning rate, ensure dataset is properly formatted
3. **Poor Quality**: Increase training epochs, check mask probability
4. **Artifacts**: Lower learning rate, add regularization

### Common Inference Issues

1. **Poor Generation Quality**: Check LoRA training quality, adjust guidance scale
2. **Inconsistent Style**: Ensure training data had consistent style
3. **Blurry Transitions**: Use gradient masking, adjust gradient width
4. **Memory Issues**: Enable CPU offloading, reduce image size

## Model Architecture

The implementation uses:
- **Base Model**: Stable Diffusion XL (SDXL)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Target Modules**: UNet attention and feed-forward layers
- **Conditioning**: Text embeddings + masked image conditioning
- **Training Objective**: Denoising loss on composite images

## License

This code is provided for research and educational purposes. Please check the licenses of the underlying models (SDXL) and libraries used.

## Citation

If you use this code in your research, please cite the relevant papers:

```bibtex
@article{contextlora2024,
  title={In-Context LoRA for Diffusion Transformers},
  author={Author et al.},
  journal={arXiv preprint},
  year={2024}
}
``` 