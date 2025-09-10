#!/usr/bin/env python3
"""
Example usage script for SDXL In-Context LoRA training and inference.

This script demonstrates how to:
1. Prepare your dataset
2. Train a LoRA model
3. Use the trained model for inference

Run this script to see example commands and usage patterns.
"""

import os
import subprocess
from pathlib import Path


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_command(description, command):
    """Print a command with description."""
    print(f"\n{description}:")
    print(f"```bash")
    print(command)
    print("```")


def main():
    print("SDXL In-Context LoRA - Example Usage")
    
    # Dataset preparation example
    print_section("1. Dataset Preparation")
    print("""
Your dataset should be organized like this:

data/training/
├── sample1.png  (4-panel composite image)
├── sample1.txt  (text description)
├── sample2.png
├── sample2.txt
└── ...

Each .txt file should contain a description like:
"This four-panel image showcases a character design transformation..."
""")
    
    # Training examples
    print_section("2. Training Examples")
    
    print_command(
        "Basic training command",
        """python src/training/train_sdxl_ic_lora.py \\
    --data_dir data/training \\
    --output_dir output/my_lora \\
    --num_epochs 50 \\
    --batch_size 4 \\
    --learning_rate 1e-4"""
    )
    
    print_command(
        "Advanced training with custom settings",
        """python src/training/train_sdxl_ic_lora.py \\
    --data_dir data/training \\
    --output_dir output/minecraft_lora \\
    --num_epochs 100 \\
    --batch_size 2 \\
    --learning_rate 5e-5 \\
    --lora_rank 128 \\
    --lora_alpha 128 \\
    --mask_probability 0.8 \\
    --max_masked_panels 3 \\
    --save_every 500 \\
    --seed 42"""
    )
    
    print_command(
        "Resume training from checkpoint",
        """python src/training/train_sdxl_ic_lora.py \\
    --data_dir data/training \\
    --output_dir output/my_lora \\
    --resume_from_checkpoint output/my_lora/checkpoint-1000"""
    )
    
    # Inference examples
    print_section("3. Inference Examples")
    
    print_command(
        "Basic inference - generate bottom panels",
        """python src/inference/sdxl_ic_lora_inference.py \\
    --lora_path output/my_lora/final_model \\
    --input_image examples/input.png \\
    --prompt "A detailed character design with armor and weapons" \\
    --panels_to_inpaint 2 3 \\
    --output_dir inference_output"""
    )
    
    print_command(
        "Generate with soft gradient mask for smooth transitions",
        """python src/inference/sdxl_ic_lora_inference.py \\
    --lora_path output/my_lora/final_model \\
    --input_image examples/input.png \\
    --prompt "A medieval knight character design" \\
    --panels_to_inpaint 1 3 \\
    --use_gradient_mask \\
    --gradient_width 30 \\
    --strength 0.8 \\
    --save_comparison"""
    )
    
    print_command(
        "High-quality generation with multiple samples",
        """python src/inference/sdxl_ic_lora_inference.py \\
    --lora_path output/my_lora/final_model \\
    --input_image examples/input.png \\
    --prompt "Complete the character design with intricate details" \\
    --panels_to_inpaint 3 \\
    --num_images 5 \\
    --num_inference_steps 75 \\
    --guidance_scale 9.0 \\
    --seed 42"""
    )
    
    # Memory optimization tips
    print_section("4. Memory Optimization")
    
    print("""
If you encounter out-of-memory errors:

Training:
- Reduce batch size: --batch_size 1
- Lower LoRA rank: --lora_rank 32
- Use smaller images: --image_size 512

Inference:
- Use CPU offloading (automatically enabled)
- Process one image at a time
- Reduce inference steps if needed
""")
    
    # Advanced usage
    print_section("5. Advanced Usage Patterns")
    
    print_command(
        "Batch processing multiple images",
        """# Create a simple batch script
for img in examples/*.png; do
    python src/inference/sdxl_ic_lora_inference.py \\
        --lora_path output/my_lora/final_model \\
        --input_image "$img" \\
        --prompt "Character design completion" \\
        --panels_to_inpaint 2 3 \\
        --output_dir "inference_output/$(basename "$img" .png)"
done"""
    )
    
    # Panel configurations
    print_section("6. Panel Configuration Examples")
    
    panel_examples = [
        ("Generate bottom-left panel only", "[2]"),
        ("Generate right side panels", "[1, 3]"),
        ("Generate bottom row", "[2, 3]"),
        ("Generate three panels (keep top-left)", "[1, 2, 3]"),
    ]
    
    for description, panels in panel_examples:
        print(f"\n{description}: --panels_to_inpaint {panels}")
    
    print("""
Panel layout reference:
┌─────────┬─────────┐
│    0    │    1    │
│ top-left│top-right│
├─────────┼─────────┤
│    2    │    3    │
│bottom-  │bottom-  │
│left     │right    │
└─────────┴─────────┘
""")
    
    # Quick start checklist
    print_section("7. Quick Start Checklist")
    
    checklist = [
        "Install dependencies: pip install -r requirements_sdxl_lora.txt",
        "Prepare dataset: 4-panel images + text descriptions",
        "Start training: python src/training/train_sdxl_ic_lora.py --data_dir data/training --output_dir output/my_lora",
        "Monitor training: Check tensorboard logs or console output",
        "Test inference: python src/inference/sdxl_ic_lora_inference.py --lora_path output/my_lora/final_model --input_image test.png --prompt 'test prompt'",
        "Experiment with different panel configurations and prompts"
    ]
    
    for i, item in enumerate(checklist, 1):
        print(f"{i}. {item}")
    
    print_section("8. Troubleshooting")
    
    print("""
Common issues and solutions:

1. CUDA out of memory:
   - Reduce batch size or LoRA rank
   - Use gradient checkpointing (automatically enabled)
   
2. Poor generation quality:
   - Train for more epochs
   - Check dataset quality and consistency
   - Adjust guidance scale during inference
   
3. Inconsistent results:
   - Use fixed seeds for reproducibility
   - Ensure consistent prompt formatting
   
4. Slow training:
   - Enable xformers: pip install xformers
   - Use multiple GPUs with accelerate config
   
5. Import errors:
   - Update diffusers: pip install --upgrade diffusers
   - Check CUDA compatibility
""")
    
    print("\n" + "="*60)
    print(" Ready to start training and inference!")
    print("="*60)


if __name__ == "__main__":
    main() 