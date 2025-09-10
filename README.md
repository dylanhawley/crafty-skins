# Crafty Skins

A tool for generating Minecraft character skins using Stable Diffusion XL.

## Setup

### Prerequisites

- Python 3.12
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crafty-skins.git
   cd crafty-skins
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Usage

### Training

To fine-tune the SDXL model on your own dataset of Minecraft skins:

```bash
python train_sdxl.py --data_dir /path/to/your/images --output_dir finetuned-sdxl --batch_size 1 --max_train_steps 18000
```

Parameters:
- `--data_dir`: Directory containing training images (PNG, JPG, JPEG)
- `--output_dir`: Directory to save the fine-tuned model (default: finetuned-sdxl)
- `--model_id`: Base model ID from Hugging Face (default: stabilityai/stable-diffusion-xl-base-1.0)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--max_train_steps`: Maximum number of training steps (default: 1000)
- `--batch_size`: Training batch size (default: 1)
- `--gradient_accumulation_steps`: Number of gradient accumulation steps (default: 4)
- `--log_interval`: Log interval in steps (default: 100)

### Inference

To generate a Minecraft skin using your fine-tuned model:

```bash
python inference.py --model_path finetuned-sdxl --input_image input.png --output_image output.png --prompt "A minecraft character skin"
```

Parameters:
- `--model_path`: Path to the fine-tuned SDXL model (default: finetuned-sdxl)
- `--input_image`: Path to the input image (required)
- `--output_image`: Path to save the output image (default: output.png)
- `--prompt`: Text prompt for image generation (default: "A minecraft character skin")
- `--strength`: How much to transform the input image (0-1, default: 0.75)
- `--guidance_scale`: Guidance scale for the diffusion model (default: 7.5)
- `--cpu`: Use CPU instead of CUDA (default: use CUDA if available)

## Requirements

The project requires the following Python packages:
- torch
- diffusers
- transformers
- accelerate
- datasets
- pillow
- numpy
- argparse

These dependencies are managed through Poetry and specified in the pyproject.toml file.

Sources
https://www.reddit.com/r/StableDiffusion/comments/1ejr20p/comparative_analysis_of_image_resolutions_with/

This four-panel image showcases a rustic living room with warm wood tones and cozy decor elements; `[TOP-LEFT]` features a large stone fireplace with wooden shelves filled with books and candles; `[TOP-RIGHT]` shows a vintage leather sofa draped in plaid blankets, complemented by a mix of textured cushions; `[BOTTOM-LEFT]` displays a corner with a wooden armchair beside a side table holding a steaming mug and a classic book; `[BOTTOM-RIGHT]` captures a cozy reading nook with a window seat, a soft fur throw, and decorative logs stacked neatly.

This four-panel image showcases a transformation from a photorealistic image to a Minecraft skin; [TOP-LEFT] features our <subject>, which has the description <describe character>; [TOP-RIGHT] shows the left front up and right back up isometric positions of the <subject> as a Minecraft avatar; [BOTTOM-LEFT] displays the left back down and right front down isometric positions of the <subject> as a Minecraft avatar; [BOTTOM-RIGHT] is a complete uv map for the Minecraft skin file constructed from the isometric views of our <subject> laid out in the previous two panels.