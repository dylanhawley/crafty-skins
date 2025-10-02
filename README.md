# Crafty Skins

A project for generating Minecraft skins using IP-Adapter with Stable Diffusion XL.

## Training

```bash
accelerate launch src/train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --dataset_name="deepwaterhorizon/minecraft-skins-legacy" \
  --resolution=1024 \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --max_train_steps=200000 \
  --learning_rate=1e-06 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=1000 \
  --mixed_precision="bf16" \
  --report_to="wandb" \
  --validation_prompt="a skin of Alan Turing" \
  --checkpointing_steps=25000 \
  --output_dir="sdxl-minecraft-model" \
  --image_column="target_image" \
  --push_to_hub \
  --allow_tf32
```

## Inference

Generate Minecraft skins using IP-Adapter with face and style images.

### Basic Usage

```bash
python src/inference_ip_adapter_sdxl.py \
  --face_image path/to/face.jpg \
  --style_image path/to/style.jpg
```

### Advanced Usage

```bash
python src/inference_ip_adapter_sdxl.py \
  --face_image data/face.jpg \
  --style_image data/style.jpg \
  --prompt "a steampunk character" \
  --ip_adapter_scale_style 0.7 \
  --ip_adapter_scale_face 0.3 \
  --num_inference_steps 100 \
  --guidance_scale 8.0 \
  --seed 42 \
  --output steampunk_skin.png
```