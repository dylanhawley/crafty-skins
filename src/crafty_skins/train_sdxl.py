import os
import torch
import numpy as np
import argparse
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import transformers
import diffusers
from accelerate.utils import ProjectConfiguration
from accelerate import notebook_launcher

class SDXLTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tokenizer, size=1024):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.size = size
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.size, self.size))
        
        # Convert image to tensor and normalize
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1
        image = image.permute(2, 0, 1)
        
        # You can customize the prompt based on your needs
        prompt = "A minecraft character skin and its 3D render"
        
        # Tokenize text
        tokenized_prompt = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": tokenized_prompt.input_ids[0],
            "attention_mask": tokenized_prompt.attention_mask[0]
        }

def train_sdxl(args):
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
        project_config=ProjectConfiguration(total_limit=args.batch_size),
    )

    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id,
        subfolder="tokenizer"
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id,
        subfolder="text_encoder"
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae"
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.model_id,
        subfolder="unet"
    )
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    
    # Create noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    
    # Create dataset and dataloader
    dataset = SDXLTrainingDataset(args.data_dir, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Prepare everything with accelerator
    unet, optimizer, dataloader = accelerator.prepare(
        unet, optimizer, dataloader
    )

    # Training loop
    global_step = 0
    
    for epoch in range(args.max_train_steps):
        unet.train()
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Calculate loss
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
            global_step += 1
            
            if global_step % args.log_interval == 0:
                print(f"Step {global_step}: Loss {loss.item()}")
                
            if global_step >= args.max_train_steps:
                break
    
    # Save the model
    accelerator.wait_for_everyone()
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_unet.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train SDXL model for Minecraft skin generation")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing training images")
    parser.add_argument("--output_dir", type=str, default="finetuned-sdxl",
                        help="Directory to save the model")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Base model ID from Hugging Face")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--max_train_steps", type=int, default=1000,
                        help="Maximum number of training steps")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Log interval in steps")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist")
    
    # Check if there are images in the data directory
    image_files = [f for f in os.listdir(args.data_dir) 
                  if f.endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {args.data_dir}")
    
    print(f"Found {len(image_files)} images in {args.data_dir}")
    print(f"Starting training with batch size {args.batch_size} for {args.max_train_steps} steps")
    
    train_sdxl(args)

if __name__ == "__main__":
    main() 