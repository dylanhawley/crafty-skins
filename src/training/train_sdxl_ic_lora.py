import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from typing import Optional, Tuple, List
import json
from datetime import datetime

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers.schedulers import DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompositeImageDataset(Dataset):
    """Dataset for loading composite images with text descriptions."""
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 1024,
        mask_probability: float = 0.5,
        max_masked_panels: int = 2,
        transform: Optional[transforms.Compose] = None
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.mask_probability = mask_probability
        self.max_masked_panels = max_masked_panels
        
        # Find all image files
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.samples = []
        
        for img_path in self.data_dir.rglob('*'):
            if img_path.suffix.lower() in self.image_extensions:
                txt_path = img_path.with_suffix('.txt')
                if txt_path.exists():
                    self.samples.append((img_path, txt_path))
        
        logger.info(f"Found {len(self.samples)} samples in {data_dir}")
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def create_panel_mask(self, image_size: int) -> torch.Tensor:
        """Create a mask for panels in a 2x2 grid composite image."""
        mask = torch.ones((image_size, image_size))
        panel_height = image_size // 2
        panel_width = image_size // 2
        
        # Define panel coordinates (top-left, top-right, bottom-left, bottom-right)
        panels = [
            (0, panel_height, 0, panel_width),  # top-left
            (0, panel_height, panel_width, image_size),  # top-right
            (panel_height, image_size, 0, panel_width),  # bottom-left
            (panel_height, image_size, panel_width, image_size)  # bottom-right
        ]
        
        # Randomly select panels to mask
        if random.random() < self.mask_probability:
            num_panels_to_mask = random.randint(1, min(self.max_masked_panels, len(panels)))
            panels_to_mask = random.sample(range(len(panels)), num_panels_to_mask)
            
            for panel_idx in panels_to_mask:
                y1, y2, x1, x2 = panels[panel_idx]
                mask[y1:y2, x1:x2] = 0.0
        
        return mask
    
    def __getitem__(self, idx):
        img_path, txt_path = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        # Load caption
        with open(txt_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        
        # Create mask for training
        mask = self.create_panel_mask(self.image_size)
        
        # Apply mask to image for conditional input
        masked_image = image * mask.unsqueeze(0)
        
        return {
            "image": image,
            "masked_image": masked_image,
            "mask": mask,
            "caption": caption,
            "image_path": str(img_path)
        }


def collate_fn(batch):
    """Custom collate function for the dataloader."""
    images = torch.stack([item["image"] for item in batch])
    masked_images = torch.stack([item["masked_image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    captions = [item["caption"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    
    return {
        "images": images,
        "masked_images": masked_images,
        "masks": masks,
        "captions": captions,
        "image_paths": image_paths
    }


class SDXLLoRATrainer:
    """SDXL LoRA trainer for image conditional generation."""
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lora_rank: int = 64,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
        device: str = "auto"
    ):
        self.model_id = model_id
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="fp16",
            log_with="tensorboard"
        )
        
        # Load models
        self._load_models()
        self._setup_lora()
        
    def _load_models(self):
        """Load SDXL components."""
        logger.info(f"Loading SDXL models from {self.model_id}")
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id,
            subfolder="unet",
            torch_dtype=torch.float16
        )
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float16
        )
        
        # Load text encoders
        self.text_encoder_1 = CLIPTextModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            torch_dtype=torch.float16
        )
        
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.model_id,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16
        )
        
        # Load tokenizers
        self.tokenizer_1 = CLIPTokenizer.from_pretrained(
            self.model_id,
            subfolder="tokenizer"
        )
        
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.model_id,
            subfolder="tokenizer_2"
        )
        
        # Load scheduler
        self.scheduler = DDPMScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler"
        )
        
        # Freeze models
        self.vae.requires_grad_(False)
        self.text_encoder_1.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        
    def _setup_lora(self):
        """Setup LoRA for UNet."""
        logger.info("Setting up LoRA configuration")
        
        # Define target modules for LoRA
        target_modules = [
            "to_k", "to_q", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2"
        ]
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.DIFFUSION_MODEL_CONDITIONING
        )
        
        # Apply LoRA to UNet
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
    def encode_text(self, captions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text using both CLIP text encoders."""
        # Tokenize with both tokenizers
        tokens_1 = self.tokenizer_1(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        tokens_2 = self.tokenizer_2(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        tokens_1 = {k: v.to(self.accelerator.device) for k, v in tokens_1.items()}
        tokens_2 = {k: v.to(self.accelerator.device) for k, v in tokens_2.items()}
        
        # Encode with both text encoders
        with torch.no_grad():
            encoder_1_output = self.text_encoder_1(**tokens_1)
            encoder_2_output = self.text_encoder_2(**tokens_2)
        
        # Get embeddings
        prompt_embeds_1 = encoder_1_output.last_hidden_state
        prompt_embeds_2 = encoder_2_output.last_hidden_state
        
        # Pool encoder 2 output
        pooled_prompt_embeds = encoder_2_output.text_embeds
        
        # Concatenate embeddings
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        
        return prompt_embeds, pooled_prompt_embeds
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        save_every: int = 1000,
        output_dir: str = "output",
        resume_from_checkpoint: Optional[str] = None
    ):
        """Train the LoRA model."""
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08
        )
        
        # Setup learning rate scheduler
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=num_epochs * len(train_dataloader)
        )
        
        # Prepare for training
        self.unet, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            self.unet, optimizer, train_dataloader, lr_scheduler
        )
        
        # Move other models to device
        self.vae = self.vae.to(self.accelerator.device)
        self.text_encoder_1 = self.text_encoder_1.to(self.accelerator.device)
        self.text_encoder_2 = self.text_encoder_2.to(self.accelerator.device)
        
        # Training variables
        global_step = 0
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            self.accelerator.load_state(resume_from_checkpoint)
        
        logger.info("Starting training...")
        
        for epoch in range(num_epochs):
            self.unet.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.unet):
                    # Get batch data
                    images = batch["images"]
                    masked_images = batch["masked_images"]
                    masks = batch["masks"]
                    captions = batch["captions"]
                    
                    batch_size = images.shape[0]
                    
                    # Encode images to latent space
                    with torch.no_grad():
                        latents = self.vae.encode(images).latent_dist.sample()
                        latents = latents * self.vae.config.scaling_factor
                        
                        masked_latents = self.vae.encode(masked_images).latent_dist.sample()
                        masked_latents = masked_latents * self.vae.config.scaling_factor
                        
                        # Resize mask to latent dimensions
                        mask_latents = F.interpolate(
                            masks.unsqueeze(1).float(),
                            size=(latents.shape[2], latents.shape[3]),
                            mode="nearest"
                        )
                    
                    # Sample noise
                    noise = torch.randn_like(latents)
                    
                    # Sample timesteps
                    timesteps = torch.randint(
                        0, self.scheduler.config.num_train_timesteps, 
                        (batch_size,), 
                        device=latents.device
                    ).long()
                    
                    # Add noise to latents
                    noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                    
                    # Concatenate masked latents and mask for conditioning
                    conditioning = torch.cat([masked_latents, mask_latents], dim=1)
                    
                    # Encode text
                    prompt_embeds, pooled_prompt_embeds = self.encode_text(captions)
                    
                    # Prepare additional conditioning for SDXL
                    add_time_ids = torch.tensor([
                        [1024, 1024, 0, 0, 1024, 1024]  # original_size + crop_coords + target_size
                    ]).repeat(batch_size, 1).to(latents.device, dtype=torch.float16)
                    
                    # UNet prediction
                    model_pred = self.unet(
                        torch.cat([noisy_latents, conditioning], dim=1),
                        timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={
                            "text_embeds": pooled_prompt_embeds,
                            "time_ids": add_time_ids
                        }
                    ).sample
                    
                    # Calculate loss
                    target = noise
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # Update progress
                if self.accelerator.sync_gradients:
                    global_step += 1
                    epoch_loss += loss.detach().item()
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        "loss": f"{loss.detach().item():.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
                    })
                    
                    # Save checkpoint
                    if global_step % save_every == 0:
                        save_path = output_dir / f"checkpoint-{global_step}"
                        self.accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint at step {global_step}")
            
            # Log epoch results
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Save model at end of epoch
            if self.accelerator.is_main_process:
                self.save_model(output_dir / f"epoch_{epoch+1}")
        
        logger.info("Training completed!")
        
        # Save final model
        if self.accelerator.is_main_process:
            self.save_model(output_dir / "final_model")
    
    def save_model(self, save_path: Path):
        """Save the trained LoRA model."""
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        self.unet.save_pretrained(save_path)
        
        # Save training config
        config = {
            "model_id": self.model_id,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(save_path / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SDXL LoRA for image conditional generation")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for model and checkpoints")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="SDXL model ID")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--image_size", type=int, default=1024, help="Image size for training")
    parser.add_argument("--mask_probability", type=float, default=0.7, help="Probability of masking panels during training")
    parser.add_argument("--max_masked_panels", type=int, default=2, help="Maximum number of panels to mask")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create dataset
    logger.info(f"Creating dataset from {args.data_dir}")
    dataset = CompositeImageDataset(
        data_dir=args.data_dir,
        image_size=args.image_size,
        mask_probability=args.mask_probability,
        max_masked_panels=args.max_masked_panels
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create trainer
    trainer = SDXLLoRATrainer(
        model_id=args.model_id,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Train model
    trainer.train(
        train_dataloader=dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint
    )


if __name__ == "__main__":
    main() 