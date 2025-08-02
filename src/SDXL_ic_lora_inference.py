import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.schedulers import DDIMScheduler
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import argparse
from typing import Optional, Union, List, Tuple
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SDXLLoRAInference:
    """SDXL LoRA inference for image conditional generation with SDEdit masking."""
    
    def __init__(
        self,
        base_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        lora_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the SDXL LoRA inference pipeline.
        
        Args:
            base_model_id (str): Base SDXL model ID
            lora_path (Optional[str]): Path to trained LoRA weights
            device (Optional[str]): Device to use for inference
            torch_dtype (torch.dtype): Data type for inference
        """
        if device is None:
            if torch.cuda.is_available():
                try:
                    # Test CUDA functionality
                    torch.cuda.empty_cache()
                    test_tensor = torch.randn(1, device="cuda")
                    device = "cuda"
                    logger.info("CUDA is available and functional")
                except Exception as e:
                    logger.warning(f"CUDA available but not functional: {e}")
                    logger.info("Falling back to CPU")
                    device = "cpu"
            elif torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using MPS (Apple Silicon)")
            else:
                device = "cpu"
                logger.info("Using CPU")
        
        self.device = device
        self.torch_dtype = torch_dtype
        self.base_model_id = base_model_id
        self.lora_path = lora_path
        
        # Load the pipeline
        self._load_pipeline()
        
        # Load LoRA if provided
        if lora_path:
            self._load_lora()
    
    def _load_pipeline(self):
        """Load the SDXL inpainting pipeline."""
        logger.info(f"Loading SDXL pipeline from {self.base_model_id}")
        
        # Try to load inpainting pipeline first, fallback to base model
        try:
            self.pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                self.base_model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                variant="fp16" if self.torch_dtype == torch.float16 else None
            )
        except Exception as e:
            logger.warning(f"Could not load as inpainting pipeline: {e}")
            logger.info("Falling back to base model and converting to inpainting pipeline")
            from diffusers import StableDiffusionXLPipeline
            base_pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.base_model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                variant="fp16" if self.torch_dtype == torch.float16 else None
            )
            # Convert to inpainting pipeline
            self.pipeline = StableDiffusionXLInpaintPipeline(
                vae=base_pipeline.vae,
                text_encoder=base_pipeline.text_encoder,
                text_encoder_2=base_pipeline.text_encoder_2,
                tokenizer=base_pipeline.tokenizer,
                tokenizer_2=base_pipeline.tokenizer_2,
                unet=base_pipeline.unet,
                scheduler=base_pipeline.scheduler
            )
        
        # Use DDIM scheduler for better quality
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        self.pipeline.to(self.device)
        
        # Disable xformers to avoid compatibility issues
        if hasattr(self.pipeline, "disable_xformers_memory_efficient_attention"):
            self.pipeline.disable_xformers_memory_efficient_attention()
        
        # Note: Disable CPU offload when using LoRA to avoid weight conflicts
        # Enable model CPU offload for memory efficiency only if no LoRA
        if self.device == "cuda" and not self.lora_path:
            self.pipeline.enable_model_cpu_offload()
        elif self.lora_path:
            logger.info("Keeping models on GPU due to LoRA usage (CPU offload disabled)")
    
    def _load_lora(self):
        """Load the trained LoRA weights."""
        logger.info(f"Loading LoRA weights from {self.lora_path}")
        
        # Load training config if available to get model info
        config_path = Path(self.lora_path) / "training_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.lora_config = json.load(f)
                logger.info(f"Loaded LoRA config: {self.lora_config}")
        
        self.lora_loaded = False
        self.lora_method = None
        
        # Load LoRA weights with error handling
        try:
            # Try loading with automatic detection
            self.pipeline.load_lora_weights(self.lora_path)
            self.lora_loaded = True
            self.lora_method = "standard"
            logger.info("✅ Successfully loaded LoRA weights using standard method")
            
            # Verify LoRA is applied by checking for adapters
            self._verify_lora_loaded()
            
        except Exception as e:
            logger.warning(f"Standard LoRA loading failed: {e}")
            try:
                # Try loading with prefix=None to resolve prefix warnings
                from peft import PeftModel
                
                # Load LoRA directly to UNet
                adapter_path = Path(self.lora_path) / "adapter_model.safetensors"
                if adapter_path.exists():
                    self.pipeline.unet = PeftModel.from_pretrained(
                        self.pipeline.unet, 
                        self.lora_path,
                        torch_dtype=self.torch_dtype
                    )
                    # Ensure UNet is on correct device
                    self.pipeline.unet.to(self.device)
                    self.lora_loaded = True
                    self.lora_method = "peft_direct"
                    logger.info("✅ Successfully loaded LoRA weights directly to UNet")
                else:
                    # Fallback to standard loading with adapter name
                    self.pipeline.load_lora_weights(self.lora_path, adapter_name="default")
                    self.lora_loaded = True
                    self.lora_method = "adapter_name"
                    logger.info("✅ Successfully loaded LoRA weights with adapter name")
                    
                self._verify_lora_loaded()
                
            except Exception as e2:
                logger.error(f"❌ CRITICAL: Failed to load LoRA weights: {e2}")
                logger.error("❌ Inference will proceed WITHOUT LoRA - results may not match training!")
                self.lora_loaded = False
                self.lora_method = None
                
                # Raise exception instead of silently continuing
                raise RuntimeError(f"Failed to load LoRA weights from {self.lora_path}. "
                                 f"Please check the path and LoRA format. Error: {e2}")
    
    def _verify_lora_loaded(self):
        """Verify that LoRA weights are properly loaded and applied."""
        try:
            # Check if UNet has adapters (for diffusers LoRA)
            if hasattr(self.pipeline.unet, 'get_active_adapters'):
                active_adapters = self.pipeline.unet.get_active_adapters()
                if active_adapters:
                    logger.info(f"✅ LoRA verification: Active adapters found: {active_adapters}")
                else:
                    logger.warning("⚠️  No active adapters found - LoRA may not be applied")
            
            # Check if UNet is a PeftModel (for direct PEFT loading)
            elif hasattr(self.pipeline.unet, 'peft_config'):
                logger.info("✅ LoRA verification: UNet is a PeftModel")
                
            # Check for LoRA layers in UNet
            elif hasattr(self.pipeline.unet, 'named_modules'):
                lora_layers = [name for name, module in self.pipeline.unet.named_modules() 
                              if 'lora' in name.lower()]
                if lora_layers:
                    logger.info(f"✅ LoRA verification: Found {len(lora_layers)} LoRA layers")
                else:
                    logger.warning("⚠️  No LoRA layers found in UNet")
            
            # Ensure models are on correct device
            self.pipeline.unet.to(self.device)
            logger.info(f"✅ Models confirmed on device: {self.device}")
            
        except Exception as e:
            logger.warning(f"Could not verify LoRA loading: {e}")
    
    def get_lora_status(self):
        """Get current LoRA loading status."""
        status = {
            "loaded": getattr(self, 'lora_loaded', False),
            "method": getattr(self, 'lora_method', None),
            "path": self.lora_path
        }
        
        # Add adapter scale information if available
        try:
            if hasattr(self.pipeline, 'get_active_adapters'):
                active_adapters = self.pipeline.get_active_adapters()
                status["active_adapters"] = active_adapters
            
            # Check for adapter scales
            if hasattr(self.pipeline.unet, 'get_active_adapters'):
                unet_adapters = self.pipeline.unet.get_active_adapters()
                status["unet_adapters"] = unet_adapters
                
        except Exception as e:
            logger.debug(f"Could not get adapter details: {e}")
            
        return status
    
    def set_lora_scale(self, scale: float = 1.0):
        """Set the LoRA adapter scale/strength."""
        try:
            if hasattr(self.pipeline, 'set_adapters') and hasattr(self.pipeline, 'get_active_adapters'):
                active_adapters = self.pipeline.get_active_adapters()
                if active_adapters:
                    # Set scale for all active adapters
                    scales = {adapter: scale for adapter in active_adapters}
                    self.pipeline.set_adapters(list(active_adapters), adapter_weights=list(scales.values()))
                    logger.info(f"🎛️  Set LoRA scale to {scale} for adapters: {active_adapters}")
                else:
                    logger.warning("No active adapters found to set scale")
            else:
                logger.info("Pipeline doesn't support adapter scaling (using PEFT direct method)")
        except Exception as e:
            logger.warning(f"Could not set LoRA scale: {e}")
    
    def create_panel_mask(
        self, 
        image_size: Tuple[int, int], 
        panels_to_inpaint: List[int]
    ) -> Image.Image:
        """
        Create a mask for specific panels in a 2x2 grid.
        
        Args:
            image_size (Tuple[int, int]): Size of the image (width, height)
            panels_to_inpaint (List[int]): List of panel indices to inpaint (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)
        
        Returns:
            Image.Image: Binary mask image (white=keep, black=inpaint)
        """
        width, height = image_size
        mask = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(mask)
        
        panel_width = width // 2
        panel_height = height // 2
        
        # Define panel coordinates
        panel_coords = [
            (0, 0, panel_width, panel_height),  # top-left
            (panel_width, 0, width, panel_height),  # top-right
            (0, panel_height, panel_width, height),  # bottom-left
            (panel_width, panel_height, width, height)  # bottom-right
        ]
        
        for panel_idx in range(len(panel_coords)):
            if panel_idx not in panels_to_inpaint:
                coords = panel_coords[panel_idx]
                draw.rectangle(coords, fill=0)
        
        return mask
    
    def generate_conditional(
        self,
        image: Union[str, Image.Image],
        panels_to_inpaint: List[int],
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        strength: float = 1.0,
        lora_scale: float = 1.0
    ) -> List[Image.Image]:
        """
        Generate conditional images by inpainting specific panels.
        
        Args:
            image (Union[str, Image.Image]): Input composite image
            panels_to_inpaint (List[int]): Panel indices to inpaint (0-3)
            prompt (str): Text prompt for generation
            negative_prompt (Optional[str]): Negative prompt
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale
            num_images_per_prompt (int): Number of images to generate
            seed (Optional[int]): Random seed
            strength (float): Denoising strength (0-1)
            lora_scale (float): LoRA adapter scale/strength (0-1, default 1.0)
        
        Returns:
            List[Image.Image]: Generated images
        """
        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Create mask (always use sharp borders)
        mask = self.create_panel_mask(image.size, panels_to_inpaint)
        
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Set LoRA scale if LoRA is loaded
        if getattr(self, 'lora_loaded', False) and lora_scale != 1.0:
            self.set_lora_scale(lora_scale)
        
        # Check and log LoRA status before generation
        lora_status = self.get_lora_status()
        if lora_status["loaded"]:
            logger.info(f"🎯 LoRA ACTIVE: Using {lora_status['method']} method from {lora_status['path']} (scale: {lora_scale})")
        else:
            logger.warning("⚠️  LoRA NOT ACTIVE: Generating without LoRA weights!")
        
        # Generate images
        logger.info(f"Generating {num_images_per_prompt} image(s) with {num_inference_steps} steps")
        
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            strength=strength
        )
        
        return result.images
    
    def batch_generate(
        self,
        image_paths: List[str],
        panels_to_inpaint: List[int],
        prompts: List[str],
        output_dir: str,
        **generation_kwargs
    ) -> List[str]:
        """
        Batch generate images from multiple inputs.
        
        Args:
            image_paths (List[str]): List of input image paths
            panels_to_inpaint (List[int]): Panel indices to inpaint
            prompts (List[str]): List of prompts (one per image)
            output_dir (str): Output directory
            **generation_kwargs: Additional generation arguments
        
        Returns:
            List[str]: List of output image paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        
        for i, (image_path, prompt) in enumerate(zip(image_paths, prompts)):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            # Generate images
            generated_images = self.generate_conditional(
                image=image_path,
                panels_to_inpaint=panels_to_inpaint,
                prompt=prompt,
                **generation_kwargs
            )
            
            # Save generated images
            for j, generated_image in enumerate(generated_images):
                output_path = output_dir / f"generated_{i:04d}_{j}.png"
                generated_image.save(output_path)
                output_paths.append(str(output_path))
                
                # Also save the mask for reference
                mask = self.create_panel_mask(generated_image.size, panels_to_inpaint)
                mask_path = output_dir / f"mask_{i:04d}_{j}.png"
                mask.save(mask_path)
        
        return output_paths
    
    def save_comparison(
        self,
        original_image: Image.Image,
        generated_image: Image.Image,
        mask: Image.Image,
        output_path: str
    ):
        """
        Save a comparison image showing original, mask, and generated result.
        
        Args:
            original_image (Image.Image): Original input image
            generated_image (Image.Image): Generated output image
            mask (Image.Image): Mask used for generation
            output_path (str): Path to save comparison
        """
        # Resize all images to same size
        size = original_image.size
        generated_image = generated_image.resize(size)
        mask = mask.resize(size)
        
        # Convert mask to RGB for visualization
        mask_rgb = mask.convert("RGB")
        
        # Create comparison image
        comparison_width = size[0] * 3
        comparison_height = size[1]
        comparison = Image.new("RGB", (comparison_width, comparison_height))
        
        # Paste images side by side
        comparison.paste(original_image, (0, 0))
        comparison.paste(mask_rgb, (size[0], 0))
        comparison.paste(generated_image, (size[0] * 2, 0))
        
        # Save comparison
        comparison.save(output_path)
        logger.info(f"Saved comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SDXL LoRA Inference for Image Conditional Generation")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to trained LoRA weights")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input composite image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--panels_to_inpaint", type=int, nargs="+", default=[1, 2, 3], 
                       help="Panel indices to inpaint (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)")
    parser.add_argument("--output_dir", type=str, default="inference_output", help="Output directory")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    parser.add_argument("--strength", type=float, default=1.0, help="Denoising strength")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA adapter scale/strength (0-1)")
    parser.add_argument("--base_model", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", help="Base SDXL model")
    parser.add_argument("--save_comparison", action="store_true", help="Save comparison image")
    
    args = parser.parse_args()
    
    # Initialize inference pipeline
    inference = SDXLLoRAInference(
        base_model_id=args.base_model,
        lora_path=args.lora_path
    )
    
    # Generate images
    generated_images = inference.generate_conditional(
        image=args.input_image,
        panels_to_inpaint=args.panels_to_inpaint,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images,
        seed=args.seed,
        strength=args.strength,
        lora_scale=args.lora_scale
    )
    
    # Save generated images
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, generated_image in enumerate(generated_images):
        output_path = output_dir / f"generated_{i}.png"
        generated_image.save(output_path)
        logger.info(f"Saved generated image to {output_path}")
        
        # Save comparison if requested
        if args.save_comparison:
            original_image = Image.open(args.input_image)
            mask = inference.create_panel_mask(
                original_image.size, 
                args.panels_to_inpaint
            )
            comparison_path = output_dir / f"comparison_{i}.png"
            inference.save_comparison(
                original_image,
                generated_image,
                mask,
                comparison_path
            )


if __name__ == "__main__":
    main() 