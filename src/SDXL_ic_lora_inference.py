import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLInpaintPipeline, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import PeftModel
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
        base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
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
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
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
        
        self.pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.base_model_id,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
            variant="fp16" if self.torch_dtype == torch.float16 else None
        )
        
        # Use DDIM scheduler for better quality
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        self.pipeline.to(self.device)
        
        # Enable memory efficient attention if available
        if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
            self.pipeline.enable_xformers_memory_efficient_attention()
        
        # Enable model CPU offload for memory efficiency
        if self.device == "cuda":
            self.pipeline.enable_model_cpu_offload()
    
    def _load_lora(self):
        """Load the trained LoRA weights."""
        logger.info(f"Loading LoRA weights from {self.lora_path}")
        
        # Load LoRA weights
        self.pipeline.load_lora_weights(self.lora_path)
        
        # Load training config if available
        config_path = Path(self.lora_path) / "training_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.lora_config = json.load(f)
                logger.info(f"Loaded LoRA config: {self.lora_config}")
    
    def create_panel_mask(
        self, 
        image_size: Tuple[int, int], 
        panels_to_mask: List[int]
    ) -> Image.Image:
        """
        Create a mask for specific panels in a 2x2 grid.
        
        Args:
            image_size (Tuple[int, int]): Size of the image (width, height)
            panels_to_mask (List[int]): List of panel indices to mask (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)
        
        Returns:
            Image.Image: Binary mask image (white=keep, black=inpaint)
        """
        width, height = image_size
        mask = Image.new("L", (width, height), 255)  # White (keep)
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
        
        # Mask selected panels (black = inpaint)
        for panel_idx in panels_to_mask:
            if 0 <= panel_idx < len(panel_coords):
                coords = panel_coords[panel_idx]
                draw.rectangle(coords, fill=0)  # Black (inpaint)
        
        return mask
    
    def create_gradient_mask(
        self, 
        image_size: Tuple[int, int], 
        panels_to_mask: List[int],
        gradient_width: int = 20
    ) -> Image.Image:
        """
        Create a soft gradient mask for smoother transitions between panels.
        
        Args:
            image_size (Tuple[int, int]): Size of the image (width, height)
            panels_to_mask (List[int]): List of panel indices to mask
            gradient_width (int): Width of the gradient transition
        
        Returns:
            Image.Image: Gradient mask image
        """
        width, height = image_size
        mask = np.ones((height, width), dtype=np.float32) * 255
        
        panel_width = width // 2
        panel_height = height // 2
        
        # Define panel coordinates
        panel_coords = [
            (0, 0, panel_width, panel_height),  # top-left
            (panel_width, 0, width, panel_height),  # top-right
            (0, panel_height, panel_width, height),  # bottom-left
            (panel_width, panel_height, width, height)  # bottom-right
        ]
        
        for panel_idx in panels_to_mask:
            if 0 <= panel_idx < len(panel_coords):
                x1, y1, x2, y2 = panel_coords[panel_idx]
                
                # Create soft transition at edges
                for y in range(max(0, y1 - gradient_width), min(height, y2 + gradient_width)):
                    for x in range(max(0, x1 - gradient_width), min(width, x2 + gradient_width)):
                        if x1 <= x < x2 and y1 <= y < y2:
                            # Inside panel - fully masked
                            mask[y, x] = 0
                        else:
                            # Outside panel - create gradient
                            dist_to_panel = min(
                                abs(x - x1) if x < x1 else (abs(x - x2) if x >= x2 else 0),
                                abs(y - y1) if y < y1 else (abs(y - y2) if y >= y2 else 0)
                            )
                            if dist_to_panel < gradient_width:
                                alpha = dist_to_panel / gradient_width
                                mask[y, x] = min(mask[y, x], alpha * 255)
        
        return Image.fromarray(mask.astype(np.uint8), mode='L')
    
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
        use_gradient_mask: bool = True,
        gradient_width: int = 20,
        strength: float = 1.0
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
            use_gradient_mask (bool): Whether to use gradient mask for smoother transitions
            gradient_width (int): Width of gradient transition
            strength (float): Denoising strength (0-1)
        
        Returns:
            List[Image.Image]: Generated images
        """
        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Create mask
        if use_gradient_mask:
            mask = self.create_gradient_mask(
                image.size, 
                panels_to_inpaint, 
                gradient_width
            )
        else:
            mask = self.create_panel_mask(image.size, panels_to_inpaint)
        
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
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
    parser.add_argument("--panels_to_inpaint", type=int, nargs="+", default=[2, 3], 
                       help="Panel indices to inpaint (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)")
    parser.add_argument("--output_dir", type=str, default="inference_output", help="Output directory")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--use_gradient_mask", action="store_true", help="Use gradient mask for smooth transitions")
    parser.add_argument("--gradient_width", type=int, default=20, help="Width of gradient transition")
    parser.add_argument("--strength", type=float, default=1.0, help="Denoising strength")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Base SDXL model")
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
        use_gradient_mask=args.use_gradient_mask,
        gradient_width=args.gradient_width,
        strength=args.strength
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