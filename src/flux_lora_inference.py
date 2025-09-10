import torch
from diffusers import FluxFillPipeline
from PIL import Image
from typing import Optional, Union, List
import os

class FluxFillInference:
    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-Fill-dev",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0
    ):
        """
        Initialize the Flux Fill inference pipeline.
        
        Args:
            model_id (str): The model ID to use for inference
            device (Optional[str]): The device to run inference on. If None, will automatically select the best available device.
            torch_dtype (torch.dtype): The torch data type to use
            lora_path (Optional[str]): Path to the LoRA weights file
            lora_scale (float): Scale factor for LoRA weights
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
        
        # Initialize the pipeline
        self.pipeline = FluxFillPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            # variant="fp16"
        )
        self.pipeline.to(device)
        
        # Load LoRA if provided
        if lora_path is not None:
            self.load_lora(lora_path, lora_scale)
        
    def load_lora(self, lora_path: str, lora_scale: float = 1.0):
        """
        Load LoRA weights into the pipeline.
        
        Args:
            lora_path (str): Path to the LoRA weights file
            lora_scale (float): Scale factor for LoRA weights
        """
        self.pipeline.load_lora_weights(lora_path)
        self.pipeline.fuse_lora(lora_scale)
        
    def generate(
        self,
        image: Union[str, Image.Image],
        mask: Union[str, Image.Image],
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate inpainted images using the Flux Fill pipeline.
        
        Args:
            image (Union[str, Image.Image]): Input image or path to image
            mask (Union[str, Image.Image]): Mask image or path to mask
            prompt (str): Text prompt for generation
            negative_prompt (Optional[str]): Negative prompt for generation
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for classifier-free guidance
            num_images_per_prompt (int): Number of images to generate per prompt
            seed (Optional[int]): Random seed for reproducibility
            
        Returns:
            List[Image.Image]: List of generated images
        """
        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(mask, str):
            mask = Image.open(mask).convert("L")
            
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            
        # Generate images
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
        )
        
        return output.images
    
    def save_images(
        self,
        images: List[Image.Image],
        output_dir: str,
        prefix: str = "generated"
    ) -> List[str]:
        """
        Save generated images to disk.
        
        Args:
            images (List[Image.Image]): List of images to save
            output_dir (str): Directory to save images in
            prefix (str): Prefix for saved image filenames
            
        Returns:
            List[str]: List of paths to saved images
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        for i, image in enumerate(images):
            output_path = os.path.join(output_dir, f"{prefix}_{i}.png")
            image.save(output_path)
            saved_paths.append(output_path)
            
        return saved_paths

# Example usage
if __name__ == "__main__":
    # Initialize the pipeline
    flux_fill = FluxFillInference(lora_path="/Users/dylanhawley/Projects/crafty-skins/data/minecraft-DiT-In-Context-LoRA_000002250.safetensors")
    
    # Example parameters
    image_path = "/Users/dylanhawley/Projects/crafty-skins/data/processed/1024x1024/13thdoctor.png"
    mask_path = "/Users/dylanhawley/Projects/crafty-skins/data/quadrant_mask.png"
    prompt = "This four-panel image showcases a transformation from a photorealistic image to a Minecraft skin; [TOP-LEFT] features our <subject>, which has the description The 13th Doctor from the British television series Doctor Who.; [TOP-RIGHT] shows the left front up and right back up isometric positions of the <subject> as a Minecraft avatar; [BOTTOM-LEFT] displays the left back down and right front down isometric positions of the <subject> as a Minecraft avatar; [BOTTOM-RIGHT] is a complete uv map for the Minecraft skin file constructed from the isometric views of our <subject> laid out in the previous two panels."
    # negative_prompt = "blurry, low quality, distorted"
    
    # Generate images
    generated_images = flux_fill.generate(
        image=image_path,
        mask=mask_path,
        prompt=prompt,
        # negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        num_images_per_prompt=1,
        seed=42
    )
    
    # Save generated images
    saved_paths = flux_fill.save_images(
        images=generated_images,
        output_dir="/Users/dylanhawley/Projects/crafty-skins/data/output",
        prefix="flux_fill"
    )
    
    print(f"Generated images saved to: {saved_paths}")
