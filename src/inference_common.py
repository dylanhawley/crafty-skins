import torch
import argparse
import logging
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from diffusers.utils import load_image


def create_inference_parser():
    """Create a standardized argument parser for inference scripts."""
    parser = argparse.ArgumentParser(description="Generate images using IP-Adapter with SDXL")
    
    parser.add_argument("--model_name", type=str, default="monadical-labs/minecraft-skin-generator-sdxl",
                       help="Pretrained model name or path")
    
    parser.add_argument("--ip_adapter_scale_style", type=float, default=0.5,
                       help="IP adapter scale for style image (default: 0.5)")
    parser.add_argument("--ip_adapter_scale_face", type=float, default=0.5,
                       help="IP adapter scale for face image (default: 0.5)")
    
    parser.add_argument("--face_image", type=str, required=True,
                       help="Path to face image")
    parser.add_argument("--style_image", type=str, required=True,
                       help="Path to style image")
    
    parser.add_argument("--prompt", type=str, default="",
                       help="Text prompt for generation (default: empty string)")
    parser.add_argument("--negative_prompt", type=str, 
                       default="monochrome, lowres, bad anatomy, worst quality, low quality",
                       help="Negative prompt for generation")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps (default: 50)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Guidance scale for generation (default: 7.5)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for generation (default: 0)")
    parser.add_argument("--height", type=int, default=768,
                       help="Height of generated image (default: 768)")
    parser.add_argument("--width", type=int, default=768,
                       help="Width of generated image (default: 768)")
    
    parser.add_argument("--output", type=str, default="output.png",
                       help="Output image filename (default: output.png)")
    
    parser.add_argument("--no_cpu_offload", action="store_true",
                       help="Disable CPU offload (uses more GPU memory)")
    
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="Produce verbose output while running")
    
    return parser


def setup_device_and_dtype():
    """Setup device and dtype based on available hardware."""
    device = "cpu"
    dtype = torch.float16
    
    if torch.cuda.is_available() and torch.backends.cuda.is_built():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        dtype = torch.float32
    
    return device, dtype


def create_pipeline(model_name, device, dtype, logger=None):
    """Create and configure the inference pipeline."""
    if logger:
        logger.info("Loading HuggingFace model: '{}'.".format(model_name))
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
    )
    
    pipeline = AutoPipelineForText2Image.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        image_encoder=image_encoder,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors","ip-adapter-plus-face_sdxl_vit-h.safetensors"]
    )
    
    return pipeline


def generate_image(pipeline, args, logger=None):
    """Generate an image using the pipeline with the given arguments."""
    # Set IP adapter scales
    pipeline.set_ip_adapter_scale([args.ip_adapter_scale_style, args.ip_adapter_scale_face])
    
    # Enable CPU offload to reduce memory usage (unless disabled)
    if not args.no_cpu_offload:
        pipeline.enable_model_cpu_offload()
    
    # Load images
    face_image = load_image(args.face_image)
    style_image = load_image(args.style_image)
    
    # Create generator
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    
    if logger:
        logger.info("Generating image with prompt: '{}'.".format(args.prompt))
    
    # Generate image
    image = pipeline(
        prompt=args.prompt,
        ip_adapter_image=[style_image, face_image],
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        height=args.height,
        width=args.width,
    ).images[0]
    
    return image


def setup_logging(verbose=False):
    """Setup logging configuration."""
    import sys
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(stream=sys.stdout, level=level, format='[%(asctime)s] %(levelname)s - %(message)s')
    return logging.getLogger("inference")
