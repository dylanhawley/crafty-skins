from diffusers import StableDiffusionXLImg2ImgPipeline
import torch
from PIL import Image
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate Minecraft skin using SDXL model')
    parser.add_argument('--model_path', type=str, default='finetuned-sdxl',
                        help='Path to the finetuned SDXL model')
    parser.add_argument('--input_image', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('--output_image', type=str, default='output.png',
                        help='Path to save the output image')
    parser.add_argument('--prompt', type=str, default='A minecraft character skin',
                        help='Text prompt for image generation')
    parser.add_argument('--strength', type=float, default=0.75,
                        help='How much to transform the input image (0-1)')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Guidance scale for the diffusion model')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of CUDA')
    
    args = parser.parse_args()
    
    # Determine device
    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"Loading model from {args.model_path}...")
    
    # Load the finetuned model
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    # Load your input image
    print(f"Loading input image from {args.input_image}...")
    try:
        input_image = Image.open(args.input_image)
    except Exception as e:
        print(f"Error loading input image: {e}")
        return
    
    # Generate image
    print(f"Generating image with prompt: '{args.prompt}'...")
    output_image = pipeline(
        prompt=args.prompt,
        image=input_image,
        strength=args.strength,
        guidance_scale=args.guidance_scale
    ).images[0]
    
    # Save the output
    output_image.save(args.output_image)
    print(f"Output saved to {args.output_image}")

if __name__ == "__main__":
    main() 