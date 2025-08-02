from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import datetime

def create_experiment_folder(experiment_name):
    """Create and return the experiment folder structure."""
    # Convert experiment name to lowercase and replace spaces with underscores
    formatted_name = experiment_name.lower().replace(" ", "_")
    
    # Create main experiment folder
    experiment_dir = Path(f"experiments/{formatted_name}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figures subfolder
    figures_dir = experiment_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    return experiment_dir, figures_dir

def get_output_path(input_path, scale, figures_dir):
    """Generate a structured filename for the output image."""
    input_stem = Path(input_path).stem
    scale_str = f"{scale:.1f}".replace(".", "_")
    return figures_dir / f"{input_stem}_scale_{scale_str}.png"

def create_output_row(pipeline, input_image_path, prompt, generator, scales, figures_dir, num_inference_steps):
    # Load and resize input image to match output size
    input_image = load_image(input_image_path)
    input_image = input_image.resize((512, 512))
    
    # Create a list to store all images in the row
    row_images = [np.array(input_image)]
    
    # Generate outputs for each scale
    for scale in scales:
        output_path = get_output_path(input_image_path, scale, figures_dir)
        
        # Skip if output already exists
        if not output_path.exists():
            pipeline.set_ip_adapter_scale(scale)
            output = pipeline(
                prompt=prompt,
                ip_adapter_image=input_image,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
            output.save(str(output_path))
        
        # Add a small delay to ensure file is fully written
        # time.sleep(1)
        
        # Load the saved image
        print(output_path)
        saved_image = load_image(str(output_path))
        row_images.append(np.array(saved_image))
    
    return row_images

def create_output_table(pipeline, input_image_paths, prompt, scales, figures_dir, num_inference_steps):
    # Initialize generator
    generator = torch.Generator(device="mps").manual_seed(26)
    
    # Create all rows
    all_rows = []
    for input_path in input_image_paths:
        row = create_output_row(pipeline, input_path, prompt, generator, scales, figures_dir, num_inference_steps)
        all_rows.append(row)
    
    # Calculate dimensions
    num_cols = len(scales) + 1  # +1 for input image
    num_rows = len(input_image_paths)
    
    # Create the figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot all images
    for row_idx, row in enumerate(all_rows):
        for col_idx, img in enumerate(row):
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.axis('off')
            
            # Add scale labels to the top row
            if row_idx == 0:
                if col_idx == 0:
                    ax.set_title('Input Image', fontsize=35)
                else:
                    ax.set_title(f'Scale: {scales[col_idx-1]:.1f}', fontsize=35)
    
    plt.tight_layout()
    
    fpath = Path(experiment_name.lower().replace(" ", "_"))
    # Save the composite figure
    composite_path = figures_dir.parent / fpath
    plt.savefig(composite_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return composite_path

def create_experiment_documentation(experiment_dir, experiment_name, input_image_paths, scales, prompt, start_time, end_time, composite_path, sdxl_model, ip_adapter_model, num_inference_steps):
    """Create a markdown file documenting the experiment."""
    doc_path = experiment_dir / "experiment_details.md"
    
    with open(doc_path, "w") as f:
        f.write(f"# {experiment_name}\n\n")
        f.write(f"## Experiment Details\n\n")
        f.write(f"- **Start Time**: {start_time}\n")
        f.write(f"- **End Time**: {end_time}\n")
        f.write(f"- **Duration**: {end_time - start_time}\n\n")
        
        f.write("## Models Used\n\n")
        f.write(f"- **SDXL Model**: {sdxl_model}\n")
        f.write(f"- **IP-Adapter Model**: {ip_adapter_model}\n\n")
        
        f.write("## Input Images\n\n")
        for img_path in input_image_paths:
            f.write(f"- {img_path}\n")
        
        f.write("\n## Parameters\n\n")
        f.write(f"- **Scales Tested**: {scales}\n")
        f.write(f"- **Prompt**: {prompt}\n")
        f.write(f"- **Number of Inference Steps**: {num_inference_steps}\n")
        
        f.write("\n## Results\n\n")
        f.write(f"![Composite Figure](figures/ip_adapter_scale_comparison.png)\n")

def run_experiment(experiment_name, input_image_paths, prompt, scales, num_inference_steps=5):
    """Run the complete experiment with documentation."""
    # Create experiment folder structure
    experiment_dir, figures_dir = create_experiment_folder(experiment_name)
    
    # Model configurations
    sdxl_model = "monadical-labs/minecraft-skin-generator-sdxl"
    ip_adapter_model = "h94/IP-Adapter"
    
    # Initialize pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        sdxl_model, 
        torch_dtype=torch.float16
    ).to("mps")
    pipeline.load_ip_adapter(
        ip_adapter_model, 
        subfolder="sdxl_models", 
        weight_name="ip-adapter_sdxl.bin"
    )
    
    # Record start time
    start_time = datetime.datetime.now()
    
    # Generate outputs
    composite_path = create_output_table(pipeline, input_image_paths, prompt, scales, figures_dir, num_inference_steps)
    
    # Record end time
    end_time = datetime.datetime.now()
    
    # Create documentation
    create_experiment_documentation(
        experiment_dir,
        experiment_name,
        input_image_paths,
        scales,
        prompt,
        start_time,
        end_time,
        composite_path,
        sdxl_model,
        ip_adapter_model,
        num_inference_steps
    )
    
    return experiment_dir

if __name__ == "__main__":
    # Define experiment parameters
    experiment_name = "IP Adapter Scale Test 100 Steps"
    input_image_paths = [
        "data/raw/input/13thdoctor.png",
        "data/raw/input/3dglasses.png",
        "data/raw/input/alberteinstein.png",
        "data/raw/input/balloonboy.png",
        "data/raw/input/animalcrossingvillager.png",
        "data/raw/input/arabman.png",
        "data/raw/input/berlioz.png",
        "data/raw/input/countrykitty.png",
        "data/raw/input/creeperboy.png",
        "data/raw/input/daughterofevil.png",
        "data/raw/input/dawnpokemon.png",
        "data/raw/input/demonicmutation.png",
        "data/raw/input/doctorstrange.png",
        "data/raw/input/donaldsuit.png",
        "data/raw/input/doratheexplorer.png",
        "data/raw/input/frontman.png"
    ]
    prompt = ""
    scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    num_inference_steps = 100
    
    # Run the experiment
    experiment_dir = run_experiment(experiment_name, input_image_paths, prompt, scales, num_inference_steps)
    print(f"Experiment completed. Results saved in: {experiment_dir}")
