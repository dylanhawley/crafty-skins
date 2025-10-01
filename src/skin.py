from scipy.spatial.distance import cdist
from pathlib import Path
from PIL import Image
import numpy as np

from inference_common import (
    create_inference_parser, 
    setup_device_and_dtype, 
    create_pipeline, 
    generate_image, 
    setup_logging
)

MASK_IMAGE = "data/half-transparency-mask.png"

# BACKGROUND_REGIONS is an array containing all of the areas that contain no pixels
# that are used in rendering the skin.  We'll use these areas to figure out the 
# color used to represent transparency in the skin.
BACKGROUND_REGIONS = [
    (32, 0, 40, 8),
    (56, 0, 64, 8)
]

# TRANSPARENT_REGIONS is an array containing all of the areas of the skin that need
# to have transparency restored.  Refer to: https://github.com/minotar/skin-spec for
# more information.
TRANSPARENT_REGIONS = [
        (40, 0, 48, 8),
        (48, 0, 56, 8),
        (32, 8, 40, 16),
        (40, 8, 48, 16),
        (48, 8, 56, 16),
        (56, 8, 64, 16)    
]

def get_background_color(image):
    '''
    Given a Minecraft skin image, loop over all of the regions considered to be the
    background, or ones that don't get rendered into a skin, and find the average
    color.  This color will be used when restoring transparency to the second layer.
    '''
    pixels = []
    
    # Loop over all the transparent regions, and create a list of the 
    # constituent pixels
    for region in BACKGROUND_REGIONS:
        swatch = image.crop(region)
        
        width, height = swatch.size
        np_swatch = np.array(swatch)

        # Reshape so that we have an list of pixel arrays.
        np_swatch = np_swatch.reshape(width * height, 3)

        if len(pixels) == 0:
            pixels = np_swatch
        else:
            np.concatenate((pixels, np_swatch))

    # Get the mean RGB values for the pixels in the background regions.
    (r, g, b) = np.mean(np_swatch, axis=0, dtype=int)
       
    return [(r, g, b)]

def restore_region_transparency(image, region, transparency_color, cutoff=50):
    changed = 0
    # Loop over all the pixels in the region we're processing.
    for x in range(region[0], region[2]):
        for y in range(region[1], region[3]):
            pixel = [image.getpixel((x, y))]
            pixel = [(pixel[0][0], pixel[0][1], pixel[0][2])]
          
            # Calculate the Cartesian distance between the current pixel and the
            # transparency color.
            dist  = cdist(pixel, transparency_color)
           
            # If the distance is less than or equal to the cutoff, then set the
            # pixel as transparent.
            if dist <= cutoff:
                image.putpixel((x, y), (255, 255, 255, 0))
                changed = changed + 1

    return image, changed

def restore_skin_transparency(image, transparency_color, cutoff=50):
    # Convert the generated RGB image back to RGBA to restore transparency.
    image = image.convert("RGBA")

    total_changed = 0
    # Restore transparency in each region.
    for region in TRANSPARENT_REGIONS:
        image, changed = restore_region_transparency(image, region, transparency_color, cutoff=cutoff)
        total_changed = total_changed + changed
        
    return image, total_changed

def extract_minecraft_skin(generated_image, width, height, cutoff=50):
    # Crop out the skin portion from the  generated file.
    image = generated_image.crop((0, 0, width, int(height/2)))

    # Scale the image down to the 64x32 size.
    skin = image.resize((64, 32), Image.NEAREST)

    # Get the average background transparency color from the skin.  We'll use this
    # later when we need to determine which cluster corresponds to the background
    # pixels.
    color = get_background_color(skin)

    # Restore the transparent parts in the skin background.
    transparent_skin, _ = restore_skin_transparency(skin, color, cutoff=cutoff)
    
    # Convert the bits of the background that aren't involved with transparency
    # to all white.
    mask = Image.open(MASK_IMAGE)
    transparent_skin.alpha_composite(mask)

    return transparent_skin

def main(args, logger):
    # Setup device and dtype
    device, dtype = setup_device_and_dtype()
    if logger:
        if device == "cuda":
            logger.info("CUDA device found, enabling.")
        elif device == "mps":
            logger.info("Apple MPS device found, enabling.")
        else:
            logger.info("No CUDA or MPS devices found, running on CPU.")

    # Create pipeline
    pipeline = create_pipeline(args.model_name, device, dtype, logger)

    # Generate the image
    generated_image = generate_image(pipeline, args, logger)

    # Extract and scale down the Minecraft skin portion of the image.
    logger.info("Extracting and scaling Minecraft skin from generated image.")
    minecraft_skin = extract_minecraft_skin(generated_image, args.width, args.height, cutoff=50)

    # Save the generated image to args.output
    logger.info("Saving generated image to: '{}'.".format(args.output))
    generated_image.save(args.output)
    
    # Generate filename for minecraft skin with "_skin" before extension
    output_path = Path(args.output)
    skin_filename = output_path.stem + "_skin" + output_path.suffix
    skin_path = output_path.parent / skin_filename
    
    logger.info("Saving minecraft skin to: '{}'.".format(skin_path))
    minecraft_skin.save(skin_path)
    
if __name__ == "__main__":
    parser = create_inference_parser()
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    main(args, logger)
    