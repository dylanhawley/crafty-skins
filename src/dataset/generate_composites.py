import yaml
from skinpy import Skin, Perspective
from PIL import Image
import argparse
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_composite(skin_path, target_path, output_path, config):
    # Convert image to RGBA mode before creating Skin object
    img = Image.open(skin_path)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    skin = Skin.from_image(img)
    
    # Create perspectives from config
    perspectives = [Perspective(**p) for p in config['perspectives']]

    # Load the target image and scale it to 1/2 of config['image']['width']
    target_image = Image.open(target_path)
    scaled_target_image = target_image.resize((int(config['image']['width']/2), int(config['image']['height']/2)))
    
    skins = [skin.to_isometric_image(perspective) for perspective in perspectives]

    # Scale each skin image to 1/2 of config['image']['height'], and up to 1/4 of config['image']['width'], preserving original aspect ratio
    for skin in skins:
        skin = skin.thumbnail((int(config['image']['width']/4), int(config['image']['height']/2)))
    # Create the composite image of our skin plus the rendered character images.
    training_image = Image.new('RGBA', (config['image']['width'], config['image']['height']), (0, 0, 0, 0))

    training_image.paste(scaled_target_image, (0,0))

    # Paste the first two skins in the top right panel of the training image
    training_image.paste(skins[0], (int(config['image']['width']/2), 0))
    training_image.paste(skins[1], (3*int(config['image']['width']/4), 0))

    # Paste the last two skins in the bottom left panel of the training image
    training_image.paste(skins[2], (0, int(config['image']['height']/2)))
    training_image.paste(skins[3], (int(config['image']['width']/4), int(config['image']['height']/2)))

    # Paste the img in the bottom right panel of the training image, scaled to fit
    scaled_uv_map = img.resize((int(config['image']['width']/2), int(config['image']['height']/2)), resample=Image.BOX)
    training_image.paste(scaled_uv_map, (int(config['image']['width']/2), int(config['image']['height']/2)))

    # Save the composite image with .png extension
    output_file = os.path.join(output_path, "composite.png")
    training_image.save(output_file)

def main():
    parser = argparse.ArgumentParser(description='Generate composite images from Minecraft skins')
    parser.add_argument('--config', type=str, default='../../configs/dataset.yaml',
                      help='Path to the configuration file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    generate_composite(
        config['directories']['skin_directory'],
        config['directories']['target_directory'],
        config['directories']['output_directory'],
        config
    )

if __name__ == "__main__":
    main()
