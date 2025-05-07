import yaml
from skinpy import Skin, Perspective
from PIL import Image
import argparse
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_composite(skin_path, target_path, output_path, config):
    # Load and convert skin image to RGBA
    img = Image.open(skin_path).convert('RGBA')
    skin = Skin.from_image(img)
    
    # Create perspectives and generate isometric views
    perspectives = [Perspective(**p) for p in config['perspectives']]
    skins = []
    for p in perspectives:
        isometric = skin.to_isometric_image(p)
        isometric.thumbnail((int(config['image']['width']/4), int(config['image']['height']/2)))
        skins.append(isometric)

    # Create composite image
    width, height = config['image']['width'], config['image']['height']
    training_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    # Paste target image (scaled to half width)
    training_image.paste(Image.open(target_path).resize((width//2, height//2)), (0, 0))

    # Paste isometric views
    training_image.paste(skins[0], (width//2, 0))
    training_image.paste(skins[1], (3*width//4, 0))
    training_image.paste(skins[2], (0, height//2))
    training_image.paste(skins[3], (width//4, height//2))

    # Paste UV map in bottom right
    training_image.paste(img.resize((width//2, height//2), Image.BOX), (width//2, height//2))

    # Save composite
    training_image.save(os.path.join(output_path, "composite.png"))

def main():
    parser = argparse.ArgumentParser(description='Generate composite images from Minecraft skins')
    parser.add_argument('--config', type=str, default='../../configs/dataset.yaml',
                      help='Path to the configuration file')
    config = load_config(parser.parse_args().config)
    
    generate_composite(
        config['directories']['skin_directory'],
        config['directories']['target_directory'],
        config['directories']['output_directory'],
        config
    )

if __name__ == "__main__":
    main()
