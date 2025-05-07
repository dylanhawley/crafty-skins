import yaml
from skinpy import Skin, Perspective
from PIL import Image
import argparse
import os

def generate_composite(config):
    skin = Skin.from_image(Image.open(config['directories']['skin']).convert('RGBA'))
    perspectives = [Perspective(**p) for p in config['perspectives']]
    
    # Generate composite image
    width, height = config['image']['width'], config['image']['height']
    composite = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Paste target and isometric views
    composite.paste(Image.open(config['directories']['target']).resize((width//2, height//2)), (0, 0))
    for i, p in enumerate(perspectives):
        isometric = skin.to_isometric_image(p)
        isometric.thumbnail((width//4, height//2))
        x = (width//2 if i < 2 else 0) + (width//4 if i % 2 else 0)
        y = height//2 if i >= 2 else 0
        composite.paste(isometric, (x, y))
    
    # Add UV map
    composite.paste(Image.open(config['directories']['skin']).resize((width//2, height//2), Image.BOX), (width//2, height//2))
    composite.save(os.path.join(config['directories']['output'], "composite.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate composite images from Minecraft skins')
    parser.add_argument('--config', type=str, default='../../configs/dataset.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        generate_composite(yaml.safe_load(f))
