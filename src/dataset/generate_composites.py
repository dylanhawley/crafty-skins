import yaml
from skinpy import Skin, Perspective
from PIL import Image
import argparse
import os
import glob

class CompositeGenerator:
    def __init__(self, width, height, skin_dir, target_dir, output_dir, perspectives):
        self.width = width
        self.height = height
        self.skin_dir = skin_dir
        self.target_dir = target_dir
        self.output_dir = output_dir
        self.perspectives = perspectives
        
    @classmethod
    def from_yaml(cls, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(
            width=config['image']['width'],
            height=config['image']['height'],
            skin_dir=config['directories']['skin'],
            target_dir=config['directories']['target'],
            output_dir=config['directories']['output'],
            perspectives=[Perspective(**p) for p in config['perspectives']],
        )
        
    def generate_composite(self, skin_path, target_path):
        uv_map = Image.open(skin_path)
        skin = Skin.from_image(uv_map.convert('RGBA'))
        
        # Generate composite image
        composite = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        
        # Paste target and isometric views
        composite.paste(Image.open(target_path).resize((self.width//2, self.height//2)), (0, 0))
        for i, p in enumerate(self.perspectives):
            isometric = skin.to_isometric_image(p)
            isometric.thumbnail((self.width//4, self.height//2))
            x = (self.width//2 if i < 2 else 0) + (self.width//4 if i % 2 else 0)
            y = self.height//2 if i >= 2 else 0
            composite.paste(isometric, (x, y))
        
        # Add UV map
        composite.paste(uv_map.resize((self.width//2, self.height//2), Image.BOX), (self.width//2, self.height//2))
        
        # Create output filename based on input filename
        base_name = os.path.splitext(os.path.basename(skin_path))[0]
        output_file = os.path.join(self.output_dir, f"{base_name}_composite.png")
        composite.save(output_file)
        print(f"Generated composite for {base_name}")
        
    def process_all_pairs(self):
        if not os.path.exists(self.output_dir):
            raise FileNotFoundError(f"Output directory '{self.output_dir}' does not exist")
        
        # Get all skin files
        skin_files = glob.glob(os.path.join(self.skin_dir, "*.png"))
        
        for skin_path in skin_files:
            base_name = os.path.splitext(os.path.basename(skin_path))[0]
            target_path = os.path.join(self.target_dir, f"{base_name}.png")
            
            # Check if matching target file exists
            if os.path.exists(target_path):
                self.generate_composite(skin_path, target_path)
            else:
                print(f"Warning: No matching target file found for {base_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate composite images from Minecraft skins')
    parser.add_argument('--config', type=str, default='../../configs/dataset.yaml')
    args = parser.parse_args()
    
    generator = CompositeGenerator.from_yaml(args.config)
    generator.process_all_pairs()
