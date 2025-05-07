import yaml
from skinpy import Skin, Perspective
from PIL import Image
import argparse
from pathlib import Path

class CompositeGenerator:
    def __init__(self, width, height, skin_dir, target_dir, output_dir, perspectives):
        self.width, self.height = width, height
        self.skin_dir = Path(skin_dir)
        self.target_dir = Path(target_dir)
        self.output_dir = Path(output_dir)
        self.perspectives = [Perspective(**p) for p in perspectives]
        
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
            perspectives=config['perspectives'],
        )
        
    def generate_composite(self, skin_path, target_path):
        uv_map = Image.open(skin_path)
        skin = Skin.from_image(uv_map.convert('RGBA'))
        composite = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        composite.paste(Image.open(target_path).resize((self.width//2, self.height//2)), (0, 0))
        for i, p in enumerate(self.perspectives):
            isometric = skin.to_isometric_image(p)
            isometric.thumbnail((self.width//4, self.height//2))
            x = (self.width//2 if i < 2 else 0) + (self.width//4 if i % 2 else 0)
            y = self.height//2 if i >= 2 else 0
            composite.paste(isometric, (x, y))
        composite.paste(uv_map.resize((self.width//2, self.height//2), Image.BOX), (self.width//2, self.height//2))
        composite.save(self.output_dir / f"{Path(skin_path).stem}.png")
        
    def process_all_pairs(self):
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Output directory '{self.output_dir}' does not exist")
            
        for skin_path in self.skin_dir.glob("*.png"):
            target_path = self.target_dir / f"{skin_path.stem}.png"
            if target_path.exists():
                self.generate_composite(skin_path, target_path)
            else:
                print(f"Warning: No matching target file found for {skin_path.stem}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate composite images from Minecraft skins')
    parser.add_argument('--config', type=str, default='../../configs/dataset.yaml')
    args = parser.parse_args()
    
    generator = CompositeGenerator.from_yaml(args.config)
    generator.process_all_pairs()
