import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import os

class PromptGenerator:
    def __init__(self, metadata_filepath, output_dir):
        self.metadata = []
        with open(metadata_filepath, 'r') as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))
        self.output_dir = Path(output_dir)
        
    @classmethod
    def from_yaml(cls, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(
            metadata_filepath=config['paths']['metadata'],
            output_dir=config['paths']['output_dir']
        )
        
    def enhance_prompt(self, text):
        return f"This four-panel image showcases a transformation from a photorealistic image to a Minecraft skin; [TOP-LEFT] features our <subject>, which has the description {text}; [TOP-RIGHT] shows the left front up and right back up isometric positions of the <subject> as a Minecraft avatar; [BOTTOM-LEFT] displays the left back down and right front down isometric positions of the <subject> as a Minecraft avatar; [BOTTOM-RIGHT] is a complete uv map for the Minecraft skin file constructed from the isometric views of our <subject> laid out in the previous two panels."
        
    def generate_prompt(self, text, output):
        enhanced_text = self.enhance_prompt(text)
        with open((self.output_dir / output).with_suffix('.txt'), 'w') as f:
            f.write(enhanced_text)
        
    def process_all_pairs(self):
        for row in tqdm(self.metadata, desc="Generating prompts"):
            self.generate_prompt(row['text'], Path(row['input_file_name']).stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform joint prompt generation of a multi-panel image')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    
    generator = PromptGenerator.from_yaml(args.config)
    generator.process_all_pairs()
