import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import json

class PromptGenerator:
    def __init__(self, metadata_filepath, output_dir):
        self.metadata = []
        with open(metadata_filepath, 'r') as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))
        self.output_dir = Path(output_dir)
        
        # Check if output directory exists, prompt user to create if not
        if not self.output_dir.exists():
            response = input(f"Output directory '{self.output_dir}' does not exist. Create it? (Y/n): ").strip().lower()
            if response in ['', 'y', 'yes']:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created output directory: {self.output_dir}")
            else:
                raise FileNotFoundError(f"Output directory '{self.output_dir}' does not exist and user chose not to create it.")
        
    @classmethod
    def from_yaml(cls, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(
            metadata_filepath=config['paths']['metadata'],
            output_dir=config['paths']['output_dir']
        )
        
    def enhance_prompt(self, text):
        return f"a four-panel image; [TOP-LEFT] features our <subject>, {text}; [TOP-RIGHT] shows isometric positions of the <subject> skin; [BOTTOM-LEFT] shows isometric positions of the <subject> skin; [BOTTOM-RIGHT] is a uv map for the Minecraft skin."
        
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
