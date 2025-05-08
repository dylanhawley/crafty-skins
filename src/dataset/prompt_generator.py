import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import os
from openai import OpenAI

class PromptGenerator:
    def __init__(self, metadata_filepath, output_dir, openai_api_key=None):
        with open(metadata_filepath, 'r') as f:
            self.metadata = json.load(f)
        self.output_dir = Path(output_dir)
        self.client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        
    @classmethod
    def from_yaml(cls, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(
            metadata_filepath=config['paths']['metadata'],
            output_dir=config['paths']['output_dir'],
            openai_api_key=config.get('openai', {}).get('api_key')
        )
        
    def enhance_prompt(self, text):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a prompt engineering expert. Your task is to enhance the given prompt to be more detailed and descriptive while maintaining the same structure and meaning. Focus on adding rich visual details and specific elements that would help in generating high-quality images. The prompt should follow the style of the In-Context LoRA paper, where the prompts generate image sets, rather than a single image. A placeholder character name uniquely references the character's identity across the images. Here are five example prompts: 1. “In this adventurous three-image sequence, [IMAGE1] Ethan, an intrepid archaeologist with a rugged appearance, uncovers an ancient map in a sunlit desert dig site, his excitement palpable as he brushes away the sand, [IMAGE2] transitioning to a bustling marketplace in a vibrant foreign city where Ethan negotiates with local merchants and gathers essential supplies for his quest, [IMAGE3] and finally, Ethan treks through a dense, mist-covered jungle, the towering trees and exotic wildlife emphasizing the challenges and mysteries that lie ahead on his journey.” 2. “This set of four images showcases a teenage girl with curly black hair wearing a stylish denim jacket, each image highlighting her dynamic personality in urban settings; [IMAGE1] she is skateboarding down a graffiti-covered alley, a confident smile on her face as she maneuvers around obstacles; [IMAGE2] she is seated at a trendy café, typing on her laptop with focused determination, the bustling city life visible through the large windows behind her; [IMAGE3] she stands on a rooftop at sunset, her hair blowing in the breeze as she gazes thoughtfully over the city skyline; and [IMAGE4] she is laughing with friends at a vibrant street market, colorful lights and stalls creating a lively atmosphere around her.” 3. “The set of four images highlights the playful energy of a young boy in a city playground. [IMAGE1] He climbs up a jungle gym with a look of determination, his hands gripping the bars as he pulls himself up; [IMAGE2] he swings high on a set of swings, his head thrown back in laughter as his feet touch the sky; [IMAGE3] a close-up captures him mid-slide, his eyes wide with excitement as he descends down a bright yellow slide; [IMAGE4] he races down a pathway lined with trees, his arms pumping with energy as he chases after a soccer ball, his face alight with joy.” 4. “The set of four images showcases a young girl exploring a cozy kitchen setting with her mother, filled with warmth and affection. [IMAGE1] She stands on a stool, her hands reaching into a bowl of cookie dough as her mother smiles beside her; [IMAGE2] she's caught mid-laugh, flour dusted across her cheeks as she playfully tosses a bit of dough in the air; [IMAGE3] the scene focuses on her concentration as she carefully uses cookie cutters, her tiny hands pressing down on the dough; [IMAGE4] she proudly holds up a finished tray of cookies, her face beaming with joy and accomplishment.” 5. “This set of four images captures the serene moments of an elderly woman tending to her garden. [IMAGE1] She kneels beside a bed of blooming flowers, her hands gently pruning a rose bush, the soft morning light illuminating her silver hair; [IMAGE2] she stands with a watering can, her face calm and peaceful as she nurtures her plants; [IMAGE3] a close-up reveals her content smile as she examines a budding flower in her hand, a sense of pride and joy evident; [IMAGE4] she sits on a small bench, sipping tea with her garden behind her, surrounded by the vibrant colors of her hard work.”"},
                    {"role": "user", "content": f"Please enhance this prompt for a four-panel image, and please use the provided schema (such as [TOP-LEFT]) to refer to image panels, do NOT use [IMAGE#] format. Respond with only your enhanced prompt. The prompt to enhance is: This four-panel image showcases a transformation from a photorealistic image to a Minecraft skin; [TOP-LEFT] features our <subject>, which has the description {text}; [TOP-RIGHT] shows the left front up and right back up isometric positions of the <subject> as a Minecraft avatar; [BOTTOM-LEFT] displays the left back down and right front down isometric positions of the <subject> as a Minecraft avatar; [BOTTOM-RIGHT] is a complete uv map for the Minecraft skin file constructed from the isometric views of our <subject> laid out in the previous two panels."}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error enhancing prompt: {e}")
            return text
        
    def generate_prompt(self, text, output):
        enhanced_text = self.enhance_prompt(text)
        with open(self.output_dir / output, 'w') as f:
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
