import os 
import argparse
from utils.openai_utils import generate_descriptions_oai
from utils.anthropic_utils import generate_descriptions_anthropic
import json
import glob
from tqdm import tqdm


class GPT4Captioner:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()

    def caption_image(self, image_path, prompt_path, max_tokens=1000, model='gpt-4o'):
        return generate_descriptions_oai(image_path, self.client, prompt_path, max_tokens, model)

class SonnetCaptioner:
    def __init__(self):
        from anthropic import Anthropic
        self.client = Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=os.environ['ANTHROPIC_API_KEY']
        )   

    def caption_image(self, image_path, prompt_path, max_tokens=1000, model='claude-3-5-sonnet-20240620'):
        return generate_descriptions_anthropic(self.client, image_path, prompt_path, max_tokens, model)

def process_directory(captioner, directory, prompt_path):
    for file in tqdm(glob.glob(os.path.join(directory, '*.original.jpg')), desc="Captioning images"):
        caption_data = captioner.caption_image(file, prompt_path)
        image_name = os.path.basename(file).split('.')[0]
        
        # Save caption data to captions.json in the same directory as the image
        captions_file = os.path.join(directory, f'{image_name}.captions.json')
        with open(captions_file, 'a') as f:
            json.dump(caption_data, f)

def main():
    parser = argparse.ArgumentParser(description="Caption images.")
    parser.add_argument('--captioner', type=str, default='anthropic', choices=['openai', 'anthropic'], help="Type of captioner to use")
    parser.add_argument('--dir', type=str, required=True, help="Top-level directory containing images")
    parser.add_argument('--prompt', type=str, default='./captioning_prompt.txt', help="Path to the prompt file")

    args = parser.parse_args()

    if args.captioner == 'openai':
        captioner = GPT4Captioner()
    elif args.captioner == 'anthropic':
        captioner = SonnetCaptioner()
    else:
        raise ValueError(f"Unsupported captioner type: {args.captioner}")

    process_directory(captioner, args.dir, args.prompt)

if __name__ == "__main__":
    main()