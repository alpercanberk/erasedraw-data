import os
import base64
from utils.utils import find_json_in_string
from PIL import Image
from io import BytesIO

def generate_descriptions_oai(image_path, client, prompt_path, max_tokens=600, model='gpt-4o'):

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    # Getting the base64 string

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    }
    print("API key is ", os.environ['OPENAI_API_KEY'])

    text = open(prompt_path, 'r').read()

    response = client.chat.completions.create(
        model = model,
        messages = [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": text
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail":"low"
                },\
            }
            ]
        }
        ],
        max_tokens = max_tokens
    )

    out = find_json_in_string(response.choices[0].message.content)
    # pprint(out)
    # Image.open(image_path).save('test.jpg')
    # breakpoint()

    return out

