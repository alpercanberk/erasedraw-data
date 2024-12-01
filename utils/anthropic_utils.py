
import base64
from utils.utils import find_json_in_string
from PIL import Image
import io
from pprint import pprint

def generate_descriptions_anthropic(client, image_path, prompt_path, max_tokens=1000, model='claude-3-5-sonnet-20240620'):
    
    img = Image.open(image_path)
    text = open(prompt_path, 'r').read()

    #resize the image to have the longest edge be 720 pixels
    width, height = img.size
    MAX_SIZE = 720
    if width > MAX_SIZE or height > MAX_SIZE:
        if height > width:
            resize_factor = MAX_SIZE / height
        else:
            resize_factor = MAX_SIZE / width
        img = img.resize((int(width * resize_factor), int(height * resize_factor)))

        image_file = io.BytesIO()
        img.save(image_file, 'JPEG')
        image_file.seek(0)
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
    #send message
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]
    )

    out = find_json_in_string(message.content[0].text)
    # pprint(out)
    # img.save('test.jpg')
    # breakpoint()

    return out
