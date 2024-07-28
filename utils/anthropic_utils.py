
import base64
from utils.utils import find_json_in_string
from PIL import Image
import io
from pprint import pprint

def generate_descriptions_anthropic(client, image_path, prompt_path, max_tokens=1000, model='claude-3-5-sonnet-20240620'):
    
    img = Image.open(image_path)

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
                        "text": "1. Describe the clearly identifiable objects in this image, along with their attributes and their spatial relations with respect to the other objects. I prefer singular objects as opposed to plural ones.\n2. For every individual object (maximum of 5),\n    a) Come up with a simple \"subject identification\" for that object. The subject identification should be a **unique** way to identify the object in the image. Color and shape may be helpful to include.\n    Examples:\n        If two fish are in the picture, one is red and the other is blue, then you could identify their subject identificaitons as \"the red fish\" and \"the blue fish\".\n        If three men are in the picture, and they're spread out into a row, then you could identify their subject identifications as \"man on the left\",\"man on the right\",man in the middle\".\n        If only a single cat is in the picture, then the subject identification \"cat\" is sufficient.\n    The subject identification should not exceed 6 words.\n    The subject identification MUST NOT include a noun other than the subject. \n    b) Come up 3 full captions that describe the location of the object with respect to other objects or the image.\n    (e.g. a man with a blue shirt standing in front of the wall, an elephant next to the tree, a bat held by a player, the dog on the right, etc.). \n    Make sure that all of the captions refer to exact the same subject i.e. your subject identification. The subject must be included in this prompt.\n3. Exclude large background elements from your captions, such as the sky, the ground, the walls, etc.\n\nFinally, return your final response as a JSON in the form\n{\n    \"[subject identification]\":\n        [\n            \"[caption 1]\", \"[caption 2]\", \"[caption 3]\", ...\n        ],\n    ...\n}\n\nYou may now begin!"
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