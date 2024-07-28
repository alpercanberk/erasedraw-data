import re
import json 
import os

def find_json_in_string(s):

    json_pattern = r'(\{.*?\}|\[.*?\])'
    potential_jsons = re.findall(json_pattern, s, re.DOTALL)

    for potential_json in potential_jsons:
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            continue
    return None