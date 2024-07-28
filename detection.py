import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, LlamaTokenizer
import re
import random
import glob
import os
import ray
import json
from tqdm import tqdm
import argparse

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection / float(area1 + area2 - intersection)
    return iou

def filter_overlapping_boxes(boxes, iou_threshold=0.9):
    filtered_box_indices = set()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if calculate_iou(boxes[i], boxes[j]) >= iou_threshold:
                filtered_box_indices.add(j)
                filtered_box_indices.add(i)
    filtered_boxes = [boxes[i] for i in filtered_box_indices]
    return filtered_boxes

def merge_boxes(boxes, iou_threshold=0.9):
    merged_boxes = boxes.copy()
    
    while True:
        merged = False
        for i in range(len(merged_boxes)):
            for j in range(i + 1, len(merged_boxes)):
                if calculate_iou(merged_boxes[i], merged_boxes[j]) >= iou_threshold:
                    # Merge the boxes
                    merged_box = [
                        min(merged_boxes[i][0], merged_boxes[j][0]),
                        min(merged_boxes[i][1], merged_boxes[j][1]),
                        max(merged_boxes[i][2], merged_boxes[j][2]),
                        max(merged_boxes[i][3], merged_boxes[j][3])
                    ]
                    merged_boxes[i] = merged_box
                    merged_boxes.pop(j)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    
    return merged_boxes

def visualize_labeled_boxes(image, bboxes_per_object, output_path):
    image_with_bbox = image.copy()
    draw = ImageDraw.Draw(image_with_bbox)
    
    colors = ["red", "blue", "green", "yellow", "purple", "cyan", "magenta", "orange"]
    
    for i, (object_name, bboxes) in enumerate(bboxes_per_object.items()):
        color = colors[i % len(colors)]
        for bbox in bboxes:
            draw.rectangle(bbox, outline=color, width=3)
            # Draw label above the bounding box
            label_position = (bbox[0], bbox[1] - 20)
            draw.text(label_position, object_name, fill=color)
    
    image_with_bbox.save(output_path)
    print(f"Image with labeled bounding boxes saved as {output_path}")

@ray.remote(num_gpus=1)
class CogVLMDetector:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        self.model = AutoModelForCausalLM.from_pretrained(
            'THUDM/cogvlm-grounding-generalist-hf',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cuda').eval()

    def infer_single(self, image_path, query):
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=query, images=[image])
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }
        print("transformed images shape", inputs['images'][0][0].shape)
        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            out = self.tokenizer.decode(outputs[0])
            print(out)

        pattern = r"\[\[(.*?)\]\]"
        positions = re.findall(pattern, out)
        boxes_list_per_object_identified = [[[int(y) for y in x.split(',')] for x in pos.split(';') if x.replace(',', '').isdigit()] for pos in positions]
        boxes_list_for_object_queried = boxes_list_per_object_identified[0] #we assume the first object is the one we are querying for

        scaled_boxes = []
        for bbox in boxes_list_for_object_queried:
            scaled_bbox = [
                int(bbox[0]) * (width / 1000),
                int(bbox[1]) * (height / 1000),
                int(bbox[2]) * (width / 1000),
                int(bbox[3]) * (height / 1000)
            ]
            scaled_boxes.append(scaled_bbox)

        return image, scaled_boxes

    def infer_multiple(self, image_path, query, n):
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        # Build inputs for a single image-query pair
        single_inputs = self.model.build_conversation_input_ids(self.tokenizer, query=query, images=[image])
        
        # Create batched inputs by repeating the single inputs n times
        batched_inputs = {
            'input_ids': single_inputs['input_ids'].repeat(n, 1).to('cuda'),
            'token_type_ids': single_inputs['token_type_ids'].repeat(n, 1).to('cuda'),
            'attention_mask': single_inputs['attention_mask'].repeat(n, 1).to('cuda'),
            'images': [[single_inputs['images'][0].to('cuda').to(torch.bfloat16)] for _ in range(n)],
        }

        gen_kwargs = {"max_length": 2048, "do_sample": True, "temperature": 0.2}

        with torch.no_grad():
            outputs = self.model.generate(**batched_inputs, **gen_kwargs)
            outputs = outputs[:, batched_inputs['input_ids'].shape[1]:]
            decoded_outputs = [self.tokenizer.decode(output) for output in outputs]

        all_scaled_boxes = []
        for out in decoded_outputs:
            pattern = r"\[\[(.*?)\]\]"
            positions = re.findall(pattern, out)
            boxes_list_per_object_identified = [[[int(y) for y in x.split(',')] for x in pos.split(';') if x.replace(',', '').isdigit()] for pos in positions]
            boxes_list_for_object_queried = boxes_list_per_object_identified[0] #we assume the first object is the one we are querying for

            scaled_boxes = []
            for bbox in boxes_list_for_object_queried:
                scaled_bbox = [
                        int(bbox[0]) * (width / 1000),
                        int(bbox[1]) * (height / 1000),
                        int(bbox[2]) * (width / 1000),
                        int(bbox[3]) * (height / 1000)
                    ]
                scaled_boxes.append(scaled_bbox)
            all_scaled_boxes.extend(scaled_boxes)

        return image, all_scaled_boxes

    def save_outputs(self, image, all_boxes, output_path):
        image_with_bbox = image.copy()
        draw = ImageDraw.Draw(image_with_bbox)
        
        colors = ["red", "blue", "green", "yellow", "purple", "cyan", "magenta", "orange"]
        
        for i, bbox in enumerate(all_boxes):
            color = colors[i % len(colors)]  # Cycle through colors if there are more sets than colors
            draw.rectangle(bbox, outline=color, width=3)
        
        image_with_bbox.save(output_path)
        print(f"Image with all bounding boxes saved as {output_path}")

    def infer_and_process_multiple(self, image_path, query, n=3):
        image, all_boxes = self.infer_multiple(image_path, query, n)
        filtered_boxes = filter_overlapping_boxes(all_boxes)
        merged_boxes = merge_boxes(filtered_boxes)
        return image, merged_boxes

    def infer_and_process_paths(self, image_and_captions_paths, save_visualizations=False, override_existing=True, sample_multiple=True):
        for image_path, captions_path in tqdm(image_and_captions_paths, desc=f"Worker {self.worker_id}"):
            object_names = json.load(open(captions_path)).keys()
            output_annotations_path = os.path.join(os.path.dirname(captions_path), os.path.basename(captions_path).replace('.captions.json', '.bounding_boxes.json'))
            output_visualizations_path = os.path.join(os.path.dirname(captions_path), os.path.basename(captions_path).replace('.captions.json', '.bbox_visualizations.jpg'))
            bboxes_per_object = {}
            image = None

            if os.path.exists(output_annotations_path) and not override_existing:
                continue

            for object_name in object_names:
                query = f'Where is the {object_name}?'
                image, bboxes = self.infer_multiple(image_path, query, n=3 if sample_multiple else 1)

                if sample_multiple:
                    bboxes = filter_overlapping_boxes(bboxes)

                if len(bboxes) == 0: #if nothing overlaps, then discard object
                    continue

                if sample_multiple:
                    bboxes = merge_boxes(bboxes)
                bboxes_per_object[object_name] = bboxes

            with open(output_annotations_path, 'w') as f:
                json.dump(bboxes_per_object, f)

            if save_visualizations:
                print(f"Worker {self.worker_id}: Saving visualization at", output_visualizations_path)
                visualize_labeled_boxes(image, bboxes_per_object, output_visualizations_path)

def main():
    parser = argparse.ArgumentParser(description="Run CogVLM detection on images")
    parser.add_argument("--dir", type=str, default='/local/vondrick/alper/erasedraw-data-prerelease/sam_sample_raw_0720',
                        help="Directory containing images and captions")
    parser.add_argument("--image_extension", type=str, default='.original.jpg',
                        help="Extension of image files")
    parser.add_argument("--captions_extension", type=str, default='.captions.json',
                        help="Extension of caption files")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save visualizations of bounding boxes")
    args = parser.parse_args()

    ray.init()

    input_image_and_captions_paths = []

    for image_path in glob.glob(os.path.join(args.dir, '*' + args.image_extension)):
        image_id = os.path.basename(image_path).split('.')[0]
        captions_path = os.path.join(args.dir, image_id + args.captions_extension)
        input_image_and_captions_paths.append((image_path, captions_path))
    print("Total input image and captions paths:", len(input_image_and_captions_paths))

    num_gpus = torch.cuda.device_count()
    chunk_size = len(input_image_and_captions_paths) // num_gpus
    chunks = [input_image_and_captions_paths[i:i + chunk_size] for i in range(0, len(input_image_and_captions_paths), chunk_size)]

    detectors = [CogVLMDetector.remote(i) for i in range(num_gpus)]
    tasks = [detector.infer_and_process_paths.remote(chunk, save_visualizations=args.save_visualizations) for detector, chunk in zip(detectors, chunks)]

    ray.get(tasks)

if __name__ == "__main__":
    main()