"""
Generates masks for all images in a directory using the SAM model.
"""

import ray
import os
import json
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import pickle as pkl
import torch
from segment_anything.utils.transforms import ResizeLongestSide
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device) 
    return image.permute(2, 0, 1).contiguous()

@ray.remote(num_gpus=1)
class SamModelActor:
    def __init__(self, data_dir, batch_size=32, save_visualizations=False):
        sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
        print("Loaded SAM model:", sam_checkpoint)
        self.sam.mask_threshold = -0.4
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.save_visualizations = save_visualizations

    def forward_and_save(self, sam_inputs, infos):
        batched_output = self.sam(sam_inputs, multimask_output=False)
        print("batched output")
        for sam_output, info in zip(batched_output, infos):
            out = {}
            
            for object_mask, labels in zip(sam_output['masks'], info['indices_in_aggregated_boxes']):
                if labels not in out:
                    out[labels] = object_mask.cpu().numpy()
                else:
                    out[labels] += object_mask.cpu().numpy()
            print("info keys", info['keys'])

            out_object_names = [] #list of object names
            out_object_prompts = [] #list of object prompts

            for idx, label in enumerate(info['keys']):
                Image.fromarray(out[label][0]).convert("L").save(os.path.join(self.data_dir, f"{info['image_dir']}.masks.{idx:02d}.jpg"))
                out_object_names.append(label)
                out_object_prompts.append(info['captions_file_content'][label])

            with open(os.path.join(self.data_dir, f"{info['image_dir']}.json"), "w") as f:
                json.dump({"names": out_object_names, "prompts": out_object_prompts}, f)
            
            if self.save_visualizations:
                for idx, label in enumerate(info['keys']):
                    Image.fromarray(out[label][0]).save(os.path.join(self.data_dir, f"{info['image_dir']}.vis.{idx:02d}.jpg"))

    def process_image_dir(self, data_dir, image_id):

        """
        returns sam_input, info, num_bboxes
        """

        sam = self.sam 

        resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

        image_path = os.path.join(data_dir, f"{image_id}.original.jpg")
        original_img = np.array(Image.open(image_path).convert("RGB"))

        processed_img = prepare_image(original_img, resize_transform, sam.device)

        with open(os.path.join(data_dir, f"{image_id}.bounding_boxes.json")) as f:
            unprocessed_bboxes = json.load(f)

        with open(os.path.join(data_dir, f"{image_id}.captions.json")) as f:
            captions_file_content = json.load(f)

        aggregated_bboxes = []
        indices_in_aggregated_boxes = []

        for key, bboxes in unprocessed_bboxes.items():
            aggregated_bboxes.extend(bboxes)
            indices_in_aggregated_boxes.extend([key] * len(bboxes))
        
        if not aggregated_bboxes:
            print("No bboxes could be processed from:", image_id)
            return None, None, 0

        sam_input = {
            'image': processed_img,
            'boxes': resize_transform.apply_boxes_torch(torch.tensor(aggregated_bboxes, device=sam.device), original_img.shape[:2]),
            'original_size': original_img.shape[:2]
        }

        info = {
            'indices_in_aggregated_boxes': indices_in_aggregated_boxes,
            'image_path': image_path,
            'image_dir': image_id,
            'size': len(aggregated_bboxes),
            'keys': list(unprocessed_bboxes.keys()),
            'captions_file_content': captions_file_content
        }

        return sam_input, info, len(aggregated_bboxes)
    
    def generate(self, data_dir, image_ids):
        sam_inputs = []
        infos = []
        cur_n_bboxes = 0

        total_bboxes_processed = 0
        total_images_processed = 0

        # Initialize tqdm progress bar
        pbar = tqdm(total=len(image_ids))

        for image_id in image_ids:
            sam_input, info, n_bboxes = self.process_image_dir(data_dir, image_id)
            print("n bboxes", n_bboxes)
            if n_bboxes > 0:
                # Check if adding this image would exceed the batch size
                if cur_n_bboxes + n_bboxes > self.batch_size:
                    # Process the current batch before adding the new image
                    if sam_inputs:
                        print("forwarding and saving")
                        self.forward_and_save(sam_inputs, infos)
                        sam_inputs = []
                        infos = []
                        cur_n_bboxes = 0

                sam_inputs.append(sam_input)
                infos.append(info)
                cur_n_bboxes += n_bboxes

            total_bboxes_processed += n_bboxes
            total_images_processed += 1

            # Update the description of the progress bar
            pbar.set_description(f"{total_images_processed} images, {total_bboxes_processed} bboxes")

            # Update the progress bar for each image processed
            pbar.update(1)

        # Process any remaining images in the last batch
        if sam_inputs:
            print("forwarding and saving final batch")
            self.forward_and_save(sam_inputs, infos)

        # Close the progress bar
        pbar.close()

def visualize_results(data_dir, image_ids):
    for image_id in tqdm(image_ids, desc="Visualizing results"):
        # Load the original image
        image_path = os.path.join(data_dir, f"{image_id}.original.jpg")
        original_img = Image.open(image_path).convert("RGB")
        original_array = np.array(original_img)

        # Find all mask files for this image
        mask_files = glob.glob(os.path.join(data_dir, f"{image_id}.masks.*.png"))

        # Create a black overlay
        overlay = np.ones_like(original_array)

        # Process each mask
        for mask_file in mask_files:
            mask = np.array(Image.open(mask_file).convert("L"))
            # Add the mask to the overlay
            overlay[mask > 0] = [0, 0, 0]  # Set to black where mask is non-zero

        #multiply the overlay by the original image
        result = original_array * overlay

        # Save the visualized result
        output_path = os.path.join(data_dir, f"{image_id}.mask_visualization.jpg")
        Image.fromarray(result).save(output_path)
        print("Saved visualization to:", output_path)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dir", type=str)
    parser.add_argument("--save_visualizations", action="store_true")
    args = parser.parse_args()

    ray.init()
    num_gpus = torch.cuda.device_count()

    #segmentation requirees bounding boxes
    bounding_boxes_files = glob.glob(os.path.join(args.dir, "*.bounding_boxes.json"))
    image_ids = [os.path.basename(f).split(".")[0] for f in bounding_boxes_files]

    split_image_ids = np.array_split(image_ids, num_gpus)
    sam_actors = [SamModelActor.remote(args.dir, args.batch_size, args.save_visualizations) for _ in range(num_gpus)]

    # Collect all future objects
    futures = [actor.generate.remote(args.dir, dirs.tolist()) for actor, dirs in zip(sam_actors, split_image_ids)]

    # Wait for all actors to complete their tasks
    results = ray.get(futures)
    
    # Visualize the results
    if args.save_visualizations:
        visualize_results(args.dir, image_ids)