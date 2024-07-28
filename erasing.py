import argparse
import cv2
import glob
import os
import ray
import torch
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
from PIL import Image

@ray.remote  # Assuming 4 actors per GPU by default
class Inpainter:
    def __init__(self):
        self.inpainting = pipeline(Tasks.image_inpainting, model='damo/cv_fft_inpainting_lama', refine=True)

    def process_single(self, image_path, mask_path, output_path):
        input_mask = Image.open(mask_path)
        input_mask = np.array(input_mask)
        input_mask = input_mask.astype(np.uint8)
        avg_img_size = np.mean(input_mask.shape)
        kernel_size = int(avg_img_size/200)
        input_mask = cv2.dilate(input_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=4)
        input_mask = Image.fromarray(input_mask)

        input_data = {
            'img': Image.open(image_path),
            'mask': input_mask,
        }

        result = self.inpainting(input_data)
        vis_img = result[OutputKeys.OUTPUT_IMG]
        cv2.imwrite(output_path, vis_img)

    def process_paths(self, path_tuples):
        for image_path, mask_path, output_path in path_tuples:
            self.process_single(image_path, mask_path, output_path)

def main(args):
    input_dir = args.dir
    num_actors_per_gpu = args.num_actors_per_gpu

    # Get number of GPUs
    num_gpus = torch.cuda.device_count()
    num_actors = num_gpus * num_actors_per_gpu

    # Initialize Ray
    ray.init()

    # Create actors
    actors = [Inpainter.options(num_gpus=1/num_actors_per_gpu).remote() for _ in range(num_actors)]

    # Get all mask files
    mask_files = glob.glob(os.path.join(input_dir, "*.masks.*.jpg"))

    # Prepare tasks
    tasks = []
    for mask_file in mask_files:
        image_id = os.path.basename(mask_file).split('.')[0]
        mask_num = os.path.basename(mask_file).split('.')[-2]
        original_file = os.path.join(input_dir, f"{image_id}.original.jpg")
        output_file = os.path.join(input_dir, f"{image_id}.edited.{mask_num}.jpg")
        tasks.append((original_file, mask_file, output_file))

    # Distribute tasks among actors
    futures = []
    for i, actor in enumerate(actors):
        actor_tasks = tasks[i::num_actors]
        futures.append(actor.process_paths.remote(actor_tasks))

    # Wait for all tasks to complete
    ray.get(futures)

    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inpainting script")
    parser.add_argument("--dir", type=str, required=True, help="Input directory")
    parser.add_argument("--num_actors_per_gpu", type=int, default=4, help="Number of actors per GPU")
    args = parser.parse_args()
    main(args)