import os
import json
import webdataset as wds
import argparse
from PIL import Image
import re
import glob 

def convert_to_wds(input_dir, output_dir, max_shard_size=1024*1024*1024):  # 1GB in bytes
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_image_files = glob.glob(f"{input_dir}/*.original.jpg")
    total_files = len(original_image_files)
    all_files = [f.split('/')[-1] for f in glob.glob(f"{input_dir}/*.jpg")]
    
    current_shard = 0
    current_shard_size = 0
    items_in_shard = 0

    with wds.ShardWriter(f"{output_dir}/erasedraw-%06d-wds.tar", maxcount=1_000_000) as sink:
        for idx, original_image_file in enumerate(original_image_files):
            image_id = original_image_file.split('.')[0].split('/')[-1]
            
            # Get all related files
            original_img = f"{image_id}.original.jpg"
            json_file = f"{image_id}.json"
            edited_files = [f for f in all_files if f.startswith(f"{image_id}.edited.") and f.endswith('.jpg')]
            mask_files = [f for f in all_files if f.startswith(f"{image_id}.masks.") and f.endswith('.jpg')]

            if len(mask_files) == 0:
                print(f"No mask files found for {image_id}")
                continue

            if len(edited_files) == 0:
                print(f"[WARNING] No edited files found for {image_id}")
                continue
            
            assert len(edited_files) == len(mask_files)
            
            # Calculate total size of all files
            total_size = sum(os.path.getsize(os.path.join(input_dir, f)) for f in 
                                [original_img, json_file] + edited_files + mask_files)
            
            # If adding these files would exceed the shard size, start a new shard
            if current_shard_size + total_size > max_shard_size:
                current_shard += 1
                current_shard_size = 0
                items_in_shard = 0
                sink = wds.ShardWriter(f"{output_dir}/erasedraw-%06d-wds.tar", maxcount=1_000_000)
            
            # Prepare data for writing
            data = {"__key__": f"{image_id}"}
            
            with open(os.path.join(input_dir, original_img), 'rb') as f:
                data['original.jpg'] = f.read()
            
            with open(os.path.join(input_dir, json_file), 'r') as f:
                data['json'] = f.read()
            
            for edited_file in edited_files:
                with open(os.path.join(input_dir, edited_file), 'rb') as f:
                    index = re.search(r'\.edited\.(\d+)\.jpg', edited_file).group(1)
                    data[f'edited.{index}.jpg'] = f.read()
            
            for mask_file in mask_files:
                with open(os.path.join(input_dir, mask_file), 'rb') as f:
                    index = re.search(r'\.masks\.(\d+)\.jpg', mask_file).group(1)
                    data[f'masks.{index}.jpg'] = f.read()
            
            # Write to WebDataset
            sink.write(data)
            
            current_shard_size += total_size
            items_in_shard += 1
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{total_files} files")

    print(f"Conversion complete. Created {current_shard + 1} shards.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert image directory to WebDataset format")
    parser.add_argument("--dir", required=True, help="Path to the input directory containing images and metadata")
    parser.add_argument("--outdir", required=True, help="Path to the output directory for WebDataset shards")
    args = parser.parse_args()

    convert_to_wds(args.dir, args.outdir)