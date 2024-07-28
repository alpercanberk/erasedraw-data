import webdataset as wds
import torch
from torch.utils.data import DataLoader
import json
import io
from PIL import Image
import glob
import numpy as np
import torchvision.transforms as transforms

class EraseDrawDataset(wds.WebDataset):
    def __init__(self, directory, resolution=512, *args, **kwargs):
        super().__init__(glob.glob(f"{directory}/*.tar"), *args, **kwargs)
        self.resolution = resolution
        self.simple_caption_percent = 0.1

    def __iter__(self):
        for item in super().__iter__():
            out = self.process_item(item)
            if out is not None:
                yield out


    def get_image_crop(self, image):
        """
        Input: PIL Image
        Output: (x1, y1, x2, y2)
        """
        width, height = image.size
        crop_size = min(width, height)
        if width < height:
            #pick random vertical coordinate
            y = np.random.randint(0, height - crop_size)
            # return (y, 0, y + crop_size, crop_size)
            return (0, y, crop_size, y + crop_size)
        else:
            #pick random horizontal crop
            x = np.random.randint(0, width - crop_size)
            # return (0, x, crop_size, x + crop_size)
            return (x, 0, x + crop_size, crop_size)
        
    def is_object_mask_in_crop(self, mask, crop_region, intersection_threshold=0.1):
        """
        Input: PIL Image, (x1, y1, x2, y2), intersection_threshold
        Output: bool

        Intersection threshold is a value between 0 and 1. If the crop region intersections less
        than this amount of the object mask, the crop region is not valid.
        """
        mask_numpy = np.array(mask)
        mask_sum = np.sum(mask_numpy)
        if mask_sum == 0:
            raise ValueError("Mask is empty")
        
        crop_mask = mask_numpy[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]
        return np.sum(crop_mask) > np.sum(mask_numpy) * intersection_threshold

    def process_item(self, item):
        # Extract shard name
        url = item['__url__']
        shard_name = url.split('/')[-1]

        # Load original image
        original_img = Image.open(io.BytesIO(item['original.jpg']))
        original_img = original_img.convert('RGB')

        # Load metadata
        metadata = json.loads(item['json'].decode('utf-8'))

        # randomly choose one of the edited images
        edited_image_keys = [k for k in item.keys() if k.startswith('edited.') and k.endswith('.jpg')]
        if len(edited_image_keys) == 0: #if there are no edited images, skip this sample
            return None 
        
        key = np.random.choice(edited_image_keys, 1)[0]
        local_index = int(key.split('.')[-2])
        edited_image = Image.open(io.BytesIO(item[key])).convert('RGB')
        mask = None
        if key.replace('edited.', 'masks.') in item.keys():
            mask = Image.open(io.BytesIO(item[key.replace('edited.', 'masks.')])).convert('L')
            mask = mask.resize(edited_image.size)

        original_img = original_img.resize(edited_image.size)

        if torch.rand(1) < self.simple_caption_percent:
            caption = metadata['names'][local_index]
        else:
            caption = np.random.choice(metadata['prompts'][local_index], 1)[0]
    
        #get crop region
        try:
            crop_region = self.get_image_crop(original_img)
            if mask is not None:
                try_count = 0
                #if the object mask exists, make sure the crop region contains at least 10% of the mask
                while not self.is_object_mask_in_crop(mask, crop_region, intersection_threshold=0.8):
                    crop_region = self.get_image_crop(original_img)
                    try_count += 1
                    if try_count > 10:
                        break #something is wrong
        except ValueError as e:
            print(f"Error: {e}")
            return None


        cropped_original_img = original_img.crop(crop_region)
        cropped_edited_image = edited_image.crop(crop_region)

        return {
            'original': cropped_original_img,
            'edited': cropped_edited_image,
            'prompt': caption,
            'shard_name': shard_name
        }

def collate_fn(batch, resolution=512):
    originals = []
    editeds = []
    prompts = []
    shard_names = []

    for item in batch:
        if item is not None:
            originals.append(transforms.Resize((resolution, resolution))(transforms.ToTensor()(item['original'])))
            editeds.append(transforms.Resize((resolution, resolution))(transforms.ToTensor()(item['edited'])))
            prompts.append(item['prompt'])
            shard_names.append(item['shard_name'])

    return {
        'original': torch.stack(originals),
        'edited': torch.stack(editeds),
        'prompt': prompts,
        'shard_name': shard_names
    }

def create_dataloader(directory, batch_size=4, num_workers=4, resolution=512):
    dataset = EraseDrawDataset(directory)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataloader = create_dataloader("./dataset-v1", batch_size=4)
    for batch in dataloader:
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        for i in range(4):
            axes[i, 0].imshow(batch['original'][i].permute(1, 2, 0))
            axes[i, 1].imshow(batch['edited'][i].permute(1, 2, 0))
            axes[i, 0].set_title(f"Original - {batch['shard_name'][i]}")
            axes[i, 1].set_title(f"Edited - {batch['prompt'][i][:50]}...")
            for ax in axes[i]:
                ax.axis('off')
        plt.tight_layout()
        plt.show()
        plt.savefig("./example.png")
        break 