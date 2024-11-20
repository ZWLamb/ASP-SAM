import os
import numpy as np
from torch.utils.data import Dataset

def log_with_condition(matrix):
    result = np.zeros_like(matrix, dtype=float)
    zero_indices = matrix == 0
    nonzero_indices = matrix != 0
    result[nonzero_indices] = np.log(matrix[nonzero_indices])
    return result


class DSB(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='none',
                 plane=False):
        self.data_path = data_path
        self.img_paths = [f.path for f in os.scandir(os.path.join(data_path, mode))]
        self.mode = mode
        self.prompt = prompt
        self.img_size = 256 #args.image_size
        self.mask_size = args.out_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        point_label = 1

        """Get the images"""
        img_path = self.img_paths[index]
        name = img_path.split('/')[-1]

        # raw image and raters images

        sample = np.load(img_path)

        image = (sample[:, :, :3] * 255.0).astype(np.uint8)
        inst_mask = sample[:, :, 3]
        nuclear_mask = sample[:, :, 4][np.newaxis, :]
        scale_mask = log_with_condition(sample[:, :, 7])[np.newaxis, :]

        pt = np.array([0, 0], dtype=np.int32)

        image = image.transpose(2,0,1)
        box_cup = [0, 0, 0, 0]
        image_meta_dict = {'filename_or_obj': name}
        return {
            'name':name,
            'image': image,
            'label': nuclear_mask,
            'inst_mask': inst_mask,
            'scale_mask':scale_mask,
            'p_label': point_label,
            'pt': pt,
            'box': box_cup,
            'image_meta_dict': image_meta_dict,
        }