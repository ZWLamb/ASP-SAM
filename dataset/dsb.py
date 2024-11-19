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
        # type_mask = sample[:, :, 6]
        scale_mask = log_with_condition(sample[:, :, 7])[np.newaxis, :]

        #centroid_prob_mask = self.distance_transform(sample[:, :, 3])  # 每个像素点到边界的欧式距离

        # if np.max(centroid_prob_mask) == 0:
        #     pass
        # else:
        #     centroid_prob_mask = (centroid_prob_mask /
        #                           np.max(centroid_prob_mask)) * 1.0  # 每个像素到边界距离的归一化

        # mask = np.zeros((nuclear_mask.shape[0], nuclear_mask.shape[0], 4))
        # mask[:, :, 0] = nuclear_mask  # bin_mask
        # mask[:, :, 1] = centroid_prob_mask  # 每个像素到边界距离的归一化
        # mask[:, :, 2] = inst_map  # 实例索引
        # mask[:, :, 3] = scale_mask  # 尺度map


        # apply preprocessing
        # if self.preprocessing:
        #     image = image / 255.0
        #     sample = self.preprocessing(image=image)  # transpose 255*255*3 --> 3*255*255
        #     image = sample["image"]

        pt = np.array([0, 0], dtype=np.int32)

        # if self.transform:
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