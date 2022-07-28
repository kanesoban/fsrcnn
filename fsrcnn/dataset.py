import cv2
import numpy as np
from numpy import asarray
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset


def to_ndarray(value):
    value = np.array([value])
    return value.astype('float').reshape(1, 1)


class Dataset(TorchDataset):
    def __init__(self, image_paths, downscaling_factor, transform=None):
        self.image_paths = image_paths
        self.downscaling_factor = downscaling_factor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame = self.image_paths[idx]

        image = Image.open(frame)
        image = asarray(image)

        height, width = image.shape[:-1]
        diff = abs(width - height) // 2
        if height < width:
            cropped_image = image[:, diff:-diff, :]
        elif width < height:
            cropped_image = image[diff:-diff, :, :]
        else:
            cropped_image = image

        low_size = cropped_image.shape[0] // self.downscaling_factor
        low_res_image = cv2.resize(cropped_image, dsize=(low_size, low_size), interpolation=cv2.INTER_CUBIC)

        sample = {'high_res_image': cropped_image, 'low_res_image': low_res_image}

        if self.transform:
            sample = self.transform(sample)

        return sample
