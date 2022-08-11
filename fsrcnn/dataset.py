import random

import cv2
import numpy as np
from numpy import asarray
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms.functional as functional


def to_ndarray(value):
    value = np.array([value])
    return value.astype('float').reshape(1, 1)


class Dataset(TorchDataset):
    def __init__(self, image_paths, upscaling_factor=2, only_luminosity=True):
        self.image_paths = image_paths
        self.upscaling_factor = upscaling_factor
        self.only_luminosity = only_luminosity
        self.downscale_factors = [1.0, 0.9, 0.8, 0.7, 0.6]
        self.rotations = [0, 90, 180, 270]
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame = self.image_paths[idx]

        image = Image.open(frame).convert('YCbCr')
        image = asarray(image)
        # Select only luminosity channel
        if self.only_luminosity:
            image = image[:, :, 0]
            image = np.expand_dims(image, axis=-1)

        height, width = image.shape[:-1]
        diff = abs(width - height) // 2
        if diff > 0 and height < width:
            cropped_image = image[:, diff:-diff, :]
        elif diff > 0 and width < height:
            cropped_image = image[diff:-diff, :, :]
        else:
            cropped_image = image

        # Downscaling augmentation
        random_downscaling_factor = random.choice(self.downscale_factors)
        new_size = int(cropped_image.shape[0] * random_downscaling_factor)
        cropped_image = cv2.resize(cropped_image, dsize=(new_size, new_size), interpolation=cv2.INTER_CUBIC)

        # Rotation augmentation
        random_rotation_factor = random.choice(self.rotations)
        (height, width) = cropped_image.shape[:2]
        center = (width / 2, height / 2)
        transform_matrix = cv2.getRotationMatrix2D(center, random_rotation_factor, 1.0)
        cropped_image = cv2.warpAffine(cropped_image, transform_matrix, (width, height))

        if len(cropped_image.shape) == 2:
            cropped_image = np.expand_dims(cropped_image, axis=-1)

        low_size = int(cropped_image.shape[0] / self.upscaling_factor)
        low_res_image = cv2.resize(cropped_image, dsize=(low_size, low_size), interpolation=cv2.INTER_CUBIC)
        # opencv throws away trailing dimensions so we have to add it back
        if len(low_res_image.shape) == 2:
            low_res_image = np.expand_dims(low_res_image, axis=-1)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        cropped_image = cropped_image.transpose((2, 0, 1)).astype(float)
        low_res_image = low_res_image.transpose((2, 0, 1)).astype(float)

        # Normalize input
        if not self.only_luminosity:
            low_res_image = functional.normalize(torch.from_numpy(low_res_image), self.means, self.stds)

        return {'high_res_image': cropped_image, 'low_res_image': low_res_image}
