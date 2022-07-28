import numpy as np
from numpy import asarray
from PIL import Image
import torch
from torch.utils.data import Dataset


def to_ndarray(value):
    value = np.array([value])
    return value.astype('float').reshape(1, 1)


class Dataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame = self.image_paths[idx]

        image = Image.open(frame)
        image = asarray(image)

        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample
