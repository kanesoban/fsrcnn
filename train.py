import os
from glob import glob
import random

import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss

from fsrcnn.dataset import Dataset
from fsrcnn.model import Model


upscaling_factor = 2
mixed_precision_enabled = False
epochs = 10

def create_dataloaders():
    image_paths = glob('datasets/T91/*')
    random.shuffle(image_paths)

    train_set = int(0.8 * len(image_paths))
    val_set = int(0.1 * len(image_paths))
    train_paths = image_paths[:train_set]
    val_paths = image_paths[train_set:train_set + val_set]
    test_paths = image_paths[train_set + val_set:]

    train_dataset = Dataset(train_paths, upscaling_factor=upscaling_factor)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    val_dataset = Dataset(val_paths, upscaling_factor=upscaling_factor)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    test_dataset = Dataset(test_paths, upscaling_factor=upscaling_factor)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    return train_dataloader, val_dataloader, test_dataloader


def calculate_loss(batch, model, criterion):
    high_res_image = batch['high_res_image'].float().to(device)
    low_res_image = batch['low_res_image'].float().to(device)
    with torch.cuda.amp.autocast(enabled=False):
        outputs = model(low_res_image.float())
        # This resizing 'hack' is needed because the deconvolved output and the high resolution image are not exactly the same shape
        high_res_image = np.expand_dims(resize(np.squeeze(high_res_image.cpu().numpy().transpose((0, 2, 3, 1))),
                                               outputs.shape[-2:]).transpose((2, 0, 1)), axis=0)
        high_res_image = torch.from_numpy(high_res_image).float().to(device)
        return criterion(outputs, high_res_image)


if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders()
    torch.save(train_dataloader, 'train_dataloader.pt')
    torch.save(val_dataloader, 'val_dataloader.pt')
    torch.save(test_dataloader, 'test_dataloader.pt')

    # Detect if we have a GPU available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(d=32, s=5, n=upscaling_factor).float().to(device)
    optimizer = torch.optim.RMSprop(model.parameters())
    criterion = MSELoss()

    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    best_val_loss = np.inf

    for _ in range(epochs):
        # Train for 1 epoch
        model.train()
        with torch.set_grad_enabled(True):
            for batch_idx, batch in tqdm(enumerate(train_dataloader)):
                loss = calculate_loss(batch, model, criterion)

                # Backpropagate losses
                # TODO: handle scaler?
                loss.backward()

                # Apply updates
                optimizer.step()
                optimizer.zero_grad()

                # statistics
                current_loss = loss.item() * len(batch)
                epoch_train_loss += current_loss

        # Validate
        model.eval()
        with torch.set_grad_enabled(False):
            for batch_idx, batch in tqdm(enumerate(val_dataloader)):
                loss = calculate_loss(batch, model, criterion)

                # statistics
                current_loss = loss.item() * len(batch)
                epoch_val_loss += current_loss
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    torch.save(model.state_dict(), 'model.pt')
                    torch.save(optimizer, 'optimizer.pt')

            print('Average training loss for epoch: {}'.format(epoch_train_loss / len(train_dataloader.dataset)))
            print('Average validation loss for epoch: {}'.format(epoch_val_loss / len(val_dataloader.dataset)))
