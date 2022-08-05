import os
from glob import glob
import random

import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torchmetrics.functional import peak_signal_noise_ratio

from fsrcnn.dataset import Dataset
from fsrcnn.model import Model


upscaling_factor = 2
mixed_precision_enabled = False
epochs = 10
batch_size = 1
learning_rate = 0.001


def create_dataloaders():
    t91_image_paths = glob('datasets/T91/*')
    random.shuffle(t91_image_paths)

    train_set = int(0.8 * len(t91_image_paths))
    train_paths = t91_image_paths[:train_set]
    val_paths = t91_image_paths[train_set:]

    train_dataset = Dataset(train_paths, upscaling_factor=upscaling_factor)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    val_dataset = Dataset(val_paths, upscaling_factor=upscaling_factor)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    bsd100_image_paths = glob('datasets/BSD100_SR/image_SRF4/*HR.png')
    bsd100_dataset = Dataset(bsd100_image_paths, upscaling_factor=upscaling_factor)
    bsd100_dataloader = DataLoader(bsd100_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    set5_image_paths = glob('datasets/Set5/image_SRF4/*HR.png')
    set5_dataset = Dataset(set5_image_paths, upscaling_factor=upscaling_factor)
    set5_dataloader = DataLoader(set5_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    set14_image_paths = glob('datasets/Set14/image_SRF4/*HR.png')
    set14_dataset = Dataset(set14_image_paths, upscaling_factor=upscaling_factor)
    set14_dataloader = DataLoader(set14_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    return train_dataloader, val_dataloader, bsd100_dataloader, set5_dataloader, set14_dataloader


def calculate_validation_metrics(batch, model, criterion):
    high_res_image = batch['high_res_image'].float().to(device)
    low_res_image = batch['low_res_image'].float().to(device)
    with torch.cuda.amp.autocast(enabled=False):
        outputs = model(low_res_image.float())
        # This resizing 'hack' is needed because the deconvolved output and the high resolution image are not exactly the same shape
        high_res_image = np.expand_dims(resize(np.squeeze(high_res_image.cpu().numpy().transpose((0, 2, 3, 1))),
                                               outputs.shape[-2:]).transpose((2, 0, 1)), axis=0)
        high_res_image = torch.from_numpy(high_res_image).float().to(device)

        return criterion(outputs, high_res_image), peak_signal_noise_ratio(outputs, high_res_image)


def evaluate(model, test_dataloader, dataset_name):
    test_psnr = 0.0
    model.eval()
    with torch.set_grad_enabled(False):
        for batch_idx, batch in tqdm(enumerate(test_dataloader)):
            high_res_image = batch['high_res_image'].float().to(device)
            low_res_image = batch['low_res_image'].float().to(device)
            with torch.cuda.amp.autocast(enabled=False):
                outputs = model(low_res_image.float())
                # This resizing 'hack' is needed because the deconvolved output and the high resolution image are not exactly the same shape
                high_res_image = np.expand_dims(resize(np.squeeze(high_res_image.cpu().numpy().transpose((0, 2, 3, 1))),
                                                       outputs.shape[-2:]).transpose((2, 0, 1)), axis=0)
                high_res_image = torch.from_numpy(high_res_image).float().to(device)

                peak_signal_noise_ratio(outputs, high_res_image)

        print('Average PSNR for test set {}: {}'.format(dataset_name, test_psnr / len(test_dataloader.dataset)))


if __name__ == '__main__':
    train_dataloader, val_dataloader, bsd100_dataloader, set5_dataloader, set14_dataloader = create_dataloaders()

    # Detect if we have a GPU available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(d=32, s=5, n=upscaling_factor).float().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    criterion = MSELoss()

    best_val_loss = np.inf

    for _ in range(epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        epoch_train_psnr = 0.0
        epoch_val_psnr = 0.0

        # Train for 1 epoch
        model.train()
        with torch.set_grad_enabled(True):
            for batch_idx, batch in tqdm(enumerate(train_dataloader)):
                loss, psnr = calculate_validation_metrics(batch, model, criterion)

                # Backpropagate losses
                # TODO: handle scaler?
                loss.backward()

                # Apply updates
                optimizer.step()
                optimizer.zero_grad()

                # statistics
                epoch_train_loss += loss.item() * batch_size
                epoch_train_psnr += float(psnr) * batch_size

        # Validate
        model.eval()
        with torch.set_grad_enabled(False):
            for batch_idx, batch in tqdm(enumerate(val_dataloader)):
                loss, psnr = calculate_validation_metrics(batch, model, criterion)

                # statistics
                epoch_val_loss += loss.item() * batch_size
                epoch_val_psnr += float(psnr) * batch_size
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_val_psnr = epoch_val_psnr
                torch.save(model.state_dict(), 'model.pt')
                torch.save(optimizer, 'optimizer.pt')

            print('Average training loss for epoch: {}'.format(epoch_train_loss / len(train_dataloader.dataset)))
            print('Average validation loss for epoch: {}'.format(epoch_val_loss / len(val_dataloader.dataset)))
            print('Average training PSNR for epoch: {}'.format(epoch_train_psnr / len(train_dataloader.dataset)))
            print('Average validation PSNR for epoch: {}'.format(epoch_val_psnr / len(val_dataloader.dataset)))

    # Test
    print('Evaluating on test sets...')
    model.load_state_dict(torch.load('model.pt'))
    evaluate(model, bsd100_dataloader, 'BSD100')
    evaluate(model, set5_dataloader, 'Set5')
    evaluate(model, set14_dataloader, 'Set14')
