import os
from glob import glob
import random

import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torchmetrics.functional import peak_signal_noise_ratio, mean_squared_error
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms.functional as functional

from fsrcnn.dataset import Dataset
from fsrcnn.model import Model


upscaling_factor = 2
mixed_precision_enabled = False
epochs = 100
batch_size = 1
learning_rate = 0.001
d = 48
s = 12
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
user_lr_scheduler = True
use_target_normalization = False
color_channels = 1
experiment_name = 'test'


def create_dataloaders():
    t91_image_paths = glob('datasets/T91/*')
    random.shuffle(t91_image_paths)

    train_set = int(0.8 * len(t91_image_paths))
    train_paths = t91_image_paths[:train_set]
    val_paths = t91_image_paths[train_set:]

    train_dataset = Dataset(train_paths, upscaling_factor=upscaling_factor, only_luminosity=(color_channels == 1))
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    val_dataset = Dataset(val_paths, upscaling_factor=upscaling_factor, only_luminosity=(color_channels == 1))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    bsd100_image_paths = glob('datasets/BSD100_SR/image_SRF_4/*HR.png')
    bsd100_dataset = Dataset(bsd100_image_paths, upscaling_factor=upscaling_factor, only_luminosity=(color_channels == 1))
    bsd100_dataloader = DataLoader(bsd100_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    set5_image_paths = glob('datasets/Set5/image_SRF_4/*HR.png')
    set5_dataset = Dataset(set5_image_paths, upscaling_factor=upscaling_factor, only_luminosity=(color_channels == 1))
    set5_dataloader = DataLoader(set5_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    set14_image_paths = glob('datasets/Set14/image_SRF_4/*HR.png')
    set14_dataset = Dataset(set14_image_paths, upscaling_factor=upscaling_factor, only_luminosity=(color_channels == 1))
    set14_dataloader = DataLoader(set14_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    return train_dataloader, val_dataloader, bsd100_dataloader, set5_dataloader, set14_dataloader


def calculate_validation_metrics(batch, model, criterion):
    high_res_image = batch['high_res_image'].float().to(device)
    low_res_image = batch['low_res_image'].float().to(device)
    with torch.cuda.amp.autocast(enabled=False):
        outputs = model(low_res_image.float())
        # This resizing 'hack' is needed because the deconvolved output and the high resolution image are not exactly the same shape

        high_res_image = np.expand_dims(resize((high_res_image.cpu().numpy().transpose((0, 2, 3, 1)))[0],
                                               outputs.shape[-2:]).transpose((2, 0, 1)), axis=0)
        high_res_image = torch.from_numpy(high_res_image).float().to(device)

        if use_target_normalization:
            # Normalize high res image for the loss
            high_res_image_normalized = functional.normalize(high_res_image, means, stds)

            # Denormalize outputs for PSNR
            # Normalization: output[channel] = (input[channel] - mean[channel]) / std[channel]
            outputs_denormalized = torch.empty_like(outputs)
            for i in range(3):
                outputs_denormalized[0, i, :, :] = (outputs[0, i, :, :] * stds[i]) + means[i]

            return criterion(outputs, high_res_image_normalized), peak_signal_noise_ratio(outputs_denormalized, high_res_image), \
                   mean_squared_error(outputs_denormalized, high_res_image)
        else:
            return criterion(outputs, high_res_image), peak_signal_noise_ratio(outputs, high_res_image), \
                   mean_squared_error(outputs, high_res_image)


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
                high_res_image = np.expand_dims(resize((high_res_image.cpu().numpy().transpose((0, 2, 3, 1)))[0],
                                                       outputs.shape[-2:]).transpose((2, 0, 1)), axis=0)
                high_res_image = torch.from_numpy(high_res_image).float().to(device)

                if use_target_normalization:
                    # Denormalize outputs for PSNR
                    # Normalization: output[channel] = (input[channel] - mean[channel]) / std[channel]
                    outputs = torch.empty_like(outputs)
                    for i in range(3):
                        outputs[0, i, :, :] = (outputs[0, i, :, :] * stds[i]) + means[i]

                test_psnr += peak_signal_noise_ratio(outputs, high_res_image)

        print('Average PSNR for test set {}: {}'.format(dataset_name, test_psnr / len(test_dataloader.dataset)))


def calculate_psnr(mse):
    return 20 * np.log10(255) - 10 * np.log10((float(mse)))


if __name__ == '__main__':
    train_dataloader, val_dataloader, bsd100_dataloader, set5_dataloader, set14_dataloader = create_dataloaders()

    experiment_path = os.path.join('tensorboard', experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    tensorboard = SummaryWriter(log_dir=experiment_path)

    # Detect if we have a GPU available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(d=d, s=s, upscaling_factor=upscaling_factor, color_channels=color_channels).float().to(device)

    params = []
    for conv_layer in model.conv_layers:
        params.append({
            'params': conv_layer.parameters(),
            'lr': 1e-3
        })
    params.append({
        'params': model.deconv.parameters(),
        'lr': 1e-4
    })

    optimizer = torch.optim.Adam(params, lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    criterion = MSELoss()

    best_val_loss = np.inf
    best_epoch = -1

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        epoch_train_psnr = 0.0
        epoch_val_psnr = 0.0

        # Train for 1 epoch
        model.train()
        with torch.set_grad_enabled(True):
            for batch_idx, batch in tqdm(enumerate(train_dataloader)):
                optimizer.zero_grad()
                loss, _, mse = calculate_validation_metrics(batch, model, criterion)
                psnr = calculate_psnr(mse)

                # Backpropagate losses
                # TODO: handle scaler?
                loss.backward()

                # Apply updates
                optimizer.step()

                # statistics
                epoch_train_loss += loss.item()
                epoch_train_psnr += float(psnr)

        # Validate
        model.eval()
        with torch.set_grad_enabled(False):
            for batch_idx, batch in tqdm(enumerate(val_dataloader)):
                loss, _, mse = calculate_validation_metrics(batch, model, criterion)
                psnr = calculate_psnr(mse)

                # statistics
                epoch_val_loss += loss.item()
                epoch_val_psnr += float(psnr)

            train_loss = epoch_train_loss / len(train_dataloader.dataset)
            val_loss = epoch_val_loss / len(val_dataloader.dataset)
            train_psnr = epoch_train_psnr / len(train_dataloader.dataset)
            val_psnr = epoch_val_psnr / len(val_dataloader.dataset)

            if val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss
                best_val_psnr = val_psnr
                torch.save(model.state_dict(), 'model.pt')
                torch.save(optimizer, 'optimizer.pt')

            # Reduce learning rate of necessary
            if user_lr_scheduler:
                lr_scheduler.step(val_loss)

            # Log train metrics
            tensorboard.add_scalar('train loss', train_loss, epoch+1)
            tensorboard.add_scalar('val loss', val_loss, epoch + 1)
            tensorboard.add_scalar('train psnr', train_psnr, epoch + 1)
            tensorboard.add_scalar('val psnr', val_psnr, epoch + 1)

            for i, param_group in enumerate(optimizer.param_groups):
                tensorboard.add_scalar('Group {} learning rate'.format(i+1), param_group['lr'], epoch+1)

            print('Average training loss for epoch: {}'.format(train_loss))
            print('Average validation loss for epoch: {}'.format(val_loss))
            print('Average training PSNR for epoch: {}'.format(train_psnr))
            print('Average validation PSNR for epoch: {}'.format(val_psnr))

    print('Best epoch: {}.:'.format(best_epoch+1))
    print('Average validation loss for epoch {}: {}'.format(best_epoch + 1, best_val_loss))
    print('Average validation PSNR for epoch {}: {}'.format(best_epoch + 1, best_val_psnr))

    # Test
    print('Evaluating on test sets...')
    model.load_state_dict(torch.load('model.pt'))
    evaluate(model, bsd100_dataloader, 'BSD100')
    evaluate(model, set5_dataloader, 'Set5')
    #evaluate(model, set14_dataloader, 'Set14')
