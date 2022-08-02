from glob import glob

import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss

from fsrcnn.dataset import Dataset
from fsrcnn.model import Model

if __name__ == '__main__':
    image_paths = glob('datasets/T91/*')

    upscaling_factor = 2
    dataset = Dataset(image_paths, upscaling_factor=upscaling_factor)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # Detect if we have a GPU available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(d=32, s=5, n=upscaling_factor).float().to(device)
    optimizer = torch.optim.RMSprop(model.parameters())
    criterion = MSELoss()

    running_loss = 0.0

    for batch_idx, batch in tqdm(enumerate(dataloader)):
        print('High resolution image shape: {}'.format(batch['high_res_image'].shape))
        print('Low resolution image shape: {}'.format(batch['low_res_image'].shape))

        # track history if only in train
        with torch.set_grad_enabled(True):
            high_res_image = batch['high_res_image'].float().to(device)
            low_res_image = batch['low_res_image'].float().to(device)

            with torch.cuda.amp.autocast(enabled=False):
                outputs = model(low_res_image.float())
                #high_res_image = cv2.resize(outputs, dsize=outputs.shape[-2:], interpolation=cv2.INTER_CUBIC)
                loss = criterion(outputs, high_res_image)

            # statistics
            current_loss = loss.item() * low_res_image.size(0)
            running_loss += current_loss

    print('Average loss: {}'.format(running_loss / len(dataloader.dataset)))
