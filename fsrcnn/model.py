import torch.nn as nn
from torch.nn import Conv2d, ConvTranspose2d


'''
Some typical settings from the paper
                    m = 2           m = 3         m = 4
d = 48, s = 12 | 32.87 (8832) | 32.88 (10128) | 33.08 (11424)
d = 56, s = 12 | 33.00 (9872) | 32.97 (11168) | 33.16 (12464)
d = 48, s = 16 | 32.95 (11232)| 33.10 (13536) | 33.18 (15840)
d = 56, s = 16 | 33.01 (12336)| 33.12 (14640) | 33.17 (16944)
'''


class Model(nn.Module):
    def __init__(self, d, s):
        super().__init__()
        self.d = d
        self.s = s
        # What is
        self.conv1 = Conv2d(3, kernel_size=5, out_channels=self.d)
        self.conv2 = Conv2d(self.d, kernel_size=1, out_channels=self.s)
        self.conv3 = Conv2d(self.s, kernel_size=3, out_channels=self.s)
        self.conv4 = Conv2d(self.s, kernel_size=1, out_channels=self.d)
        self.deconv = ConvTranspose2d(self.d, kernel_size=9, out_channels=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv31(x)
        x = self.conv4(x)
        return self.deconv(x)
