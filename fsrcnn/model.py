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
    def __init__(self, d, s, m=2, n=2):
        super().__init__()
        self.d = d
        self.s = s
        self.m = m
        self.n = n
        # What is
        self.conv1 = Conv2d(3, kernel_size=5, out_channels=self.d, padding='same')
        self.activation1 = nn.PReLU()
        self.conv2 = Conv2d(self.d, kernel_size=1, out_channels=self.s, padding='same')
        self.activation2 = nn.PReLU()
        self.conv3 = []
        self.activation3 = []
        for _ in range(self.m):
            self.conv3.append(Conv2d(self.s, kernel_size=3, out_channels=self.s, padding='same'))
            self.activation3.append(nn.PReLU())
        self.conv4 = Conv2d(self.s, kernel_size=1, out_channels=self.d, padding='same')
        self.activation4 = nn.PReLU()
        self.deconv = ConvTranspose2d(self.d, kernel_size=9, out_channels=3, stride=self.n)

    def float(self):
        super().float()
        self.conv1.float()
        self.conv2.float()
        for conv in self.conv3:
            conv.float()
        self.conv4.float()
        self.deconv.float()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # We need to manually send some layers to the appropriate device
        self.conv1.to(*args, **kwargs)
        self.conv2.to(*args, **kwargs)
        for conv in self.conv3:
            conv.to(*args, **kwargs)
        self.conv4.to(*args, **kwargs)
        self.deconv.to(*args, **kwargs)
        return self

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        for conv, prelu in zip(self.conv3, self.activation3):
            x = conv(x)
            x = prelu(x)
        x = self.conv4(x)
        x = self.activation4(x)
        return self.deconv(x)
