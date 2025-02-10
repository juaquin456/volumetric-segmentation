
import os
import torch
import torch.nn as nn
from model import GeneralizedDiceLoss 

class DarknetBottleneck(nn.Module):
    def __init__(self, in_chan, out_chan, add=True):
        super().__init__()

        self.add = add

        self.s = nn.Sequential(
            nn.Conv3d(in_chan, in_chan, kernel_size=1, stride=1, padding=0), 
            nn.Conv3d(in_chan, out_chan, kernel_size=3, stride=1, padding=1)
        )

    def forward(self,x):
        y = self.s(x)
        if self.add:
            y = x + y
        return y


class CSPlayerModule(nn.Module):
    def __init__(self,in_chan, out_chan, k, s, p, N, add=True):
        super().__init__()

        self.c1 = nn.Conv3d(in_chan, in_chan//2, kernel_size=k, stride=s, padding=p) 

        self.a = nn.Sequential(
            nn.Conv3d(in_chan, in_chan//2, kernel_size=k, stride=s, padding=p),
            *[DarknetBottleneck(in_chan//2, in_chan//2, add) for _ in range(N)]
        )

        self.c3 = nn.Conv3d(in_chan, out_chan, kernel_size=k, stride=s, padding=p)


    def forward(self,x):
        y = self.c1(x)
        z = self.a(x)
        x = torch.cat((y,z), dim=1)
        x = self.c3(x)
        return x


def convModule(in_chan, k, s, p, out_chan):
    return nn.Sequential(
        nn.Conv3d(in_chan, out_chan, kernel_size=k, stride=s, padding=p),
        nn.BatchNorm3d(out_chan, eps=1e-03, momentum=0.1),
        nn.SiLU(inplace=True)
    )


class SPPFBottleneck(nn.Module):
    def __init__(self, in_chan, out_chan, k, s, p):
        super().__init__()

        self.c1 = convModule(in_chan, k, s, p, out_chan)

        self.pool1 = nn.MaxPool3d(2, stride=1, padding=0)

        self.pool2 = nn.MaxPool3d(2, stride=1, padding=0)

        # self.pool3 = nn.MaxPool3d(2, stride=1, padding=0)
        
        # self.c2 = convModule(out_chan*4, k, s, p, out_chan)
        self.c2 = convModule(out_chan*3, k, s, p, out_chan)


    def forward(self,x):
        a = self.c1(x)
        b = self.pool1(a)
        c = self.pool2(b)
        # d = self.pool3(c)
        # [n_batch, 4, h, w, d]
        b = nn.functional.interpolate(b, size=tuple(a.shape[2:]), scale_factor=None, mode='nearest')
        c = nn.functional.interpolate(c, size=tuple(a.shape[2:]), scale_factor=None, mode='nearest')
        # d = nn.functional.interpolate(d, size=tuple(a.shape[2:]), scale_factor=None, mode='nearest')

        # x = torch.cat([a, b, c, d], dim=1)
        x = torch.cat([a, b, c], dim=1)
        x = self.c2(x)
        
        return x



class Brainet(nn.Module):
    def __init__(self,n_input=4, n_output=4):
        super().__init__()

        # 240 x 240 x 155

        self.c1 = convModule(n_input, 6, 2, 2, 64) # 120
        self.c2 = convModule(64, 3, 2, 1, 128) # 60
        self.cp1 = CSPlayerModule(128, 128, 3, 2, 1, 3, True)
        self.c3 = convModule(128, 3, 2, 1, 256) # 30
        self.cp2 = CSPlayerModule(256, 256, 1, 1, 0, 6, True)
        # self.c4 = convModule(256, 3, 2, 1, 512) # 15
        # self.cp3 = CSPlayerModule(512, 512, 1, 1, 0, 9, True)
        # self.c5 = convModule(512, 3, 2, 1, 1024) # 7 x 7 x 5
        # self.cp4 = CSPlayerModule(1024, 1024, 1, 1, 0, 3, True)
        self.sppf = SPPFBottleneck(256, 256, 3, 1, 0) 
        
        self.up = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose3d(128, 64, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, kernel_size=7),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, kernel_size=9),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 8, kernel_size=11),
            nn.ReLU(True),
            nn.ConvTranspose3d(8, n_output, kernel_size=13),
            nn.ReLU(True),
            nn.ConvTranspose3d(n_output, n_output, kernel_size=41),
            nn.ReLU(True),
            nn.ConvTranspose3d(n_output, n_output, kernel_size=73),
            nn.ReLU(True),
            nn.ConvTranspose3d(n_output, n_output, kernel_size=83),
            nn.ReLU(True),
        )

        self.xd = nn.ConvTranspose3d(237, 155, kernel_size=1)

    

    def forward(self,x):
        #  [n_batch, 4, h, w, d]
        x = self.c1(x)
        x = self.c2(x)
        x = self.cp1(x)
        x = self.c3(x)
        x = self.cp2(x)
        # x = self.c4(x)
        # x = self.cp3(x)
        # x = self.c5(x)
        # x = self.cp4(x)
        x = self.sppf(x)
        x = self.up(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.xd(x)
        x = x.permute(0, 2, 3, 4, 1)
        return x


import numpy as np

x = torch.tensor(np.random.rand(1, 4, 240, 240, 155).astype(np.float32))
model = Brainet()
criterion = GeneralizedDiceLoss("softmax")