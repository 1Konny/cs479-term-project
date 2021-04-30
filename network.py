import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperNetwork(nn.Module):
    def __init__(self, zdim, wdim):
        super(HyperNetwork, self).__init__()
        self.zdim = zdim
        self.wdim = wdim
        self.layers = nn.Sequential(
                nn.Linear(1, 1),
                )

    def forward(self, z):
        assert z.ndim == 2
        B, _ = z.shape
        w = torch.randn(B, self.wdim)
        return w


class Generator(nn.Module):
    def __init__(self, xdim, ydim, wdim):
        super(Generator, self).__init__()
        self.xdim = xdim
        self.ydim = ydim
        self.wdim = wdim
        self.layers = nn.Sequential(
                nn.Linear(1, 1),
                )

    def forward(self, x, w):
        assert x.ndim == 2
        assert x.shape[1] == self.xdim 
        B, _ = x.shape
        y = torch.randn(B, self.ydim)
        return y


class Discriminator(nn.Module):
    def __init__(self, xdim, ydim):
        super(Discriminator, self).__init__()
        self.xdim = xdim
        self.ydim = ydim
        self.layers = nn.Sequential(
                nn.Linear(xdim+ydim, 1),
                )

    def forward(self, x, y):
        B, _ = y.shape
        xy = torch.cat([x, y], -1)
        out = self.layers(xy)
        return out



