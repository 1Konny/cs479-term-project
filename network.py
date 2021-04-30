from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableFourierFeatures(nn.Module):
    def __init__(self, xdim, fdim):
        super(LearnableFourierFeatures, self).__init__()
        self.fdim = fdim
        self.mat = nn.Conv1d(xdim, 2*fdim, 1, 1, 0)

    def forward(self, x):
        assert x.ndim == 3 # B, Np, xdim
        B, Np, _ = x.shape
        x = self.mat(x.reshape(B*Np, -1, 1))
        x = x.reshape(B, Np, -1)
        x_cos = x[:, :, :self.fdim]
        x_cos = torch.cos(2*pi*x_cos)
        x_sin = x[:, :, self.fdim:]
        x_sin = torch.sin(2*pi*x_sin)
        x = torch.cat([x_cos, x_sin], -1) 
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, activation='none'):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if activation == 'none':
            self.activation = nn.Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU(True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, True)
        else:
            raise ValueError()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


class HyperNetwork(nn.Module):
    def __init__(self, zdim, wdim, num_layers=3): 
        super(HyperNetwork, self).__init__()
        self.zdim = zdim
        self.wdim = wdim
        self.num_layers = num_layers
        self.layers = nn.Sequential(
            nn.Linear(zdim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, wdim*num_layers),
            )

    def forward(self, z):
        B = z.shape[0]
        w = self.layers(z).reshape(self.num_layers, B, self.wdim)
        return w


class FunctionalRepresentation(nn.Module):
    def __init__(self, xdim=3, ydim=1, wdim=128, num_layers=3):
        super().__init__()
        self.lff = LearnableFourierFeatures(xdim, wdim)

        layers = []
        for idx in range(num_layers):
            if idx == 0:
                layers += [LinearBlock(2*wdim, wdim, 'relu')]
            else:
                layers += [LinearBlock(wdim, wdim, 'relu')]
        self.layers = nn.Sequential(*layers)
        self.last_layer = LinearBlock(wdim, ydim, 'none')


    def forward(self, ws, x):
        y = self.lff(x)
        for layer, w in zip(self.layers, ws):
            y = layer(y)
            y = y * w.unsqueeze(1)
        y = self.last_layer(y)
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
        xy = torch.cat([x, y], -1)
        out = self.layers(xy)
        return out
