import torch
import torch.nn as nn
import torch.nn.functional as F

from modsiren import SirenNet

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


#def __init__(self, dim_in, dim_hidden, dim_out, latent_dim, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
class Generator(nn.Module):
    def __init__(self, zdim, wdim, xdim, ydim, num_layers=5):
        super(Generator, self).__init__()
        self.zdim = zdim
        self.wdim = wdim
        self.xdim = xdim
        self.ydim = ydim
        self.layers = SirenNet(xdim, wdim, ydim, zdim, num_layers)  

    def forward(self, z, x):
        assert z.ndim == 2 # B, zdim 
        assert x.ndim == 3 # B, Np, xdim
        y = self.layers(z, x) # B, Np, 1
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
