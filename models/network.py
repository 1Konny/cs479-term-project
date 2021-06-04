from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointconv_util import PointConvDensitySetAbstraction


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
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid() 
        elif activation == 'tanh':
            self.activation = nn.Tanh() 
        else:
            raise ValueError()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


# class HyperNetwork(nn.Module):
#     def __init__(self, zdim, wdim, num_layers=3): 
#         super(HyperNetwork, self).__init__()
#         self.zdim = zdim
#         self.wdim = wdim
#         self.num_layers = num_layers
#         self.layers = nn.Sequential(
#             nn.Linear(zdim, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 512),
#             nn.ReLU(True),
#             nn.Linear(512, wdim*num_layers),
#             )

#     def forward(self, z):
#         B = z.shape[0]
#         w = self.layers(z).reshape(self.num_layers, B, self.wdim)
#         return w


# class FunctionalRepresentation(nn.Module):
#     def __init__(self, xdim=3, ydim=1, wdim=128, num_layers=3):
#         super().__init__()
#         self.lff = nn.Sequential(
#             nn.Linear(xdim, wdim),
#             nn.LeakyReLU(0.2, True),
#             )

#         layers = []
#         for idx in range(num_layers):
#             if idx == 0:
#                 layers += [LinearBlock(wdim, wdim, 'lrelu')]
#             else:
#                 layers += [LinearBlock(wdim, wdim, 'lrelu')]
#         self.layers = nn.Sequential(*layers)
#         self.last_layer = LinearBlock(wdim, ydim, 'none')

#     def forward(self, ws, x):
#         y = self.lff(x)
#         for layer, w in zip(self.layers, ws):
#             y = layer(y)
#             y = y * w.unsqueeze(1)
#         y = self.last_layer(y)
#         return y



class HyperNetwork(nn.Module):
    def __init__(self, zdim, xdim, wdim, ydim, num_layers=3): 
        super(HyperNetwork, self).__init__()
        self.zdim = zdim
        self.xdim = xdim
        self.wdim = wdim
        self.ydim = ydim
        self.num_layers = num_layers
        num_w_params = xdim*wdim + ydim*wdim + wdim*wdim*(num_layers-1)
        num_b_params = wdim*num_layers + ydim
        self.layers = nn.Sequential(
            nn.Linear(zdim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            )
        out_params = {
            'w_in': nn.Linear(512, xdim*wdim),
            'b_in': nn.Linear(512, wdim),
            'w_out': nn.Linear(512, ydim*wdim),
            'b_out': nn.Linear(512, ydim),
            }
        for i in range(1, num_layers):
            out_params['w_%d' % i] = nn.Linear(512, wdim*wdim)
            out_params['b_%d' % i] = nn.Linear(512, wdim)
        self.out_params = nn.ModuleDict(out_params)

    def forward(self, z):
        h = self.layers(z)
        w = {k:self.out_params[k](h) for k in self.out_params}
        return w


class FunctionalRepresentation(nn.Module):
    def __init__(self, xdim=3, ydim=1, wdim=128, num_layers=3):
        super().__init__()
        self.xdim = xdim
        self.wdim = wdim
        self.ydim = ydim
        self.num_layers = num_layers
        self.actv_h = nn.LeakyReLU(0.2, True)
        self.actv_y = nn.Tanh()

    def forward(self, ws, x):
        B = x.shape[0]
        h = x @ ws['w_in'].reshape(B, self.xdim, self.wdim) + ws['b_in'].reshape(B, 1, self.wdim)
        h = self.actv_h(h) 
        for i in range(1, self.num_layers):
            h = h @ ws['w_%d' % i].reshape(B, self.wdim, self.wdim) + ws['b_%d' % i].reshape(B, 1, self.wdim)
            h = self.actv_h(h) 
        h = h @ ws['w_out'].reshape(B, self.wdim, self.ydim) + ws['b_out'].reshape(B, 1, self.ydim)
        y = self.actv_y(h)
        return y


class PointConv(nn.Module):
    def __init__(self, npoint, nsample, feature_dim, mlp, bandwith, group_all=False):
        super(PointConv, self).__init__()
        self.layer = PointConvDensitySetAbstraction(\
                npoint=npoint, nsample=nsample, in_channel=feature_dim + 3, mlp=mlp, bandwidth=bandwith, group_all=group_all)

    def forward(self, x, y):
        return self.layer(x, y)


class Discriminator(nn.Module):
    """
        x : 3D coordinates (B, N, 3) 
        y : Occupancy value (B, N, 1)
    """

    def __init__(self, xdim, ydim, hdim=[64,128,256]):
        super(Discriminator, self).__init__()
        self.xdim = xdim
        self.ydim = ydim
        self.hdim = hdim
        self.nsample = 3**self.xdim
        self.ds_rate = 2**self.xdim
        self.layer1 = PointConv(512, self.nsample, self.ydim, [hdim[0]//2, hdim[0]//2, hdim[0]], bandwith=0.1)
        self.layer2 = PointConv(64, self.nsample, hdim[0], [hdim[1]//2, hdim[1]//2, hdim[1]], bandwith=0.1)
        self.bn2 = nn.BatchNorm1d(self.hdim[1])
        self.layer3 = PointConv(8, self.nsample, hdim[1], [hdim[2]//2, hdim[2]//2, hdim[2]], bandwith=0.1)
        self.bn3 = nn.BatchNorm1d(self.hdim[2])
        self.layer4 = PointConv(1, min(self.nsample, 8), hdim[2], [hdim[2]//2, hdim[2]//4, 1],  bandwith=0.1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, y):
        # To fit into pointConv network
        if x.shape[-1] == self.xdim:
            x = x.permute(0,2,1)
            y = y.permute(0,2,1)

        B, _, N = y.shape
        x_out, y_out = self.layer1(x, y)
        y_out = self.lrelu(y_out)
        x_out, y_out = self.layer2(x_out, y_out)
        y_out = self.lrelu(self.bn2(y_out))
        x_out, y_out = self.layer3(x_out, y_out)
        y_out = self.lrelu(self.bn3(y_out))
        x_out, y_out = self.layer4(x_out, y_out)
        return y_out
