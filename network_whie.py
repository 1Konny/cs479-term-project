import torch
import torch.nn as nn
import torch.nn.functional as F
from pointconv_util import PointConvDensitySetAbstraction

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



