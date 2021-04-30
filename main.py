import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

from network import Generator, Discriminator, HyperNetwork 
from dataset import PointCloudDataset, iterate



zdim = 64
wdim = 128
xdim = 3
ydim = 1
B = 64
max_iter = 10000
glr = 0.0001
dlr = 0.0003

dset = PointCloudDataset()
dloader = DataLoader(dset, B, shuffle=True)
dloader = iterate(dloader)


gen = Generator(xdim, ydim, wdim)
disc = Discriminator(xdim, ydim)
hypn = HyperNetwork(zdim, wdim)

g_optim = torch.optim.Adam(list(gen.parameters()) + list(hypn.parameters()),
                           lr=glr) 
d_optim = torch.optim.Adam(list(disc.parameters()),
                           lr=glr) 

pbar = tqdm(range(max_iter))
for _  in pbar:
    x_real, y_real = next(dloader)

    z = torch.randn(B, zdim)
    w = hypn(z)
    # x = torch.rand(B, xdim)
    y_fake = gen(x_real, w)

    d_fake = disc(x_real, y_fake)
    d_real = disc(x_real, y_real)
    d_loss = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)) + \
             F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
    d_optim.zero_grad()
    d_loss.backward()
    d_optim.step()

    d_fake = disc(x_real, y_fake)
    g_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
    g_optim.zero_grad()
    g_loss.backward()
    g_optim.step()
