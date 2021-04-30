import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from pathlib import Path

from network import HyperNetwork, FunctionalRepresentation, Discriminator
from dataset import PointCloudDataset, iterate
from utils import draw_pointcloud

device = 'cuda'
zdim = 64
wdim = 128
xdim = 3
ydim = 1
num_layers = 3
bs = 16
Np = 4096
glr = 0.001
dlr = 0.003
max_iter = 10000
global_iter = 0
print_iter = 1
log_scalar_iter = 10
# log_image_iter = 500
log_image_iter = 100
log_nsample = 4

# name = 'run2'
# bs = 24
# glr = 2e-5
# dlr = 8e-5

# name = 'run3'
# bs = 24
# glr = 0.0001
# dlr = 0.0003

name = 'run4'
bs = 24
glr = 0.001
dlr = 0.003

# name = 'debug2'
# log_scalar_iter = 1
# log_image_iter = 100

exp_dir = Path('outputs') / name
log_dir = exp_dir / 'tensorboard'
ckpt_dir = exp_dir / 'ckpt'


dset = PointCloudDataset(root='data/ShapeNetVox32/', Np=Np)
dloader = DataLoader(dset, bs, shuffle=True, drop_last=True)
dloader = iterate(dloader)

hypn = HyperNetwork(zdim, wdim, num_layers).to(device)
gen = FunctionalRepresentation(xdim, ydim, wdim, num_layers).to(device)
disc = Discriminator(xdim, ydim).to(device)

g_optim = torch.optim.Adam(list(gen.parameters()) + list(hypn.parameters()),
                           lr=glr) 
d_optim = torch.optim.Adam(list(disc.parameters()),
                           lr=dlr) 

'''
load pretrained here along with global iter
'''

writer = SummaryWriter(log_dir, purge_step=global_iter)

d_reg = 0

pbar = tqdm(range(global_iter, max_iter))
for _  in pbar:
    global_iter += 1
    x, y_real = next(dloader)
    x = x.to(device)
    y_real = y_real.to(device)*2-1

    z = torch.randn(bs, zdim, device=device)
    w = hypn(z)
    y_fake = gen(w, x)

    d_fake = disc(x, y_fake.detach())
    d_real = disc(x, y_real)
    d_loss = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)) + \
             F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))

    d_optim.zero_grad()
    d_loss.backward(retain_graph=True)
    d_optim.step()

    d_fake = disc(x, y_fake)
    g_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))

    g_optim.zero_grad()
    g_loss.backward()
    g_optim.step()

    if global_iter % print_iter == 0:
        pbar.set_description('[%010d] name:%s g_loss: %.4f | d_loss: %.4f | d_reg: %.4f | y_real_mean: %.4f | y_fake_min: %.4f y_fake_mean: %.4f y_fake_max: %.4f' % (
            global_iter, name, g_loss, d_loss, d_reg, y_real.mean(), y_fake.min(), y_fake.mean(), y_fake.max()))

    if global_iter % log_image_iter == 0:
        writer.add_scalar('g_loss', g_loss, global_step=global_iter)
        writer.add_scalar('d_loss', d_loss, global_step=global_iter)
        writer.add_scalar('d_reg', d_reg, global_step=global_iter)

    if global_iter % log_image_iter == 0:
        x = x[:log_nsample].data.cpu()
        yr = y_real[:log_nsample].data.cpu().add(1).div(2)
        yf = (y_fake.add(1).div(2) >= 0.5)[:log_nsample].data.cpu()
        real = draw_pointcloud(x, yr)
        fake = draw_pointcloud(x, yf)
        writer.add_images('real', real, global_step=global_iter)
        writer.add_images('fake', fake, global_step=global_iter)
        writer.flush()

writer.close()
