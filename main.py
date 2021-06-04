import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

from tqdm import tqdm
from pathlib import Path

from models.network import HyperNetwork, FunctionalRepresentation, Discriminator
from dataset import PointCloudDataset, iterate
from utils import draw_pointcloud

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='run', type=str)
parser.add_argument('--glr', default=2e-5, type=float)
parser.add_argument('--dlr', default=8e-5, type=float)
parser.add_argument('--batch_size', default=12, type=int)
parser.add_argument('--Np', default=4096, type=int)

parser.add_argument('--zdim', default=64, type=int)
parser.add_argument('--wdim', default=128, type=int)
parser.add_argument('--xdim', default=3, type=int)
parser.add_argument('--ydim', default=1, type=int)
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--w_dreg', default=10, type=float)
parser.add_argument('--disc_type', default='pointconv', type=str, choices=['pointconv', 'settsfm'])

parser.add_argument('--max_iter', default=1000000, type=int)
parser.add_argument('--print_iter', default=1, type=int)
parser.add_argument('--log_scalar_iter', default=10, type=int)
parser.add_argument('--log_image_iter', default=100, type=int)
parser.add_argument('--log_nsample', default=4, type=int)

args = parser.parse_args()

device = 'cuda'
zdim = args.zdim 
wdim = args.wdim 
xdim = args.xdim
ydim = args.ydim
num_layers = args.num_layers
bs = args.batch_size
Np = args.Np
glr = args.glr
dlr = args.dlr
name = args.name

global_iter = 0
max_iter = args.max_iter 
print_iter = args.print_iter
log_scalar_iter = args.log_scalar_iter
log_image_iter = args.log_image_iter
log_nsample = args.log_nsample

exp_dir = Path('outputs') / name
log_dir = exp_dir / 'tensorboard'
ckpt_dir = exp_dir / 'ckpt'

dset = PointCloudDataset(root='data/ShapeNetVox32/', Np=Np)
dloader = DataLoader(dset, bs, shuffle=True, drop_last=True)
dloader = iterate(dloader)

hypn = HyperNetwork(zdim, xdim, wdim, ydim, num_layers).to(device)
gen = FunctionalRepresentation(xdim, ydim, wdim, num_layers).to(device)
if args.disc_type == 'pointconv':
    from models.network import Discriminator
    disc = Discriminator(xdim, ydim).to(device)
elif args.disc_type == 'settsfm':
    from models.sets import SetTransformer as Discriminator
    disc = Discriminator(xdim+ydim, dim_hidden=256, num_heads=4, num_inds=16).to(device)

g_optim = torch.optim.Adam(list(gen.parameters()) + list(hypn.parameters()),
                           lr=glr, betas=(0.5, 0.999)) 
d_optim = torch.optim.Adam(list(disc.parameters()),
                           lr=dlr, betas=(0.5, 0.999)) 

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

    d_optim.zero_grad()

    x.requires_grad = True
    y_real.requires_grad = True
    d_fake = disc(x, y_fake.detach())
    d_real = disc(x, y_real)

    # R1 reguralization
    d_pred_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)) 
    d_pred_real = F.softplus(-d_real).mean()
    (d_pred_fake+d_pred_real).backward(retain_graph=True)
   
    grad_real = grad(d_pred_real.sum(), (x, y_real), create_graph=True)[0]
    d_reg = grad_real.reshape(grad_real.size(0), -1).norm(2, 1)**2
    d_reg = 1/2* d_reg.mean()

    (args.w_dreg*d_reg).backward()
    d_optim.step()
    d_loss = d_pred_fake + d_pred_real

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
        yf = (y_fake.clamp(-1, 1).add(1).div(2) >= 0.5)[:log_nsample].data.cpu()
        real = draw_pointcloud(x, yr)
        fake = draw_pointcloud(x, yf)
        writer.add_images('real', real, global_step=global_iter)
        writer.add_images('fake', fake, global_step=global_iter)
        writer.flush()

writer.close()
