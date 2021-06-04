from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import binvox_rw
import random
import torch

import mcubes
import trimesh
from skimage import measure


class PointCloudDataset(object):
    def __init__(self, root, Np=4096, class_id='03001627'):
        self.root = Path(root)
        self.paths = sorted((self.root / class_id).glob('**/*.binvox'))
        self.len = len(self.paths)
        self.Np = Np

        res = 32
        self.vertex_idxs = torch.arange(res**3) 
        self.vertex_coords = torch.stack(torch.meshgrid(*[torch.linspace(-1, 1, res) for _ in range(3)]), -1).reshape(-1, 3)
        self[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        path = self.paths[idx]
        with open(path, 'rb') as f:
            m1 = binvox_rw.read_as_3d_array(f)
        data = torch.from_numpy(m1.data).float().reshape(-1)

        import ipdb; ipdb.set_trace(context=15)
        verts, faces, normals, values = measure.marching_cubes((1-torch.from_numpy(m1.data).float()).numpy())
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        export = trimesh.exchange.obj.export_obj(mesh)
        with open('ho.obj', 'wb') as f:
            trimesh.util.write_encoded(f, export)
        import ipdb; ipdb.set_trace(context=15)

        loc_ones = self.vertex_idxs[data == 1]
        loc_zeros = self.vertex_idxs[data != 1]
        N_ones = min(loc_ones.shape[0], int(self.Np*0.8))
        N_zeros = self.Np - N_ones
        
        loc_ones = loc_ones[torch.randperm(loc_ones.shape[0])[:N_ones]]
        loc_zeros = loc_zeros[torch.randperm(loc_zeros.shape[0])[:N_zeros]]

        sampled_y = torch.cat([data[loc_ones], data[loc_zeros]]).reshape(self.Np, 1)
        sampled_x = torch.cat([self.vertex_coords[loc_ones], self.vertex_coords[loc_zeros]])
        # sampled_idxs = random.sample(self.vertex_idxs.tolist(), k=self.Np)
        # sampled_y = data.reshape(-1, 1)[sampled_idxs]
        # sampled_x = self.vertex_coords[sampled_idxs]
        return sampled_x, sampled_y 


def iterate(dloader):
    while True:
        for data in dloader:
            yield data


if __name__ == '__main__':
    dset = PointCloudDataset('data/ShapeNetVox32/')
    import ipdb; ipdb.set_trace(context=15)
