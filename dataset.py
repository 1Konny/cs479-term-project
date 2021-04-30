from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import binvox_rw
import random
import torch

class PointCloudDataset(object):
    def __init__(self, root, Np=4096, class_id='03001627'):
        self.root = Path(root)
        self.paths = sorted((self.root / class_id).glob('**/*.binvox'))
        self.len = len(self.paths)
        self.Np = Np

        res = 32
        self.vertex_idxs = range(res**3) 
        self.vertex_coords = torch.stack(torch.meshgrid(*[torch.linspace(-1, 1, res) for _ in range(3)]), -1).reshape(-1, 3)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        path = self.paths[idx]
        with open(path, 'rb') as f:
            m1 = binvox_rw.read_as_3d_array(f)
        data = torch.from_numpy(m1.data).float()
        sampled_idxs = random.sample(self.vertex_idxs, k=self.Np)
        sampled_y = data.reshape(-1, 1)[sampled_idxs]
        sampled_x = self.vertex_coords[sampled_idxs]
        # sampled_x = torch.from_numpy(rotate_point_cloud_by_angle(sampled_x.unsqueeze(0), 3)).squeeze(0)
        return sampled_x, sampled_y 


def iterate(dloader):
    while True:
        for data in dloader:
            yield data


if __name__ == '__main__':
    dset = PointCloudDataset('data/ShapeNetVox32/')
    import ipdb; ipdb.set_trace(context=15)
