import matplotlib.pyplot as plt
import numpy as np
import binvox_rw
import cv2
import torch

import trimesh
from skimage import measure


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    # https://github.com/hxdengBerkeley/PointCNN.Pytorch/blob/master/provider.py
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


@torch.no_grad()
def draw_pointcloud(x: torch.Tensor, x_mask: torch.Tensor, grid_on=True):
    # https://github.com/jw9730/setvae/blob/master/utils.py
    """ Make point cloud image
    :param x: Tensor([B, N, 3])
    :param x_mask: Tensor([B, N, 1])
    :param grid_on
    :return: Tensor([3 * B, W, H])
    """
    figw, figh = 16., 12.
    W, H = 256, int(256 * figh / figw)

    x = rotate_point_cloud_by_angle(x, 3)
    x_mask = x_mask.squeeze(-1).bool()

    imgs = list()
    for p, m in zip(x, x_mask):
        p = p[m, :]
        p = p

        fig = plt.figure(figsize=(figw, figh))
        ax = fig.gca(projection='3d')
        ax.set_facecolor('xkcd:steel')
        ax.w_xaxis.set_pane_color((0., 0., 0., 1.0))
        ax.w_yaxis.set_pane_color((0., 0., 0., 1.0))
        ax.w_zaxis.set_pane_color((0., 0., 0., 1.0))

        ax.scatter(-p[:, 2], p[:, 0], p[:, 1], color=(1, 1, 1), marker='o', s=100)
        fig.tight_layout()
        fig.canvas.draw()

        buf = fig.canvas.buffer_rgba()
        l, b, w, h = fig.bbox.bounds
        img = np.frombuffer(buf, np.uint8).copy()
        img.shape = int(h), int(w), 4
        img = img[:, :, 0:3]
        img = cv2.resize(img, dsize=(W, H), interpolation=cv2.INTER_CUBIC)  # [H, W, 3]

        imgs.append(torch.tensor(img).transpose(2, 0).transpose(2, 1))  # [3, H, W]
        plt.close(fig)

    return torch.stack(imgs, dim=0)


def of_to_voxel(of, th=0.5):
    return (of.clamp(-1, 1).add(1).div(2) >= th).float() 


def marching_cube(voxels):
    # voxels = (torch.rand_like(voxels) >= 0.5).float()
    meshes = []
    for voxel in voxels:
        try:
            verts, faces, normals, values = measure.marching_cubes_lewiner(1-voxel.float().numpy())
        except RuntimeError:
            meshes += [None]
        else:
            meshes += [trimesh.Trimesh(vertices=verts, faces=faces)]
    return meshes


def save_obj(mesh, filepath):
    export = trimesh.exchange.obj.export_obj(mesh)
    with open(filepath, 'wb') as f:
        trimesh.util.write_encoded(f, export)
