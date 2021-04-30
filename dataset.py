import torch

class PointCloudDataset(object):
    def __init__(self):
        pass

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        xdim = 3
        ydim = 1

        x = torch.randn(xdim)
        y = torch.randn(ydim)

        return x, y 


def iterate(dloader):
    while True:
        for data in dloader:
            yield data
