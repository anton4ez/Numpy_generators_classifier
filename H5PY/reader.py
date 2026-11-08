import torch
import h5py
from torch.utils.data import Dataset

class HDF5Reader(Dataset):
    def __init__(self, path):
        self.path = path
        self.file = None
        with h5py.File(self.path, 'r') as f:
            self.length = len(f['X'])

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
        x = torch.from_numpy(self.file['X'][index]).to(torch.float32)
        y = torch.tensor(self.file['y'][index], dtype=torch.int64)
        return x, y

    def __len__(self):
        return self.length


class InMemoryHDF5Reader(Dataset):
    def __init__(self, path: str):
        with h5py.File(path, "r") as f:
            self.x = torch.from_numpy(f["X"][:])
            self.y = torch.from_numpy(f["y"][:]).long()

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)
    