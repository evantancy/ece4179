import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, mode="", transform=None):
        data = np.load("hw3.npz")
        if "train" in mode:
            # trainloader
            self.images = data["arr_0"]
            self.labels = data["arr_1"]
        elif "val" in mode:
            # valloader
            self.images = data["arr_2"]
            self.labels = data["arr_3"]
        elif "test" in mode:
            # testloader
            self.images = data["arr_4"]
            self.labels = data["arr_5"]
        else:
            raise ValueError("mode should be 'train', 'val' or 'test'")

        # BUG FIXED: arr_N are all np.uint8,
        # T.ToTensor() WILL NOT convert np.float32
        self.images = np.uint8(self.images)

        # DO NOT DEFINE OTHER TRANSFORMS HERE use nn.sequential
        # see https://github.com/pytorch/vision/releases/tag/v0.8.0
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.images[idx, :]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
