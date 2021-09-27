import multiprocessing
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
n_workers = 0
torch.backends.cudnn.benchmark = True

import pprint

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colorbar import ColorbarBase
from matplotlib.pyplot import cm
from sklearn.metrics import confusion_matrix
from torchsummary import summary


class STLData(Dataset):
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

        # self.images = np.float32(self.images) / 1.0
        # BUG FIXED: arr_N are all np.uint8,
        # T.ToTensor() WILL NOT convert np.float32
        self.images = np.uint8(self.images)
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

        label = torch.from_numpy(np.array(label)).long()
        return sample, label


class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()

        self.blocks = self._build_blocks()
        # global average pooling
        # since the output of our conv blocks is (6,6)
        self.gap = nn.AvgPool2d(kernel_size=6, stride=1)
        self.fc1 = nn.Linear(192, 10)

        self._name = self.__class__.__name__

    def _build_blocks(self):
        conv_blk_dims = [3, 32, 64, 128, 192]
        blocks_list = []
        for i in range(len(conv_blk_dims) - 1):
            conv_block = self._create_conv_block(conv_blk_dims[i], conv_blk_dims[i + 1])
            named_block = (f"Conv-Blk-{i+1}", conv_block)
            # blocks_list.append(conv_block)
            blocks_list.append(named_block)

        # return nn.Sequential(*blocks_list)
        return nn.Sequential(OrderedDict(blocks_list))

    def _create_conv_block(self, in_channels, out_channels):
        """Create conv_block based on in/out channels"""
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1),
            nn.ReLU(),
        )
        return conv_block

    def forward(self, x):
        x = self.blocks(x)
        x = self.gap(x)
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


from torch_lr_finder import LRFinder

tmp_loader = DataLoader(
    STLData("train", T.ToTensor()),
    batch_size=128,
    num_workers=0,
    shuffle=False,
    pin_memory=False,
)

model = DeepCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
lr_finder = LRFinder(model, optimizer, criterion, device=device)
lr_finder.range_test(tmp_loader, end_lr=1e-1, num_iter=1000)
lr_finder.plot()  # to inspect the loss-learning rate graph
lr_finder.reset()  # to reset the model and optimizer to their initial state
