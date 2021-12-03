from collections import OrderedDict

import torch
import torch.nn as nn

n_workers = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = False

""" Read report Section 2.3 for design choices """


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        self.blocks = self._build_blocks()
        self.fc1 = nn.Linear(384, 192)
        self.fc2 = nn.Linear(192, 10)
        self.ap = nn.AvgPool2d(kernel_size=6, stride=1, padding=0)
        self.mp = nn.MaxPool2d(kernel_size=6, stride=1, padding=0)
        # # (6,6) -> (3,3)
        # self.ap = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        # # (3,3) -> (1,1)
        # self.mp = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.25)

        self._name = self.__class__.__name__

    def _build_blocks(self):
        conv_blk_dims = [3, 32, 64, 128, 192]
        blocks_list = []
        for i in range(len(conv_blk_dims) - 1):
            conv_block = self._create_conv_block(conv_blk_dims[i], conv_blk_dims[i + 1])
            blocks_list.append((f"Conv-Blk-{i+1}", conv_block))

        return nn.Sequential(OrderedDict(blocks_list))

    def _create_conv_block(self, in_channels, out_channels):
        """Create conv_block based on in/out channels"""
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (5, 5), stride=2, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.25),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.25),
        )
        return conv_block

    def forward(self, x):
        x = self.blocks(x)
        # x = self.ap(x)
        # x = self.mp(x)

        # concatenate pool2d to preserve more information
        x = torch.cat([self.mp(x), self.ap(x)], dim=1)
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)

        x = self.lrelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
