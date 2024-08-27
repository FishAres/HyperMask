import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import vmap


class HyperNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(HyperNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(
            64 * 7 * 7, out_channels * in_channels * kernel_size * kernel_size
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        mask = torch.sigmoid(self.fc(x))
        return mask.view(x.size(0), -1)


class PrimaryNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(PrimaryNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x, mask):
        masked_weight = self.conv.weight * mask.view_as(self.conv.weight)
        return F.conv2d(x, masked_weight, self.conv.bias)


class GatedConvNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=5):
        super(GatedConvNet, self).__init__()
        self.hyper = HyperNetwork(in_channels, out_channels, kernel_size)
        self.primary = PrimaryNetwork(in_channels, out_channels, kernel_size)

    def apply_primary(self, sample, sample_mask):
        return self.primary(
            sample.unsqueeze(0), sample_mask.view_as(self.primary.conv.weight)
        )

    def forward(self, x):
        masks = self.hyper(x)
        return vmap(self.apply_primary)(x, masks).squeeze(1)
