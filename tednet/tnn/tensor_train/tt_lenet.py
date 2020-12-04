# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import TTConv2D, TTLinear


class TTLeNet5(nn.Module):
    def __init__(self, num_classes: int, rs: list):
        """
        LeNet-5 based on the Tensor Train.
        @param num_classes: The number of classes.
        @param rs: The ranks of network.
        """
        super(TTLeNet5, self).__init__()

        assert len(rs) == 4, "The length of the rank should be 4."

        self.c1 = TTConv2D([1, 1], [4, 5], [rs[0], rs[0]], 5, padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.c3 = TTConv2D([4, 5], [5, 10], [rs[1], rs[1]], 5)
        self.s4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc5 = TTLinear([5, 5, 5, 10], [4, 4, 4, 5], [rs[2], rs[2], rs[2]])
        self.fc6 = TTLinear([4, 4, 4, 5], [1, 1, 1, num_classes], [rs[3], rs[3], rs[3]])

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forwarding method.
        @param inputs: A Tensor: [b, C, H, W]
        @return: A Tensor: [b, C', H', W']
        """
        out = self.c1(inputs)
        out = F.relu(out)
        out = self.s2(out)

        out = self.c3(out)
        out = F.relu(out)
        out = self.s4(out)

        out = out.view(inputs.size(0), -1)

        out = self.fc5(out)
        out = F.relu(out)

        out = self.fc6(out)
        return out
