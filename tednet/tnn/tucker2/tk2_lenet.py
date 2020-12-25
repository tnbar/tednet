# -*- coding: UTF-8 -*-

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np

from .base import TK2Conv2D, TK2Linear


class TK2LeNet5(nn.Module):
    def __init__(self, num_classes: int, rs: Union[list, np.ndarray]):
        """LeNet-5 based on the Tucker-2.

        Parameters
        ----------
        num_classes : int
                The number of classes
        rs : Union[list, numpy.ndarray]
                The ranks of network.
        """
        super(TK2LeNet5, self).__init__()

        assert len(rs) == 4, "The length of the rank should be 4."

        self.c1 = TK2Conv2D(1, 20, [rs[0], rs[0]], 5, padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.c3 = TK2Conv2D(20, 50, [rs[1], rs[1]], 5)
        self.s4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc5 = TK2Linear([5, 10, 25], 320, [rs[2], rs[2]])
        self.fc6 = TK2Linear([320], num_classes, [rs[3]])

    def forward(self, inputs: Tensor) -> Tensor:
        """forwarding method.

        Parameters
        ----------
        inputs : torch.Tensor
                tensor :math:`\in \mathbb{R}^{b \\times C \\times H \\times W}`

        Returns
        -------
        torch.Tensor
            tensor :math:`\in \mathbb{R}^{b \\times num\_classes}`
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
