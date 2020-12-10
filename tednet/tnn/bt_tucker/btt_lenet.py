# -*- coding: UTF-8 -*-

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np

from .base import BTTConv2D, BTTLinear


class BTTLeNet5(nn.Module):
    def __init__(self, num_classes: int, rs: Union[list, np.ndarray]):
        """LeNet-5 based on the Block-Term Tucker.

        Parameters
        ----------
        num_classes : int
                The number of classes.
        rs : Union[list, numpy.ndarray]
                The ranks of network.
        """
        super(BTTLeNet5, self).__init__()

        assert len(rs) == 4, "The length of the rank should be 4."

        self.c1 = BTTConv2D([1, 1], [4, 5], [rs[0], rs[0], rs[0], rs[0]], 2, 5, padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.c3 = BTTConv2D([4, 5], [5, 10], [rs[1], rs[1], rs[1], rs[1]], 2, 5)
        self.s4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc5 = BTTLinear([5, 10, 25], [5, 8, 8], [rs[2], rs[2], rs[2]], 2)
        self.fc6 = BTTLinear([5, 8, 8], [num_classes, 1, 1], [rs[3], rs[3], rs[3]], 2)

    def forward(self, inputs: Tensor) -> Tensor:
        """Block-Term Tucker linear forwarding method.

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
