# -*- coding: UTF-8 -*-

""" Difference between this model and usual resnet
The first conv kernel is 3 and stride is 1, while another is 7 and stride is 2.
"""

from typing import Union

import numpy as np
from numpy import ndarray

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import TK2Conv2D, TK2Linear
from ..tn_module import LambdaLayer


class TK2Block(nn.Module):
    def __init__(self, c_in: int, c_out: int, r: int, stride: int = 1, option='A'):
        """Tucker-2 Block.

        Parameters
        ----------
        c_in : int
                The input channel size.
        c_out : int
                The output channel size.
        r : int
                The rank of this block.
        stride : int
                The conv stride
        option : str
                Set "A" or "B" to choose the shortcut type
        """
        super(TK2Block, self).__init__()
        in_planes = c_in
        planes = c_out


        self.conv1 = TK2Conv2D(c_in, c_out, [r, r], 3, padding=1,
                              stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = TK2Conv2D(c_out, c_out, [r, r], 3,
                              stride=1,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     TK2Conv2D(c_in, c_out, [r, r],
                              kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(planes)
                )

    def forward(self, inputs: Tensor) -> Tensor:
        """Forwarding method.

        Parameters
        ----------
        inputs : torch.Tensor
                tensor :math:`\in \mathbb{R}^{b \\times C \\times H \\times W}`

        Returns
        -------
        torch.Tensor
            tensor :math:`\in \mathbb{R}^{b \\times C' \\times H' \\times W'}`
        """
        out = F.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(inputs)
        out = F.relu(out)

        return out


class TK2ResNet(nn.Module):
    def __init__(self, block, rs: Union[list, np.ndarray], num_blocks: list, num_classes:int):
        """ResNet based on Tensor Ring.

        Parameters
        ----------
        block :
                The block class of ResNet
        rs : Union[list, numpy.ndarray]
                rs :math:`\in \mathbb{R}^{6}`. The ranks of network
        num_blocks : list
                The number of each layer
        num_classes : int
                The number of classes
        """
        super(TK2ResNet, self).__init__()
        assert len(rs) == 7, "The length of ranks should be 7."

        self.conv1 = TK2Conv2D(3, 16, [rs[0], rs[0]], kernel_size=3, stride=1,
                                           padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, 16, rs[1], num_blocks[0])
        self.down_sample2 = block(16, 32, rs[2], stride=2)
        self.layer2 = self._make_layer(block, 32, 32, rs[3], num_blocks[1]-1)
        self.down_sample3 = block(32, 64, rs[4], stride=2)
        self.layer3 = self._make_layer(block, 64, 64, rs[5], num_blocks[2]-1)

        self.linear = TK2Linear([4, 4, 4], num_classes, [rs[6], rs[6]], bias=True)

    def _make_layer(self, block, c_in: int, c_out: int, r: int, blocks: int) -> nn.Sequential:
        """Make each block layer.

        Parameters
        ----------
        block :
                The block class of ResNet
        c_in : int
                The input channel size
        c_out : int
                The output channel size
        r : int
                The rank of this block
        blocks : int
                The number of block

        Returns
        -------
        torch.nn.Sequential
            The block network
        """
        strides = [1]*blocks
        layers = []
        for stride in strides:
            layers.append(block(c_in, c_out, r, stride=stride))

        return nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forwarding method.

        Parameters
        ----------
        inputs : torch.Tensor
                tensor :math:`\in \mathbb{R}^{b \\times C \\times H \\times W}`

        Returns
        -------
        torch.Tensor
            tensor :math:`\in \mathbb{R}^{b \\times num\_classes}`
        """
        out = F.relu(self.bn1(self.conv1(inputs)))
        out = self.layer1(out)
        out = self.down_sample2(out)
        out = self.layer2(out)
        out = self.down_sample3(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class TK2ResNet20(TK2ResNet):
    def __init__(self, rs: Union[list, np.ndarray], num_classes: int):
        """ResNet-20 based on Tucker-2.

        Parameters
        ----------
        rs : Union[list, numpy.ndarray]
                rs :math:`\in \mathbb{R}^{7}`. The ranks of network
        num_classes : int
                The number of classes
        """
        super(TK2ResNet20, self).__init__(block=TK2Block, rs=rs, num_blocks=[3, 3, 3], num_classes=num_classes)


class TK2ResNet32(TK2ResNet):
    def __init__(self, rs: Union[list, np.ndarray], num_classes: int):
        """ResNet-32 based on Tucker-2.

        Parameters
        ----------
        rs : Union[list, numpy.ndarray]
                rs :math:`\in \mathbb{R}^{7}`. The ranks of network
        num_classes : int
                The number of classes
        """
        super(TK2ResNet32, self).__init__(block=TK2Block, rs=rs, num_blocks=[5, 5, 5], num_classes=num_classes)
