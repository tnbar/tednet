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

from .base import BTTConv2D, BTTLinear
from ..tn_module import LambdaLayer


class BTTBlock(nn.Module):
    def __init__(self, in_shape: Union[list, ndarray], out_shape: Union[list, ndarray], r: int, stride: int = 1, option='A'):
        """Block-Term Tucker Block.

        Parameters
        ----------
        in_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The input shape of block
        out_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The output shape of block
        r : int
                The rank of this block
        stride : int
                The conv stride
        option : str
                Set "A" or "B" to choose the shortcut type
        """
        super(BTTBlock, self).__init__()
        in_planes = np.prod(in_shape)
        planes = np.prod(out_shape)


        self.conv1 = BTTConv2D(in_shape, out_shape, [r for _ in range(len(in_shape)+2)], 2, 3, padding=1,
                              stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BTTConv2D(out_shape, out_shape, [r for _ in range(len(out_shape)+2)], 2, 3,
                              stride=1, padding=1, bias=False)
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
                     BTTConv2D(in_shape, out_shape, [r for _ in range(len(in_shape)+2)], 2,
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


class BTTResNet(nn.Module):
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
        super(BTTResNet, self).__init__()
        assert len(rs) == 7, "The length of ranks should be 7."

        self.conv1 = BTTConv2D([3, 1, 1], [4, 2, 2], [rs[0], rs[0], rs[0], rs[0], rs[0]], 2, kernel_size=3, stride=1,
                                           padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, [4, 2, 2], [4, 2, 2], rs[1], num_blocks[0])
        self.down_sample2 = block([4, 2, 2], [4, 4, 2], rs[2], stride=2)
        self.layer2 = self._make_layer(block, [4, 4, 2], [4, 4, 2], rs[3], num_blocks[1]-1)
        self.down_sample3 = block([4, 4, 2], [4, 4, 4], rs[4], stride=2)
        self.layer3 = self._make_layer(block, [4, 4, 4], [4, 4, 4], rs[5], num_blocks[2]-1)

        if num_classes == 10:
            self.linear = BTTLinear([4, 4, 4], [10, 1, 1], [rs[6], rs[6], rs[6]], 2, bias=True)
        elif num_classes == 100:
            self.linear = BTTLinear([4, 4, 4], [10, 10, 1], [rs[6], rs[6], rs[6]], 2, bias=True)

    def _make_layer(self, block, in_shape: Union[list, ndarray], out_shape: Union[list, ndarray], r: int,
                    blocks: int) -> nn.Sequential:
        """Make each block layer.

        Parameters
        ----------
        block :
                The block class of ResNet
        in_shape : Union[list, numpy.ndarray]
                The input shape of block
        out_shape : Union[list, numpy.ndarray]
                The output shape of block
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
            layers.append(block(in_shape, out_shape, r, stride=stride))

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


class BTTResNet20(BTTResNet):
    def __init__(self, rs: Union[list, np.ndarray], num_classes: int):
        """ResNet-20 based on Block-Term Tucker.

        Parameters
        ----------
        rs : Union[list, numpy.ndarray]
                rs :math:`\in \mathbb{R}^{7}`. The ranks of network
        num_classes : int
                The number of classes
        """
        super(BTTResNet20, self).__init__(block=BTTBlock, rs=rs, num_blocks=[3, 3, 3], num_classes=num_classes)


class BTTResNet32(BTTResNet):
    def __init__(self, rs: Union[list, np.ndarray], num_classes: int):
        """ResNet-32 based on Block-Term Tucker.

        Parameters
        ----------
        rs : Union[list, numpy.ndarray]
                rs :math:`\in \mathbb{R}^{7}`. The ranks of network
        num_classes : int
                The number of classes
        """
        super(BTTResNet32, self).__init__(block=BTTBlock, rs=rs, num_blocks=[5, 5, 5], num_classes=num_classes)
