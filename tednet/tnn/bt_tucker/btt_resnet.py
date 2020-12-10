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


class BTTBlock(nn.Module):
    expansion = 1

    def __init__(self, in_shape: Union[list, ndarray], out_shape: Union[list, ndarray], r: int, stride: int = 1,
                 downsample=None):
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
        downsample :
                The downsample model. Set None for no model
        """
        super(BTTBlock, self).__init__()
        out_size = np.prod(out_shape)
        self.conv1 = BTTConv2D(in_shape, out_shape, [r for _ in range(len(in_shape)+2)], 2, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = BTTConv2D(out_shape, out_shape, [r for _ in range(len(out_shape)+2)], 2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)

        self.downsample = downsample

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
        residual = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        out += residual
        out = self.relu(out)

        return out


class BTTResNet(nn.Module):
    def __init__(self, block, rs: Union[list, np.ndarray], layers: list, num_classes:int):
        """ResNet based on Block-Term Tucker.

        Parameters
        ----------
        block :
                The block class of ResNet
        rs : Union[list, numpy.ndarray]
                The ranks of network
        layers : list
                The number of each layer
        num_classes : int
                The number of classes
        """
        super(BTTResNet, self).__init__()
        assert len(rs) == 6, "The length of ranks should be 6."

        self.conv1 = BTTConv2D([3, 1, 1], [4, 4, 4], [rs[0], rs[0], rs[0], rs[0], rs[0]], 2, 3, stride=1, padding=3)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, [4, 4, 4], [4, 4, 4], rs[1], layers[0])
        self.layer2 = self._make_layer(block, [4, 4, 4], [4, 4, 8], rs[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, [4, 4, 8], [4, 8, 8], rs[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, [4, 8, 8], [8, 8, 8], rs[4], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if num_classes == 10:
            self.fc = BTTLinear([8, 8, 8 * block.expansion], [10, 1, 1], [rs[5], rs[5], rs[5]], 2)
        elif num_classes == 100:
            self.fc = BTTLinear([8, 8, 8 * block.expansion], [10, 10, 1], [rs[5], rs[5], rs[5], rs[5]], 2)

    def _make_layer(self, block, in_shape: Union[list, ndarray], out_shape: Union[list, ndarray], r: int,
                    blocks: int, stride: int=1) -> nn.Sequential:
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
        stride : int
                The stride of downsample conv

        Returns
        -------
        torch.nn.Sequential
            The block network
        """
        downsample = None
        if stride != 1 or np.prod(in_shape) != np.prod(out_shape):
            in_len = len(in_shape)
            downsample = nn.Sequential(
                # nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BTTConv2D(in_shape, out_shape, [r for _ in range(in_len+2)], 2, 1, padding=0, stride=stride),
                nn.BatchNorm2d(np.prod(out_shape)),
            )

        layers = []
        layers.append(block(in_shape, out_shape, r, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(out_shape, out_shape, r))

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
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class BTTResNet18(BTTResNet):
    def __init__(self, rs: Union[list, np.ndarray], num_classes: int):
        """ResNet-18 based on Block-Term Tucker.

        Parameters
        ----------
        rs : Union[list, numpy.ndarray]
                The ranks of network
        num_classes : int
                The number of classes
        """
        super(BTTResNet18, self).__init__(block=BTTBlock, rs=rs, layers=[2, 2, 2, 2], num_classes=num_classes)


class BTTResNet34(BTTResNet):
    def __init__(self, rs: Union[list, np.ndarray], num_classes: int):
        """ResNet-34 based on Block-Term Tucker.

        Parameters
        ----------
        rs : Union[list, numpy.ndarray]
                The ranks of network
        num_classes : int
                The number of classes
        """
        super(BTTResNet34, self).__init__(block=BTTBlock, rs=rs, layers=[3, 4, 6, 3], num_classes=num_classes)
