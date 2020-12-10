# -*- coding: UTF-8 -*-

""" Difference between this model and usual resnet
The first conv kernel is 3 and stride is 1, while another is 7 and stride is 2.
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np

from .base import CPConv2D, CPLinear


class CPBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in: int, c_out: int, r: int, stride: int=1, downsample=None):
        """CANDECOMP/PARAFAC Block.

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
        downsample :
                The downsample model. Set None for no model
        """
        super(CPBlock, self).__init__()
        self.conv1 = CPConv2D(c_in, c_out, r, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = CPConv2D(c_out, c_out, r, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)

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


class CPResNet(nn.Module):
    def __init__(self, block, rs: Union[list, np.ndarray], layers: list, num_classes:int):
        """ResNet based on CANDECOMP/PARAFAC.

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
        super(CPResNet, self).__init__()
        assert len(rs) == 6, "The length of ranks should be 6."

        self.conv1 = CPConv2D(3, 64, rs[0], 3, stride=1, padding=3)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, 64, rs[1], layers[0])
        self.layer2 = self._make_layer(block, 64, 128, rs[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 256, rs[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 512, rs[4], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = CPLinear([8, 8, 8 * block.expansion], [num_classes], rs[5])

    def _make_layer(self, block, c_in: int, c_out: int, r: int, blocks: int, stride: int=1) -> nn.Sequential:
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
        stride : int
                The stride of downsample conv

        Returns
        -------
        torch.nn.Sequential
            The block network
        """
        downsample = None
        if stride != 1 or c_in != c_out:
            downsample = nn.Sequential(
                # nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                CPConv2D(c_in, c_out, r, 1, padding=0, stride=stride),
                nn.BatchNorm2d(c_out),
            )

        layers = []
        layers.append(block(c_in, c_out, r, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(c_out, c_out, r))

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


class CPResNet18(CPResNet):
    def __init__(self, rs: Union[list, np.ndarray], num_classes: int):
        """ResNet-18 based on CANDECOMP/PARAFAC.

        Parameters
        ----------
        rs : Union[list, numpy.ndarray]
                The ranks of network
        num_classes : int
                The number of classes
        """
        super(CPResNet18, self).__init__(block=CPBlock, rs=rs, layers=[2, 2, 2, 2], num_classes=num_classes)


class CPResNet34(CPResNet):
    def __init__(self, rs: Union[list, np.ndarray], num_classes: int):
        """ResNet-34 based on CANDECOMP/PARAFAC.

        Parameters
        ----------
        rs : Union[list, numpy.ndarray]
                The ranks of network
        num_classes : int
                The number of classes
        """
        super(CPResNet34, self).__init__(block=CPBlock, rs=rs, layers=[3, 4, 6, 3], num_classes=num_classes)
