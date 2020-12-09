# -*- coding: UTF-8 -*-

from typing import Union

import torch
import torch.nn as nn

import numpy as np

from .tn_module import _TNBase

__all__ = ["_TNConvNd"]

class _TNConvNd(_TNBase):
    def __init__(self, in_shape: Union[list, np.ndarray], out_shape: Union[list, np.ndarray],
                 ranks: Union[list, np.ndarray], kernel_size: Union[int, tuple], stride=1, padding=0, bias=True):
        """Tensor Decomposition Convolution.

        Parameters
        ----------
        in_shape : Union[list, numpy.ndarray]
                 1-D param :math:`\in \mathbb{R}^m`. The decomposition shape of channel in
        out_shape : Union[list, numpy.ndarray]
                 1-D param :math:`\in \mathbb{R}^n`. The decomposition shape of channel out
        ranks : Union[list, numpy.ndarray]
                 1-D param :math:`\in \mathbb{R}^r`. The ranks of the decomposition
        kernel_size : Union[int, tuple]
                The convolutional kernel size
        stride : int
                The length of stride
        padding : int
                The size of padding
        bias : bool
                 use bias of convolution or not. ``True`` to use, and ``False`` to not use
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        super(_TNConvNd, self).__init__(in_shape=in_shape, out_shape=out_shape, ranks=ranks, bias=bias)

    def forward(self, inputs: torch.Tensor):
        """Tensor convolutional forwarding method.

        Parameters
        ----------
        inputs : torch.Tensor
                 tensor :math:`\in \mathbb{R}^{b \\times C \\times H \\times W}`

        Returns
        -------
        torch.Tensor
            tensor :math:`\in \mathbb{R}^{b \\times H' \\times W' \\times C'}`
        """
        # inputs: [b, C, H, W] -> res: [b, H', W', C']
        res = self.tn_contract(inputs)
        if self.bias is not None:
            res = torch.add(self.bias, res)

        # res: [b, H', W', C'] -> res: [b', C', H', W']
        res = res.permute(0, 3, 1, 2).contiguous()
        res = res.contiguous()

        return res
