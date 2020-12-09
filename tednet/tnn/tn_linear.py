# -*- coding: UTF-8 -*-

from typing import Union

import torch
import torch.nn as nn

import numpy as np

from .tn_module import _TNBase

__all__ = ["_TNLinear"]


class _TNLinear(_TNBase):
    def __init__(self, in_shape: Union[list, np.ndarray], out_shape: Union[list, np.ndarray],
                 ranks: Union[list, np.ndarray], bias=True):
        """The Tensor Decomposition Linear.

        Parameters
        ----------
        in_shape : Union[list, numpy.ndarray]
                 1-D param :math:`\in \mathbb{R}^m`. The decomposition shape of feature in
        out_shape : Union[list, numpy.ndarray]
                 1-D param :math:`\in \mathbb{R}^n`. The decomposition shape of feature out
        ranks : Union[list, numpy.ndarray]
                 1-D param :math:`\in \mathbb{R}^r`. The ranks of linear
        bias : bool
                 use bias of linear or not. ``True`` to use, and ``False`` to not use
        """
        super(_TNLinear, self).__init__(in_shape=in_shape, out_shape=out_shape, ranks=ranks, bias=bias)

    def forward(self, inputs):
        """Tensor linear forwarding method.

        Parameters
        ----------
        inputs : torch.Tensor
                 tensor :math:`\in \mathbb{R}^{b \\times C}`

        Returns
        -------
        torch.Tensor
            tensor :math:`\in \mathbb{R}^{b \\times C'}`
        """
        # inputs: [b, C] -> res: [b, C']
        res = self.tn_contract(inputs)
        if self.bias is not None:
            # fused op is marginally faster
            res = torch.add(self.bias, res)

        return res
