# -*- coding: UTF-8 -*-

from typing import Union

import numpy as np
from numpy import ndarray

import torch.nn as nn

from ..tn_rnn import _TNLSTM

from .base import BTTLinear


class BTTLSTM(_TNLSTM):
    def __init__(self, in_shape: Union[list, np.ndarray], hidden_shape: Union[list, np.ndarray],
                 ranks: Union[list, np.ndarray], block_num: int, drop_ih: float = 0.3, drop_hh: float = 0.35):
        """LSTM based on Block-Term Tucker.

        Parameters
        ----------
        in_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The input shape of LSTM
        hidden_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The hidden shape of LSTM
        ranks : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The ranks of linear
        block_num : int
                The number of blocks
        drop_ih : float
                The dropout rate of input-to-hidden door
        drop_hh : float
                The dropout rate of hidden-to-hidden door
        """
        self.input_size = np.prod(in_shape)
        self.hidden_size = np.prod(hidden_shape)
        hidden_shape[0] *= 4
        tn_block = BTTLinear(in_shape, hidden_shape, ranks, block_num)
        super(BTTLSTM, self).__init__(self.hidden_size, tn_block, drop_ih, drop_hh)
        self.reset_ih()

    def reset_ih(self):
        """Reset parameters of input-to-hidden layer.
        """
        for weight in self.cell.weight_ih.parameters():
            nn.init.trunc_normal_(weight.data, 0., 0.5)

        if self.cell.weight_ih.bias is not None:
            nn.init.zeros_(self.cell.weight_ih.bias)
