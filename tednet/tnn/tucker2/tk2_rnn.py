# -*- coding: UTF-8 -*-

from typing import Union

import numpy as np
from numpy import ndarray

import torch.nn as nn

from ..tn_rnn import _TNLSTM

from .base import TK2Linear


class TK2LSTM(_TNLSTM):
    def __init__(self, in_shape: Union[list, np.ndarray], hidden_size: int, ranks: Union[list, np.ndarray],
                 drop_ih: float = 0.3, drop_hh: float = 0.35):
        """LSTM based on Tucker-2.

        +-------------------+--------------------+
        | input length      | ranks length       |
        +===================+====================+
        | 1                 | 1                  |
        +-------------------+--------------------+
        | 3                 | 2                  |
        +-------------------+--------------------+

        Parameters
        ----------
        in_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m, m \in \{1, 3\}`. The input shape of LSTM
        hidden_size : int
                The hidden size of LSTM
        ranks : Union[list, numpy.ndarray]
                1-D param. The ranks of linear
        drop_ih : float
                The dropout rate of input-to-hidden door
        drop_hh : float
                The dropout rate of hidden-to-hidden door
        """
        self.input_size = np.prod(in_shape)
        self.hidden_size = hidden_size
        fc_size = hidden_size * 4
        tn_block = TK2Linear(in_shape, fc_size, ranks)
        super(TK2LSTM, self).__init__(hidden_size, tn_block, drop_ih, drop_hh)
        self.reset_ih()

    def reset_ih(self):
        """Reset parameters of input-to-hidden layer.
        """
        for weight in self.cell.weight_ih.parameters():
            nn.init.trunc_normal_(weight.data, 0., 0.5)

        if self.cell.weight_ih.bias is not None:
            nn.init.zeros_(self.cell.weight_ih.bias)
