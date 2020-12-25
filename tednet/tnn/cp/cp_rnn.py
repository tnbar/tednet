# -*- coding: UTF-8 -*-

from typing import Union

import numpy as np
from numpy import ndarray

import torch.nn as nn

from ..tn_rnn import _TNLSTM

from .base import CPLinear


class CPLSTM(_TNLSTM):
    def __init__(self, in_shape: Union[list, np.ndarray], hidden_shape: Union[list, np.ndarray],
                 ranks: int, drop_ih: float = 0.3, drop_hh: float = 0.35):
        """LSTM based on CANDECOMP/PARAFAC.

        Parameters
        ----------
        in_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The input shape of LSTM
        hidden_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^n`. The hidden shape of LSTM
        ranks : int
                The rank of linear
        drop_ih : float
                The dropout rate of input-to-hidden door
        drop_hh : float
                The dropout rate of hidden-to-hidden door
        """
        self.input_size = np.prod(in_shape)
        self.hidden_size = np.prod(hidden_shape)
        hidden_shape[0] *= 4
        tn_block = CPLinear(in_shape, hidden_shape, ranks)
        super(CPLSTM, self).__init__(self.input_size, tn_block, drop_ih, drop_hh)
        self.reset_ih()

    def reset_ih(self):
        """Reset parameters of input-to-hidden layer.
        """
        for weight in self.cell.weight_ih.parameters():
            nn.init.trunc_normal_(weight.data, 0., 0.5)

        if self.cell.weight_ih.bias is not None:
            nn.init.zeros_(self.cell.weight_ih.bias)
