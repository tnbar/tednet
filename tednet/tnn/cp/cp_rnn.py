# -*- coding: UTF-8 -*-


from ..tn_rnn import _TNLSTM

from .base import CPLinear


class CPLSTM(_TNLSTM):
    def __init__(self, input_size, hidden_size, drop_ih: float = 0.3, drop_hh: float = 0.35):
        """LSTM based on CANDECOMP/PARAFAC.

        Parameters
        ----------
        input_size : int
                The input size of LSTM
        hidden_size : int
                The hidden size of LSTM
        drop_ih : float
                The dropout rate of input-to-hidden door
        drop_hh : float
                The dropout rate of hidden-to-hidden door
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        tn_block = CPLinear([64, 32], [4*32, 64], 6)
        super(CPLSTM, self).__init__(hidden_size, tn_block, drop_ih, drop_hh)

