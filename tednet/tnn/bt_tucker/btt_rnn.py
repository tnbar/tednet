# -*- coding: UTF-8 -*-

from ..tn_rnn import _TNLSTM

from .base import BTTLinear


class BTTLSTM(_TNLSTM):
    def __init__(self, input_size, hidden_size, drop_ih: float = 0.3, drop_hh: float = 0.35):
        self.input_size = input_size
        self.hidden_size = hidden_size
        tn_block = BTTLinear([64, 32], [4*32, 64], [6, 6], 2)
        super(BTTLSTM, self).__init__(hidden_size, tn_block, drop_ih, drop_hh)
