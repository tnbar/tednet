# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from collections import namedtuple

__all__ = ["_TNLSTMCell", "_TNLSTM"]

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


class _TNLSTMCell(nn.Module):
    def __init__(self, hidden_size, tn_block, drop_ih = 0.3, drop_hh=0.35):
        super(_TNLSTMCell, self).__init__()
        # self.input_size = input_size
        # self.hidden_size = hidden_size

        self.weight_ih = tn_block
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        # self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.zeros(4 * hidden_size))

        # self.dropout = nn.Dropout(0.25)

        self.reset_hh()

        self.dropout_ih = nn.Dropout(drop_ih)
        self.dropout_hh = nn.Dropout(drop_hh)

    def reset_hh(self):
        nn.init.orthogonal_(self.weight_hh.data)

    def forward(self, input, state):
        hx, cx = state
        gate_ih = self.dropout_ih(input)
        gate_ih = self.weight_ih(gate_ih)

        # gate_hh = self.dropout_hh(torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        # gates = gate_ih + gate_hh

        gate_hh = torch.mm(self.dropout_hh(hx), self.weight_hh.t())
        gates = gate_ih + gate_hh + self.bias_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # ingate = hard_sigmoid(ingate)
        ingate = torch.sigmoid(ingate)
        # forgetgate = hard_sigmoid(forgetgate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        # outgate = hard_sigmoid(outgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        # hy = self.dropout(hy)

        return hy, (hy, cy)


class _TNLSTM(nn.Module):
    def __init__(self, hidden_size, tn_block, drop_ih = 0.3, drop_hh=0.35):
        super(_TNLSTM, self).__init__()
        self.cell = _TNLSTMCell(hidden_size, tn_block, drop_ih, drop_hh)

    def forward(self, x, state):
        x = x.unbind(0)
        for i in range(len(x)):
            out, state = self.cell(x[i], state)

        return state



