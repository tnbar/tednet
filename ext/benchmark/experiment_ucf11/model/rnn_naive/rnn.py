# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from collections import namedtuple

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.zeros(4 * hidden_size))
        self.bias_hh = Parameter(torch.zeros(4 * hidden_size))

        self.dropout_ih = nn.Dropout(0.35)
        self.dropout_hh = nn.Dropout(0.25)

        self.reset_parameters()

    def forward(self, input, state):
        hx, cx = state
        gate_ih = torch.mm(input, self.weight_ih.t()) + self.bias_ih
        gate_ih = self.dropout_ih(gate_ih)

        gate_hh = torch.mm(self.dropout_hh(hx), self.weight_hh.t())
        gates = gate_ih + gate_hh + self.bias_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight_ih, mean=0, std=0.3)

        nn.init.orthogonal_(self.weight_hh)


class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, x, state):
        x = x.unbind(0)
        outs = []
        for i in range(len(x)):
            out, state = self.cell(x[i], state)
            outs.append(out)

        outs = torch.stack(outs)

        return outs, state


def NaiveLSTM(input_size, hidden_size):
    return LSTMLayer(LSTMCell, input_size, hidden_size)
