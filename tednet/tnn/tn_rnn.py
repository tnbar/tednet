# -*- coding: UTF-8 -*-

from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

__all__ = ["_TNLSTMCell", "_TNLSTM"]

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


class _TNLSTMCell(nn.Module):
    def __init__(self, hidden_size: int, tn_block, drop_ih=0.3, drop_hh=0.35):
        """Tensor LSTMCell.

        Parameters
        ----------
        hidden_size : int
                The hidden size of LSTMCell
        tn_block :
                The block class of input-to-hidden door
        drop_ih : float
                The dropout rate of input-to-hidden door
        drop_hh : float
                The dropout rate of hidden-to-hidden door
        """
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
        """Reset parameters of hidden-to-hidden layer.
        """
        nn.init.orthogonal_(self.weight_hh.data)

    def forward(self, inputs: torch.Tensor, state: LSTMState):
        """Forwarding method.
        LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

        Parameters
        ----------
        inputs : torch.Tensor
                 tensor :math:`\in \mathbb{R}^{b \\times C}`
        state : LSTMState
                 namedtuple: [hx :math:`\in \mathbb{R}^{H}`, cx :math:`\in \mathbb{R}^{H}`]

        Returns
        -------
        torch.Tensor, [torch.Tensor, torch.Tensor]
            result: hy :math:`\in \mathbb{R}^{H}`, [hy :math:`\in \mathbb{R}^{H}`, cy :math:`\in \mathbb{R}^{H}`]
        """
        hx, cx = state
        gate_ih = self.weight_ih(inputs)
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


class _TNLSTM(nn.Module):
    def __init__(self, hidden_size, tn_block, drop_ih = 0.3, drop_hh=0.35):
        """Tensor LSTM.

        Parameters
        ----------
        hidden_size : int
                The hidden size of LSTM
        tn_block :
                The block class of input-to-hidden door
        drop_ih : float
                The dropout rate of input-to-hidden door
        drop_hh : float
                The dropout rate of hidden-to-hidden door
        """
        super(_TNLSTM, self).__init__()
        self.cell = _TNLSTMCell(hidden_size, tn_block, drop_ih, drop_hh)

    def forward(self, inputs, state):
        """Forwarding method.
        LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

        Parameters
        ----------
        inputs : torch.Tensor
                 tensor :math:`\in \mathbb{R}^{S \\times b \\times C}`
        state : LSTMState
                 namedtuple: [hx :math:`\in \mathbb{R}^{H}`, cx :math:`\in \mathbb{R}^{H}`]

        Returns
        -------
        torch.Tensor, LSTMState
            tensor :math:`\in \mathbb{R}^{S \\times b \\times C'}`,
            LSTMState is a namedtuple: [hy :math:`\in \mathbb{R}^{H}`, cy :math:`\in \mathbb{R}^{H}`]
        """
        inputs = inputs.unbind(0)
        outs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outs.append(out)

        outs = torch.stack(outs)

        return outs, state



