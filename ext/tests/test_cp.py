# -*- coding: UTF-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath('../tednet'))

import tednet as tdt
import tednet.tnn.cp as cp

import torch


class Test_CP:
    def test_CPConv2D(self):
        print('\n CPConv2D')
        data = torch.Tensor(16, 1, 28, 28)
        model = cp.CPConv2D(1, 20, 6, 3)
        res = model(data)

        assert len(res.shape) == 4

    def test_CPLinear(self):
        print('\n CPLinear')
        data = torch.Tensor(16, 20)
        model = cp.CPLinear([4, 5], [2, 5], 6)
        res = model(data)

        assert len(res.shape) == 2

    def test_CPLeNet5(self):
        print('\n CPLeNet5')
        data = torch.Tensor(16, 1, 28, 28)
        model = cp.CPLeNet5(10, rs=[6, 6, 6, 6])
        res = model(data)

        assert len(res.shape) == 2

    def test_CPResNet20(self):
        print('\n CPResNet20')
        data = torch.Tensor(16, 3, 28, 28)
        model = cp.CPResNet20([6, 6, 6, 6, 6, 6, 6], 10)
        res = model(data)

        assert len(res.shape) == 2

    def test_CPResNet32(self):
        print('\n CPResNet32')
        data = torch.Tensor(16, 3, 28, 28)
        model = cp.CPResNet32([6, 6, 6, 6, 6, 6, 6], 10)
        res = model(data)

        assert len(res.shape) == 2

    def test_CPLSTM(self):
        print('\n CPLSTM')
        from collections import namedtuple
        LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

        data = torch.Tensor(10, 16, 6)
        model = cp.CPLSTM([2, 3], [3, 3], 6)

        state = LSTMState(torch.zeros(16, 9),
                          torch.zeros(16, 9))
        res, _ = model(data, state)

        assert len(res.shape) == 3


